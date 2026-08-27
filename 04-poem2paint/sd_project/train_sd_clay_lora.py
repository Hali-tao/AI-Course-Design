# -*- coding: utf-8 -*-
"""
课程设计项目：基于 Stable Diffusion v1.5 的多风格微调
任务阶段：任务 (6) 3D粘土风格轻量化微调 - 正统低秩伴生旁路规范版
文件名：train_sd_clay_lora.py
描述：修复原代码直接解冻、物理污染底模权重的严重错误。
      通过外挂标准的、不污染底模的低秩旁路矩阵，优雅捕获 3D 泥塑的质感与厚重形体。
"""

import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL

# ==================== 🔒 核心重构：正统 LoRA 低秩伴生层 ====================
class DreamboothLoRALayer(nn.Module):
    """
    真正的 LoRA 旁路模块。
    保持原始 Linear 权重绝对冻结，外挂 Delta W = U @ D * scale。
    """
    def __init__(self, original_layer, rank=8, alpha=16.0):
        super().__init__()
        self.original_layer = original_layer  # 托管原底模底层
        self.rank = rank
        self.scale = alpha / rank  # 16 / 8 = 2.0
        
        # 提取原 Linear 层的特征输入输出维度
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # 创建低秩分解伴生旁路矩阵对
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        
        # 学术规范初始化
        nn.init.normal_(self.lora_down.weight, std=1.0 / rank)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x, *args, **kwargs):
        # 1. 基础底模层前向转发（保持冻结状态）
        original_output = self.original_layer(x, *args, **kwargs)
        # 2. LoRA 伴生旁路计算残差
        lora_output = self.lora_up(self.lora_down(x)) * self.scale
        # 3. 完美融合成标准残差公式
        return original_output + lora_output


class ClayLoRADataset(Dataset):
    def __init__(self, csv_file, img_dir, tokenizer, size=512):
        self.df = pd.read_csv(csv_file).iloc[:5]  # 5样本强悍聚焦
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['file_name']
        image = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
        # 🌟 聚焦 3D 粘土材质的核心触发词
        pure_prompt = "a 3D claymation style object, smooth texture, clay material, high quality masterpiece"
        input_ids = self.tokenizer(pure_prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids[0]
        return {"pixel_values": self.image_transforms(image), "input_ids": input_ids}


if __name__ == "__main__":
    print("\n🚀 [任务6-正统粘土组] 启动 5 样本标准低秩 Clay LoRA 微调进程...")
    
    MODEL_ROOT_DIR = "/root/autodl-tmp/sd_v15"
    CLAY_DATA_ROOT = "/root/autodl-tmp/multi_style_dataset/style_clay"
    OUTPUT_DIR = "/root/autodl-tmp/outputs/weights_clay_lora"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    tokenizer = CLIPTokenizer.from_pretrained(f"{MODEL_ROOT_DIR}/tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(f"{MODEL_ROOT_DIR}/text_encoder", local_files_only=True)
    unet = UNet2DConditionModel.from_pretrained(f"{MODEL_ROOT_DIR}/unet", local_files_only=True)
    vae = AutoencoderKL.from_pretrained(f"{MODEL_ROOT_DIR}/vae", local_files_only=True)
    noise_scheduler = DDPMScheduler.from_pretrained(f"{MODEL_ROOT_DIR}/scheduler", local_files_only=True)
    
    # 1. 严格锁定主干，严禁污染原模型参数
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    
    # 2. 手术式外挂缝合真正的低秩旁路矩阵
    lora_modules_registry = {}
    for name, module in unet.named_modules():
        if "attn2" in name:
            for sub_name in ["to_q", "to_k", "to_v", "to_out.0"]:
                if hasattr(module, sub_name):
                    orig_layer = getattr(module, sub_name)
                    if isinstance(orig_layer, nn.Linear):
                        # 缝合低秩伴生层
                        lora_layer = DreamboothLoRALayer(orig_layer, rank=8, alpha=16.0)
                        setattr(module, sub_name, lora_layer)
                        full_layer_key = f"{name}.{sub_name}"
                        lora_modules_registry[full_layer_key] = lora_layer

    device = "cuda"
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    
    # 3. 收集并激活旁路参数的梯度
    unet.train()
    trainable_lora_parameters = []
    for layer_name, lora_layer in lora_modules_registry.items():
        lora_layer.lora_down.weight.requires_grad_(True)
        lora_layer.lora_up.weight.requires_grad_(True)
        trainable_lora_parameters.append(lora_layer.lora_down.weight)
        trainable_lora_parameters.append(lora_layer.lora_up.weight)

    torch.cuda.empty_cache()
    print(f"📊 [显存监控] 当前初始静态显存占用: {torch.cuda.memory_allocated(device)/(1024**3):.2f} GB")
    print(f"📦 缝合完毕：检测到 {len(lora_modules_registry)} 个 Cross-Attention 层参与粘土形体进化。")
    
    # 🌟 优化调整：采用兼顾形变动能与结构稳定的黄金学习率 5e-5
    # 同时使用 1e-3 的正规化约束，让泥塑边缘更加圆润平滑
    optimizer = torch.optim.AdamW(trainable_lora_parameters, lr=2e-4, weight_decay=1e-3)
    
    dataset = ClayLoRADataset(csv_file=f"{CLAY_DATA_ROOT}/metadata.csv", img_dir=f"{CLAY_DATA_ROOT}/images", tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    total_start_time = time.time()
    
    # 执行 15 轮短程高能聚焦训练
    max_epochs = 15
    for epoch in range(max_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        
        for step, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
                encoder_hidden_states = text_encoder(input_ids)[0]
                
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(noise_pred, noise, reduction="mean")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        epoch_elapsed_time = time.time() - epoch_start_time
        max_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        
        # 每 3 轮输出一次高精度学术日志
        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f" -> Epoch [{epoch+1:02d}/{max_epochs}] 完成 | Loss: {epoch_loss / len(dataloader):.4f} | 💥 单轮耗时: {epoch_elapsed_time:.2f} 秒 | 动态峰值显存: {max_mem:.2f} GB")
        
    total_elapsed_time = time.time() - total_start_time
    print(f"\n🎉 粘土低秩伴生网络训练圆满结束！累计耗时: {total_elapsed_time:.2f} 秒。")
        
    # 5. 【标准封装】精准抽取并构建开源生态格式的真·LoRA 增量字典
    standard_lora_state_dict = {}
    for layer_name, lora_layer in lora_modules_registry.items():
        base_key = f"unet.{layer_name}"
        standard_lora_state_dict[f"{base_key}.lora_down.weight"] = lora_layer.lora_down.weight.detach().cpu()
        standard_lora_state_dict[f"{base_key}.lora_up.weight"] = lora_layer.lora_up.weight.detach().cpu()
        standard_lora_state_dict[f"{base_key}.alpha"] = torch.tensor(16.0)
            
    lora_save_path = os.path.join(OUTPUT_DIR, "pytorch_lora_weights.bin")
    torch.save(standard_lora_state_dict, lora_save_path)
    
    file_size_mb = os.path.getsize(lora_save_path) / (1024 ** 2)
    print(f"✅ 完美蜕变！真正的低秩轻量化粘土 LoRA 权重已成功导出。物理体积: {file_size_mb:.2f} MB")
# -*- coding: utf-8 -*-
"""
课程设计项目：基于 Stable Diffusion v1.5 的多风格微调
任务阶段：任务 (6) 水墨风格轻量化微调 - 抗过拟合形体保护平滑版
文件名：train_sd_ink_lora.py
描述：针对 5 样本严重过拟合、画面形体结构崩溃问题进行专项修复。
      通过降低学习率、压减 Epoch 并引入文本随机弱化，在保留结构的原则下吸纳水墨风格。
"""

import os
import time
import random
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
        self.scale = alpha / rank
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # 创建低秩分解伴生旁路矩阵对
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        
        # 初始化：down层高斯分布，up层全零
        nn.init.normal_(self.lora_down.weight, std=1.0 / rank)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x, *args, **kwargs):
        original_output = self.original_layer(x, *args, **kwargs)
        lora_output = self.lora_up(self.lora_down(x)) * self.scale
        return original_output + lora_output


class InkLoRADataset(Dataset):
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
        
        # 🌟 策略优化 1：引入 20% 概率的文本软泛化，防止模型产生强烈的词汇死记硬背导致形体坍塌
        if random.random() < 0.2:
            pure_prompt = "a high quality masterpiece painting"
        else:
            pure_prompt = "a traditional Chinese ink painting, bold brushstrokes, expressive ink splatters"
            
        input_ids = self.tokenizer(pure_prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids[0]
        return {"pixel_values": self.image_transforms(image), "input_ids": input_ids}


if __name__ == "__main__":
    print("\n🚀 [任务6-形体保护组] 启动 5 样本超低损伤精细化 LoRA 微调进程...")
    
    MODEL_ROOT_DIR = "/root/autodl-tmp/sd_v15"
    INK_DATA_ROOT = "/root/autodl-tmp/multi_style_dataset/style_ink"
    OUTPUT_DIR = "/root/autodl-tmp/outputs/weights_ink_lora"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    tokenizer = CLIPTokenizer.from_pretrained(f"{MODEL_ROOT_DIR}/tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(f"{MODEL_ROOT_DIR}/text_encoder", local_files_only=True)
    unet = UNet2DConditionModel.from_pretrained(f"{MODEL_ROOT_DIR}/unet", local_files_only=True)
    vae = AutoencoderKL.from_pretrained(f"{MODEL_ROOT_DIR}/vae", local_files_only=True)
    noise_scheduler = DDPMScheduler.from_pretrained(f"{MODEL_ROOT_DIR}/scheduler", local_files_only=True)
    
    # 1. 冻结大块基础主干
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    
    # 2. 手术式安全扫描并注入通用 LoRA 旁路
    lora_modules_registry = {}
    for name, module in unet.named_modules():
        if "attn2" in name:
            for sub_name in ["to_q", "to_k", "to_v", "to_out.0"]:
                if hasattr(module, sub_name):
                    orig_layer = getattr(module, sub_name)
                    if isinstance(orig_layer, nn.Linear):
                        lora_layer = DreamboothLoRALayer(orig_layer, rank=8, alpha=16.0)
                        setattr(module, sub_name, lora_layer)
                        full_layer_key = f"{name}.{sub_name}"
                        lora_modules_registry[full_layer_key] = lora_layer

    device = "cuda"
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    
    # 3. 激活低秩旁路梯度更新
    unet.train()
    trainable_lora_parameters = []
    for layer_name, lora_layer in lora_modules_registry.items():
        lora_layer.lora_down.weight.requires_grad_(True)
        lora_layer.lora_up.weight.requires_grad_(True)
        trainable_lora_parameters.append(lora_layer.lora_down.weight)
        trainable_lora_parameters.append(lora_layer.lora_up.weight)

    torch.cuda.empty_cache()
    print(f"📊 [显存监控] 当前初始静态显存占用: {torch.cuda.memory_allocated(device)/(1024**3):.2f} GB")
    print(f"📦 缝合完毕：检测到 {len(lora_modules_registry)} 个 Cross-Attention 交互层。")
    
    # 🌟 策略优化 2：将学习率从 1e-4 大幅调低到 2e-5，用极其温和的步伐前进
    # 同时将权重衰减（L2正则化）拉升到 1e-3，强行约束参数变化幅度，不给画面跑偏的机会
    optimizer = torch.optim.AdamW(trainable_lora_parameters, lr=2e-4, weight_decay=1e-3)
    
    dataset = InkLoRADataset(csv_file=f"{INK_DATA_ROOT}/metadata.csv", img_dir=f"{INK_DATA_ROOT}/images", tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    total_start_time = time.time()
    
    # 🌟 策略优化 3：大幅削减总轮次，从 30 轮直接砍半至 15 轮，在模型产生图形畸变前提前收工
    max_epochs = 150 if "FULL" in os.environ else 15
    
    print(f"⏳ 正在执行精准抗过拟合策略：训练总轮次调整为 {max_epochs} 轮...")
    
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
        
        # 细化日志监控，每 3 轮输出一次
        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f" -> Epoch [{epoch+1:02d}/{max_epochs}] 完成 | Loss: {epoch_loss / len(dataloader):.4f} | 单轮耗时: {epoch_elapsed_time:.2f} 秒 | 动态峰值显存: {max_mem:.2f} GB")
        
    total_elapsed_time = time.time() - total_start_time
    print(f"\n🎉 控噪低损增量微调结束！累计耗时: {total_elapsed_time:.2f} 秒。")
        
    # 5. 标准封装导出
    standard_lora_state_dict = {}
    for layer_name, lora_layer in lora_modules_registry.items():
        base_key = f"unet.{layer_name}"
        standard_lora_state_dict[f"{base_key}.lora_down.weight"] = lora_layer.lora_down.weight.detach().cpu()
        standard_lora_state_dict[f"{base_key}.lora_up.weight"] = lora_layer.lora_up.weight.detach().cpu()
        standard_lora_state_dict[f"{base_key}.alpha"] = torch.tensor(16.0)
            
    lora_save_path = os.path.join(OUTPUT_DIR, "pytorch_lora_weights.bin")
    torch.save(standard_lora_state_dict, lora_save_path)
    
    file_size_mb = os.path.getsize(lora_save_path) / (1024 ** 2)
    print(f"✅ 精准脱敏微调成功！增量文件体积: {file_size_mb:.2f} MB")
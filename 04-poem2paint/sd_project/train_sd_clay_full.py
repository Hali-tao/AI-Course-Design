# -*- coding: utf-8 -*-
"""
课程设计项目：基于 Stable Diffusion v1.5 的多风格微调
任务阶段：任务 (3) 自行寻找其他数据集进行模型的全量微调
文件名：train_sd_clay_full.py
描述：解冻 UNet 全局所有参数进行 Full Fine-Tuning，建立 3D 粘土风格全量 Baseline（已注入显存与耗时精准度量）。
"""

import os
import time  # 🌟 引入时间开销计算库
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL

class ClayFullDataset(Dataset):
    def __init__(self, csv_file, img_dir, tokenizer, size=512):
        # 强行聚焦前 5 张核心图片，实行极限小样本聚焦轰炸
        self.df = pd.read_csv(csv_file).iloc[:5] 
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
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.image_transforms(image)
        
        # 统一的高纯度风格触发词
        pure_prompt = "a 3D claymation style object, smooth texture, clay material"
        input_ids = self.tokenizer(pure_prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids[0] 
        return {"pixel_values": pixel_values, "input_ids": input_ids}

if __name__ == "__main__":
    print("\n🚀 [任务3-粘土全量] 启动 5 样本粘土风格全量参数微调 (Full Fine-Tuning)...")
    model_path = "/root/autodl-tmp/sd_v15"
    clay_data_root = "/root/autodl-tmp/multi_style_dataset/style_clay"
    
    tokenizer = CLIPTokenizer.from_pretrained(f"{model_path}/tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(f"{model_path}/text_encoder", local_files_only=True)
    unet = UNet2DConditionModel.from_pretrained(f"{model_path}/unet", local_files_only=True)
    vae = AutoencoderKL.from_pretrained(f"{model_path}/vae", local_files_only=True)
    noise_scheduler = DDPMScheduler.from_pretrained(f"{model_path}/scheduler", local_files_only=True)
    
    # 固化低参与度组件：冻结 VAE 和 文本编码器
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # 【核心逻辑】：彻底解冻 UNet 的全局所有参数，使其具备全局权重改写权
    unet.requires_grad_(True) 
    unet.train()
                        
    device = "cuda"
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    
    # 🌟 核心修改点 1：清理显存残余，度量初始静态显存占用
    if device == "cuda":
        torch.cuda.empty_cache()
        initial_static_memory = torch.cuda.memory_allocated(device) / (1024 ** 3)
        print(f"📊 [显存监控] 模型加载完成。当前初始静态显存占用: {initial_static_memory:.2f} GB")
    
    # 警告：由于全量微调参数量极大，学习率必须调小（从局部微调的 2e-4 降到 5e-6）以防梯度爆炸
    optimizer = torch.optim.AdamW(unet.parameters(), lr=5e-6)
    
    dataset = ClayFullDataset(csv_file=f"{clay_data_root}/metadata.csv", img_dir=f"{clay_data_root}/images", tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    print(f"📦 载入成功！当前训练集包含: {len(dataset)} 张核心图片，将进行全局权重洗牌。")
    
    # 🌟 核心修改点 2：初始化耗时计时器
    total_start_time = time.time()
    
    for epoch in range(15):
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
            
        # 🌟 核心修改点 3：精准捕捉反向传播时爆发的硬核「动态最大峰值显存」
        if device == "cuda":
            peak_dynamic_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            memory_str = f" | 💥 动态峰值显存: {peak_dynamic_memory:.2f} GB"
        else:
            memory_str = ""
            
        print(f" -> Epoch [{epoch+1:02d}/15] 完成 | 粘土全量微调 Loss: {epoch_loss / len(dataloader):.4f}{memory_str}")
        
    total_elapsed_time = time.time() - total_start_time
    print(f"\n🎉 粘土全量参数微调训练圆满结束！累计耗时: {total_elapsed_time:.2f} 秒。")
        
    # 保存全量微调后的完整 UNet 状态字典
    output_dir = "/root/autodl-tmp/outputs/weights_clay_full"
    os.makedirs(output_dir, exist_ok=True)
    
    target_save_path = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(unet.state_dict(), target_save_path)
    
    # 🌟 核心修改点 4：打印导出文件的物理体积大小
    file_size_gb = os.path.getsize(target_save_path) / (1024 ** 3)
    print(f"🎉 [全量导出成功] 完整的 UNet 风格化模型已保存。文件体积: {file_size_gb:.2f} GB")
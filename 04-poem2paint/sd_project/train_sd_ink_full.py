# -*- coding: utf-8 -*-
"""
课程设计项目：基于 Stable Diffusion v1.5 的多风格微调
任务阶段：任务 (2)
文件名：train_sd_ink_full.py
描述：纯 PyTorch 手写训练框架，执行 UNet 2D 全量通道核心微调 (Full Fine-Tuning)，引入显存与耗时精准度量。
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

class InkPoetryDataset(Dataset):
    def __init__(self, csv_path, images_dir, tokenizer_instance, image_size=512):
        self.dataframe = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.tokenizer = tokenizer_instance
        self.transform_pipeline = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), 
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image_name = self.dataframe.iloc[index]['file_name']
        full_image_path = os.path.join(self.images_dir, image_name)
        
        loaded_image = Image.open(full_image_path).convert("RGB")
        pixel_tensors = self.transform_pipeline(loaded_image)
        
        raw_prompt = self.dataframe.iloc[index]['text']
        tokenized_ids = self.tokenizer(
            raw_prompt, 
            padding="max_length", 
            max_length=self.tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt"
        ).input_ids[0]
        
        return {"pixel_values": pixel_tensors, "input_ids": tokenized_ids}


def run_full_tuning_pipeline():
    print("\n🔥 [任务 2] 启动：古诗转水墨画 · UNet 全量通道核心微调...")
    
    MODEL_ROOT_DIR = "/root/autodl-tmp/sd_v15"
    DATA_ROOT_DIR = "/root/autodl-tmp/multi_style_dataset/style_ink"
    WEIGHTS_OUTPUT_DIR = "/root/autodl-tmp/outputs/weights_ink_full"
    
    NUM_TRAIN_EPOCHS = 10
    LEARNING_RATE = 1e-5  
    
    # 1. 加载五大核心组件
    print("⚙️  正在从本地目录预加载预训练底模组件...")
    text_tokenizer = CLIPTokenizer.from_pretrained(f"{MODEL_ROOT_DIR}/tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(f"{MODEL_ROOT_DIR}/text_encoder", local_files_only=True)
    unet_network = UNet2DConditionModel.from_pretrained(f"{MODEL_ROOT_DIR}/unet", local_files_only=True)
    vae_autoencoder = AutoencoderKL.from_pretrained(f"{MODEL_ROOT_DIR}/vae", local_files_only=True)
    diffusion_scheduler = DDPMScheduler.from_pretrained(f"{MODEL_ROOT_DIR}/scheduler", local_files_only=True)
    
    # 2. 冻结非必要的辅助组件
    vae_autoencoder.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # 3. 开启整个 UNet 的全部参数梯度
    unet_network.train()
    all_param_names = []
    print("🔓 [全量机制激活] 解冻 UNet 神经网络内所有卷积、残差、自注意力层参数。")
    for name, param in unet_network.named_parameters():
        param.requires_grad = True
        all_param_names.append(name)
        
    computation_device = "cuda" if torch.cuda.is_available() else "cpu"
    vae_autoencoder.to(computation_device)
    text_encoder.to(computation_device)
    unet_network.to(computation_device)
    
    # 🌟 核心修改点 1：清理显存残余，度量初始静态显存占用
    if computation_device == "cuda":
        torch.cuda.empty_cache()
        initial_static_memory = torch.cuda.memory_allocated(computation_device) / (1024 ** 3)
        print(f"📊 [显存监控] 模型加载完成。当前初始静态显存占用: {initial_static_memory:.2f} GB")
    
    # 4. 构建全局参数优化器
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, unet_network.parameters()), lr=LEARNING_RATE)
    
    # 5. 建立数据流管道
    training_dataset = InkPoetryDataset(
        csv_path=f"{DATA_ROOT_DIR}/metadata.csv", 
        images_dir=f"{DATA_ROOT_DIR}/images", 
        tokenizer_instance=text_tokenizer
    )
    training_dataloader = DataLoader(training_dataset, batch_size=1, shuffle=True)
    print(f"📦 数据加载成功，水墨画样本总数: {len(training_dataset)} 张。")
    
    # 🌟 核心修改点 2：初始化耗时计时器
    total_start_time = time.time()
    
    # 6. 梯度迭代循环
    for epoch in range(NUM_TRAIN_EPOCHS):
        cumulative_loss = 0.0
        for step, data_batch in enumerate(training_dataloader):
            pixels = data_batch["pixel_values"].to(computation_device)
            tokens = data_batch["input_ids"].to(computation_device)
            
            with torch.no_grad():
                latent_embeddings = vae_autoencoder.encode(pixels).latent_dist.sample() * 0.18215
                conditional_hidden_states = text_encoder(tokens)[0]
                
            gaussian_noise = torch.randn_like(latent_embeddings)
            timesteps = torch.randint(0, diffusion_scheduler.config.num_train_timesteps, (latent_embeddings.shape[0],), device=computation_device).long()
            
            noisy_latents = diffusion_scheduler.add_noise(latent_embeddings, gaussian_noise, timesteps)
            predicted_noise_residual = unet_network(noisy_latents, timesteps, conditional_hidden_states).sample
            
            loss_value = F.mse_loss(predicted_noise_residual, gaussian_noise, reduction="mean")
            
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            cumulative_loss += loss_value.item()
            
        # 🌟 核心修改点 3：捕捉反向传播和梯度更新期间爆发的硬核「动态最大峰值显存」
        if computation_device == "cuda":
            peak_dynamic_memory = torch.cuda.max_memory_allocated(computation_device) / (1024 ** 3)
            memory_str = f" | 💥 动态峰值显存: {peak_dynamic_memory:.2f} GB"
        else:
            memory_str = ""
            
        print(f" -> Epoch [{epoch+1:02d}/{NUM_TRAIN_EPOCHS}] | 水墨全量微调 MSE Loss: {cumulative_loss / len(training_dataloader):.4f}{memory_str}")
        
    total_elapsed_time = time.time() - total_start_time
    print(f"\n🎉 全量参数微调训练圆满结束！累计耗时: {total_elapsed_time:.2f} 秒。")
        
    # 7. 全量导出改变后的字典状态
    os.makedirs(WEIGHTS_OUTPUT_DIR, exist_ok=True)
    incremental_weights_dict = {}
    full_unet_state_dict = unet_network.state_dict()
    for layer_name in all_param_names:
        if layer_name in full_unet_state_dict:
            incremental_weights_dict[layer_name] = full_unet_state_dict[layer_name].cpu()

    target_save_path = os.path.join(WEIGHTS_OUTPUT_DIR, "pytorch_model.bin")
    torch.save(incremental_weights_dict, target_save_path)
    
    # 🌟 核心修改点 4：打印导出文件的物理体积大小
    file_size_gb = os.path.getsize(target_save_path) / (1024 ** 3)
    print(f"🎉 [任务2成功闭环] 全量更新后的水墨风格矩阵已导出。文件体积: {file_size_gb:.2f} GB")

if __name__ == "__main__":
    run_full_tuning_pipeline()
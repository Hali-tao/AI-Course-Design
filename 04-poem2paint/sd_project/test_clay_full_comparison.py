# -*- coding: utf-8 -*-
"""
课程设计项目：基于 Stable Diffusion v1.5 的多风格微调
任务阶段：任务 (3) 补充实验 - 粘土全量微调推理验证
文件名：test_clay_full_comparison.py
"""

import os
import torch
from diffusers import StableDiffusionPipeline

def run_clay_full_inference():
    print("\n🔮 [任务 3 粘土全量验证] 启动全局全量微调前后效果横向对比...")
    
    MODEL_ROOT_DIR = "/root/autodl-tmp/sd_v15"
    CLAY_FULL_WEIGHTS = "/root/autodl-tmp/outputs/weights_clay_full/pytorch_model.bin"
    OUTPUT_DIR = "/root/autodl-tmp/outputs/comparison_clay_full"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    negative_prompt_string = "blurry, low quality, distorted, photorealistic, metal gloss, real human, messy"
    RANDOM_SEED = 2048 
    
    # 测试两组不同的实体
    test_prompts_dict = {
        "ironman": "a 3D claymation style Iron Man superhero, smooth clay texture, vibrant colors, solid background",
        "fox": "a 3D claymation style cute little fox figurine, soft clay material, handcrafted texture, studio lighting"
    }
    
    # 1. 建立基线：底模出图
    print("\n📦 步骤 1: 使用纯净原生底模建立 Baseline...")
    pipeline = StableDiffusionPipeline.from_pretrained(MODEL_ROOT_DIR, torch_dtype=torch.float32, local_files_only=True).to("cuda")
    pipeline.unet.eval()
    
    for key, text in test_prompts_dict.items():
        generator = torch.Generator(device="cuda").manual_seed(RANDOM_SEED)
        with torch.no_grad():
            img = pipeline(prompt=text, negative_prompt=negative_prompt_string, num_inference_steps=35, guidance_scale=7.5, generator=generator).images[0]
        img.save(os.path.join(OUTPUT_DIR, f"base_{key}.png"))
        
    # 2. 载入任务(3)全量微调的整个 UNet 权重进行覆盖
    print("\n🔥 步骤 2: 强行载入全量微调的整个 UNet 权重进行物理替换...")
    if os.path.exists(CLAY_FULL_WEIGHTS):
        pipeline.unet.load_state_dict(torch.load(CLAY_FULL_WEIGHTS, map_location="cuda"))
        print("💡 UNet 骨架全部参数全局重构完毕！")
    else:
        print(f"❌ 错误：未在 {CLAY_FULL_WEIGHTS} 找到全量微调文件！")
        return
        
    pipeline.unet.eval()
    
    # 3. 全量微调模型出图
    for key, text in test_prompts_dict.items():
        generator = torch.Generator(device="cuda").manual_seed(RANDOM_SEED)
        with torch.no_grad():
            img_tuned = pipeline(prompt=text, negative_prompt=negative_prompt_string, num_inference_steps=35, guidance_scale=7.5, generator=generator).images[0]
        img_tuned.save(os.path.join(OUTPUT_DIR, f"clay_full_tuned_{key}.png"))
        
    print(f"\n🎉 任务(3)粘土全量对比图已全部生成在: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_clay_full_inference()
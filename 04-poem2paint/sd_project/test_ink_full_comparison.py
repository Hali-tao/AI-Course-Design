# -*- coding: utf-8 -*-
"""
课程设计项目：基于 Stable Diffusion v1.5 的多风格微调
任务阶段：任务 (2) 后续效果多样本横向对比验证
文件名：test_ink_comparison.py
描述：在一个脚本内自动遍历“写意奔马”与“山水孤舟”双提示词，一键导出微调前后的多组矩阵对比图。
"""

import os
import torch
from diffusers import StableDiffusionPipeline

def run_multi_prompt_comparison():
    print("\n🔮 [任务 2 推理验证] 启动古诗转水墨画【多提示词联合测试】...")
    
    # 1. 路径与核心配置
    MODEL_ROOT_DIR = "/root/autodl-tmp/sd_v15"
    INK_WEIGHTS_PATH = "/root/autodl-tmp/outputs/weights_ink_full/pytorch_model.bin"
    OUTPUT_DIR = "/root/autodl-tmp/outputs/comparison_ink"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 通用严苛负向提示词
    negative_prompt_string = "modern, western style, oil painting, photorealistic, 3D render, colorful, neon, low quality"
    RANDOM_SEED = 1024
    
    # 2. 核心测试字典：一键打包你的两个核心提示词，配置专属小文件名防止缓存欺骗
    test_prompts_dict = {
        "horse": "a traditional Chinese ink painting of a galloping horse, bold brushstrokes, expressive ink splatters, minimalism, ancient poem spirit",
        "boat": "a traditional Chinese ink painting of a single small boat on a quiet river, misty mountains background, ink wash, ancient poem landscape"
    }
    
    # ------------------ 【第一阶段：生成底模 Baseline 数据】 ------------------
    print("\n📦 [Stage 1] 正在加载纯净原生底模以建立 Baseline...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        MODEL_ROOT_DIR, 
        torch_dtype=torch.float32, 
        local_files_only=True
    ).to("cuda")
    pipeline.unet.eval()
    
    # 循环遍历字典，用底模把马和船各画一遍
    for key_name, prompt_text in test_prompts_dict.items():
        print(f"🎨 底模绘制中 -> 目标主题: [{key_name.upper()}]")
        generator = torch.Generator(device="cuda").manual_seed(RANDOM_SEED)
        
        with torch.no_grad():
            img = pipeline(
                prompt=prompt_text,
                negative_prompt=negative_prompt_string,
                num_inference_steps=35,
                guidance_scale=7.5, # 适当折中底模的引导系数
                generator=generator
            ).images[0]
            
        save_path = os.path.join(OUTPUT_DIR, f"base_{key_name}.png")
        img.save(save_path)
        print(f"   => 成功留存底模原图: {save_path}")

    # ------------------ 【第二阶段：合流微调权重并二次出图】 ------------------
    print("\n🔥 [Stage 2] 正在将任务(2)全量微调的水墨风格矩阵注入 UNet 骨架...")
    if os.path.exists(INK_WEIGHTS_PATH):
        trained_weights = torch.load(INK_WEIGHTS_PATH, map_location="cuda")
        pipeline.unet.load_state_dict(trained_weights, strict=False)
        print("💡 成功合流全量水墨风格特征矩阵！")
    else:
        print(f"❌ 错误：在 {INK_WEIGHTS_PATH} 未找到微调权重，请先跑通训练！")
        return
        
    pipeline.unet.eval()
    
    # 再次循环遍历字典，用微调后的水墨模型把马和船重新画一遍
    for key_name, prompt_text in test_prompts_dict.items():
        print(f"🎨 水墨模型绘制中 -> 目标主题: [{key_name.upper()}]")
        # 严格重置完全相同的随机种子，确保构图骨架可比
        generator = torch.Generator(device="cuda").manual_seed(RANDOM_SEED)
        
        with torch.no_grad():
            img_tuned = pipeline(
                prompt=prompt_text,
                negative_prompt=negative_prompt_string,
                num_inference_steps=35,
                guidance_scale=7.5,
                generator=generator
            ).images[0]
            
        save_path_tuned = os.path.join(OUTPUT_DIR, f"ink_tuned_{key_name}.png")
        img_tuned.save(save_path_tuned)
        print(f"   => 成功留存微调风格图: {save_path_tuned}")
        
    print(f"\n🎉 [多样本闭环] 全部图像生成完毕！请前往 {OUTPUT_DIR} 查看两组两两对比矩阵。")

if __name__ == "__main__":
    run_multi_prompt_comparison()
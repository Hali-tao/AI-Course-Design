# -*- coding: utf-8 -*-
"""
课程设计项目：基于 Stable Diffusion v1.5 的多风格微调
任务阶段：任务 (5) 推理阶段超参数多维矩阵消融对照实验
文件名：test_param_optimization.py
描述：通过 3x3 矩阵式网格搜索 (Grid Search)，横向对比 CFG 与 Steps 对全量微调模型的影响，并自动拼图。
"""

import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont

def run_matrix_param_ablation():
    print("\n🔮 [任务 5 高级进阶] 启动 3x3 超参数网格消融矩阵生成...")
    
    # 请根据你实际扩容后的路径对齐（这里默认为扩容后的本地路径，可根据实际需要修改）
    MODEL_ROOT_DIR = "/root/autodl-tmp/sd_v15"
    INK_WEIGHTS_PATH = "/root/autodl-tmp/outputs/weights_ink_full/pytorch_model.bin"
    OUTPUT_DIR = "/root/autodl-tmp/outputs/parameter_study"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    test_prompt = "a traditional Chinese ink painting of a single small boat on a quiet river, misty mountains background, ink wash, ancient poem landscape"
    negative_prompt = "modern, western style, oil painting, photorealistic, 3D render, colorful, blurry, deformed"
    RANDOM_SEED = 1024
    
    if not os.path.exists(INK_WEIGHTS_PATH):
        print(f"❌ 错误：未在 {INK_WEIGHTS_PATH} 找到任务2的全量权重，请检查路径！")
        return
        
    print("📦 正在加载全量微调模型...")
    pipeline = StableDiffusionPipeline.from_pretrained(MODEL_ROOT_DIR, torch_dtype=torch.float16, local_files_only=True).to("cuda")
    pipeline.unet.load_state_dict(torch.load(INK_WEIGHTS_PATH, map_location="cuda"))
    pipeline.unet.eval()
    
    # 🌟 定义 3x3 网格的搜索空间
    cfg_list = [4.5, 7.0, 9.5]    # 纵轴：低引导、标准引导、高强度强刷
    steps_list = [20, 35, 50]   # 横轴：低步数欠采样、标准采样、高步数精细采样
    
    grid_images = [] # 用于存放生成的 9 张图
    
    print("🎨 开始矩阵式渲染（总计 9 场渲染），请耐心等待...")
    for cfg in cfg_list:
        for steps in steps_list:
            print(f" ──> 正在测试组合: CFG = {cfg} | Steps = {steps} ...")
            generator = torch.Generator(device="cuda").manual_seed(RANDOM_SEED)
            
            with torch.no_grad():
                # 使用 float16 提速并节省显存
                img = pipeline(
                    prompt=test_prompt, 
                    negative_prompt=negative_prompt, 
                    num_inference_steps=steps, 
                    guidance_scale=cfg, 
                    generator=generator
                ).images[0]
            
            # 🖼️ 在每张子图的左上角用 PIL 简单绘制参数水印，防止混淆
            draw = ImageDraw.Draw(img)
            label_text = f"CFG:{cfg} S:{steps}"
            # 创建带有半透明背景的标签框，确保字迹清晰
            draw.rectangle([(5, 5), (140, 25)], fill="black")
            draw.text((10, 8), label_text, fill="white")
            
            grid_images.append(img)
            
    # 📐 自动拼接成 3x3 的大图画布
    print("\n🧩 渲染完毕，正在执行图像矩阵物理拼接...")
    single_w, single_h = grid_images[0].size  # 512x512
    matrix_img = Image.new('RGB', (single_w * 3, single_h * 3))
    
    idx = 0
    for row in range(3):
        for col in range(3):
            matrix_img.paste(grid_images[idx], (col * single_w, row * single_h))
            idx += 1
            
    # 保存大图和单图
    matrix_save_path = os.path.join(OUTPUT_DIR, "parameter_grid_search_3x3.png")
    matrix_img.save(matrix_save_path)
    
    print(f"🎉 [任务5完胜闭环] 3x3 超参数消融矩阵大图已成功导出至: {matrix_save_path}")

if __name__ == "__main__":
    run_matrix_param_ablation()
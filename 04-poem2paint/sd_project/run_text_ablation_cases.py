# -*- coding: utf-8 -*-
"""
课程设计项目：基于 Stable Diffusion v1.5 的多风格微调
任务阶段：第6部分 双案例多模态文本语义消融实验
文件名：run_text_ablation_cases.py
"""

import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont

# 1. 基础配置
model_dir = "/root/autodl-tmp/sd_v15"
output_dir = "/root/autodl-tmp/outputs/text_ablation"
os.makedirs(output_dir, exist_ok=True)

# 2. 定义双案例测试矩阵 (A: 中文直输 | B: 机器直译 | C: LLM解耦重构)
test_matrix = {
    "Case_1_JiangXue": {
        "A_Chinese": "孤舟蓑笠翁，独钓寒江雪",
        "B_Literal": "An old man in a straw raincoat and bamboo hat on a lonely boat, fishing alone in the cold river snow.",
        "C_LLM_Ours": "A traditional Chinese ink painting, a single small wooden boat on a vast quiet river, a lonely old fisherman wearing a straw raincoat, misty winter background, heavy snow falling, atmospheric, ink wash style, minimalism."
    },
    "Case_2_BaiDiCheng": {
        "A_Chinese": "两岸猿声啼不住，轻舟已过万重山",
        "B_Literal": "The monkeys on both banks keep crying, and the light boat has passed ten thousand mountains.",
        "C_LLM_Ours": "Traditional Chinese ink wash illustration, a tiny fast-moving wooden boat speeding through a narrow river canyon, majestic and towering mountains covered in mist, kinetic brushstrokes, expressive ink splatters, sense of high speed and relief."
    }
}
negative_prompt = "modern, western style, oil painting, photorealistic, 3D render, colorful, blurry"

# 3. 加载干净底模
print("⚙️ 正在加载原生底模以评估文本编码器...")
pipe = StableDiffusionPipeline.from_pretrained(model_dir, torch_dtype=torch.float32, local_files_only=True).to("cuda")
pipe.unet.eval()

# 4. 循环渲染整个 2x3 矩阵
w, h = 512, 512
canvas = Image.new('RGB', (w * 3, h * 2))

row_idx = 0
for case_name, prompts in test_matrix.items():
    col_idx = 0
    for mode_name, text_prompt in prompts.items():
        print(f"🚀 正在渲染 [{case_name}] ── 分支 [{mode_name}]...")
        
        # 严格锁定物理种子 42，保证每次生成的底层潜变量完全一致
        generator = torch.Generator("cuda").manual_seed(42)
        
        img = pipe(
            prompt=text_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=35,
            guidance_scale=7.5,
            generator=generator
        ).images[0]
        
        # 将图片安全保存备份
        img.save(os.path.join(output_dir, f"{case_name}_{mode_name}.png"))
        
        # 将单图拼入 2x3 大画布对应位置
        canvas.paste(img, (col_idx * w, row_idx * h))
        col_idx += 1
    row_idx += 1

# 5. 保存终极看板
canvas_path = os.path.join(output_dir, "text_encoder_cases_2x3.png")
canvas.save(canvas_path)
print(f"🎉 实验大获成功！双案例终极对比看板已保存至: {canvas_path}")
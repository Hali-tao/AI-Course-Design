# -*- coding: utf-8 -*-
"""
课程设计项目：基于 Stable Diffusion v1.5 的多风格微调
任务阶段：任务( task_6 ) 水墨孤舟 (ink_boat) 2x5 阶梯消融矩阵生成
文件名：test_ink_boat_matrix.py
"""

import os
import gc
import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw

# ==================== 🔒 核心重构：LoRA 伴生层 ====================
class DreamboothLoRALayer(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16.0):
        super().__init__()
        self.original_layer = original_layer  
        self.rank = rank
        self.base_scale = alpha / rank  # 16 / 8 = 2.0
        self.current_scale = 1.0  
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)

    def forward(self, x, *args, **kwargs):
        original_output = self.original_layer(x, *args, **kwargs)
        lora_output = self.lora_up(self.lora_down(x)) * (self.base_scale * self.current_scale)
        return original_output + lora_output


def generate_ink_boat_matrix(model_dir, lora_path, output_dir):
    print("=" * 70)
    print("🚀 启动水墨孤舟 [ink_boat] 2x5 紧凑消融矩阵渲染引擎...")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 🎯 依然使用极具水墨留白特性的提示词
    prompt = "a traditional Chinese ink painting of a single small boat on a quiet river, misty mountains background, ink wash, ancient poem landscape, high quality masterpiece"
    neg_prompt = "chaotic black blocks, severe deformation, modern structures, realistic reflection, colorful, oil painting, ugly, blurry"
    
    seed = 42
    steps = 35
    cfg = 7.5
    img_size = 512  # 单张图分辨率
    
    # ⏳ 0.0 到 0.9 共 10 张图
    test_scales = [round(x * 0.1, 1) for x in range(10)]
    rendered_images = []
    
    if not os.path.exists(lora_path):
        print(f"❌ 错误：未在 {lora_path} 找到权重文件！")
        return
    raw_lora_state_dict = torch.load(lora_path, map_location="cpu")

    for scale in test_scales:
        print(f"[串行渲染] 正在捕获 Scale = {scale} 的画面形态...")
        
        pipe = StableDiffusionPipeline.from_pretrained(
            model_dir, torch_dtype=torch.float32, local_files_only=True
        ).to("cuda")
        pipe.unet.eval()
        
        for name, module in pipe.unet.named_modules():
            if "attn2" in name:
                for sub_name in ["to_q", "to_k", "to_v", "to_out.0"]:
                    if hasattr(module, sub_name):
                        orig_layer = getattr(module, sub_name)
                        if isinstance(orig_layer, nn.Linear):
                            lora_layer = DreamboothLoRALayer(orig_layer, rank=8, alpha=16.0)
                            lora_layer.current_scale = scale
                            lora_layer.to("cuda")
                            setattr(module, sub_name, lora_layer)
        
        if scale > 0.0:
            target_load_dict = {}
            for k, v in raw_lora_state_dict.items():
                if "weight" in k:
                    target_load_dict[k] = v.to(device="cuda", dtype=torch.float32)
            pipe.unet.load_state_dict(target_load_dict, strict=False)
            
        generator = torch.Generator(device="cuda").manual_seed(seed)
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                negative_prompt=neg_prompt,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=generator
            ).images[0]
            
        # 在每张图底部绘制高对比度的黑条和白字标签
        draw = ImageDraw.Draw(image)
        draw.rectangle([(0, img_size - 40), (img_size, img_size)], fill="black")
        draw.text((15, img_size - 30), f"Scale: {scale:.1f}", fill="white", stroke_width=1)
        
        rendered_images.append(image)
        
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    # ==================== 📦 核心拼接：2x5 看板矩阵合成 ====================
    print("\n📊 正在进行 2x5 矩阵看板几何卡合拼接...")
    
    # 计算大图的总宽高：横向 5 张，纵向 2 张
    total_width = img_size * 5
    total_height = img_size * 2
    
    # 创建空白大画布
    matrix_canvas = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    
    # 双层逻辑定位插值
    for idx, img in enumerate(rendered_images):
        if idx < 5:
            # 第一行：Scale 0.0 ~ 0.4
            x_pos = idx * img_size
            y_pos = 0
        else:
            # 第二行：Scale 0.5 ~ 0.9
            x_pos = (idx - 5) * img_size
            y_pos = img_size
            
        matrix_canvas.paste(img, (x_pos, y_pos))
        
    # 保存结果大图
    matrix_save_path = os.path.join(output_dir, "ink_boat_ablation_matrix_2x5.png")
    matrix_canvas.save(matrix_save_path)
    
    print("\n" + "=" * 70)
    print(f"🎉 [2x5 矩阵看板合成成功] 阶梯消融比对大图已无损导出！")
    print(f"📂 终极报告图片查看路径: {matrix_save_path}")
    print("=" * 70)


if __name__ == "__main__":
    MODEL_ROOT_DIR = "/root/autodl-tmp/sd_v15"
    INK_LORA_PATH = "/root/autodl-tmp/outputs/weights_ink_lora/pytorch_lora_weights.bin"
    TEST_OUTPUT_DIR = "/root/autodl-tmp/outputs/test_ink_lora_report"
    
    generate_ink_boat_matrix(MODEL_ROOT_DIR, INK_LORA_PATH, TEST_OUTPUT_DIR)
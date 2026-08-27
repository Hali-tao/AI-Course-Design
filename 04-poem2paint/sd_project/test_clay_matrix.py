# -*- coding: utf-8 -*-
"""
课程设计项目：基于 Stable Diffusion v1.5 的多风格微调
任务阶段：任务( task_6 ) 3D粘土钢铁侠 (clay_ironman) 2x5 阶梯消融矩阵生成
文件名：test_clay_matrix.py
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


def generate_clay_matrix(model_dir, lora_path, output_dir):
    print("=" * 70)
    print("🚀 启动 3D 粘土钢铁侠 [clay_ironman] 2x5 矩阵消融渲染引擎...")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 🎯 针对泥塑质感优化的正面与负面提示词
    prompt = "a 3D claymation style Iron Man superhero, smooth clay texture, vibrant colors, solid background, high quality masterpiece"
    neg_prompt = "photorealistic metal, shiny armor, metallic reflections, chaotic plastic blocks, blurry, low quality, deformed anatomy"
    
    seed = 42
    steps = 35
    cfg = 7.5
    img_size = 512  # 单张图分辨率
    
    # ⏳ 0.0 到 0.9 共 10 张图梯度
    test_scales = [round(x * 0.1, 1) for x in range(10)]
    rendered_images = []
    
    if not os.path.exists(lora_path):
        print(f"❌ 错误：未在 {lora_path} 找到权重文件！请确认 train_sd_clay_lora.py 已运行完毕。")
        return
    raw_lora_state_dict = torch.load(lora_path, map_location="cpu")

    for scale in test_scales:
        print(f"[串行渲染] 正在捕获 Scale = {scale} 的材质演变...")
        
        # 1. 每次加载干净底模
        pipe = StableDiffusionPipeline.from_pretrained(
            model_dir, torch_dtype=torch.float32, local_files_only=True
        ).to("cuda")
        pipe.unet.eval()
        
        # 2. 精准外挂伴生层
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
        
        # 3. 灌注粘土 LoRA 权重
        if scale > 0.0:
            target_load_dict = {}
            for k, v in raw_lora_state_dict.items():
                if "weight" in k:
                    target_load_dict[k] = v.to(device="cuda", dtype=torch.float32)
            pipe.unet.load_state_dict(target_load_dict, strict=False)
            
        # 4. 锁定种子渲染
        generator = torch.Generator(device="cuda").manual_seed(seed)
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                negative_prompt=neg_prompt,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=generator
            ).images[0]
            
        # 5. 在每张图底部绘制高对比度的黑条和白字标签
        draw = ImageDraw.Draw(image)
        draw.rectangle([(0, img_size - 40), (img_size, img_size)], fill="black")
        draw.text((15, img_size - 30), f"Scale: {scale:.1f}", fill="white", stroke_width=1)
        
        rendered_images.append(image)
        
        # 6. 垃圾回收
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    # ==================== 📦 核心拼接：2x5 看板矩阵合成 ====================
    print("\n📊 正在进行 2x5 粘土矩阵几何拼接...")
    
    total_width = img_size * 5
    total_height = img_size * 2
    
    matrix_canvas = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    
    for idx, img in enumerate(rendered_images):
        if idx < 5:
            x_pos = idx * img_size
            y_pos = 0
        else:
            x_pos = (idx - 5) * img_size
            y_pos = img_size
            
        matrix_canvas.paste(img, (x_pos, y_pos))
        
    # 保存结果大图
    matrix_save_path = os.path.join(output_dir, "clay_ironman_ablation_matrix_2x5.png")
    matrix_canvas.save(matrix_save_path)
    
    print("\n" + "=" * 70)
    print(f"🎉 [2x5 粘土矩阵看板合成成功] 阶梯消融比对大图已无损导出！")
    print(f"📂 终极报告图片查看路径: {matrix_save_path}")
    print("=" * 70)


if __name__ == "__main__":
    MODEL_ROOT_DIR = "/root/autodl-tmp/sd_v15"
    CLAY_LORA_PATH = "/root/autodl-tmp/outputs/weights_clay_lora/pytorch_lora_weights.bin"
    TEST_OUTPUT_DIR = "/root/autodl-tmp/outputs/test_clay_report"
    
    generate_clay_matrix(MODEL_ROOT_DIR, CLAY_LORA_PATH, TEST_OUTPUT_DIR)
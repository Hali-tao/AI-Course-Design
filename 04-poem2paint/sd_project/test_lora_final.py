# -*- coding: utf-8 -*-
"""
课程设计项目：基于 Stable Diffusion v1.5 的多风格微调
任务阶段：任务(6) 进程隔离渲染总控 (Base -> Full -> LoRA) ── 修复花括号转义版
文件名：run_separated_render.py
"""

import os

# 创建独立的临时缓存文件夹
CACHE_DIR = "/root/autodl-tmp/outputs/ablation_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ─── 核心定义：独立进程脚本模板 (所有原生 Python 花括号均已双写转义) ───
RENDER_TEMPLATE = """
import os
import torch
from diffusers import StableDiffusionPipeline

model_dir = "/root/autodl-tmp/sd_v15"
cache_dir = "{cache_dir}"
mode = "{mode}"  # base, full, lora

# 提示词完全对齐
ink_neg = "chaotic black blocks, severe deformation, modern structures, realistic reflection, colorful, oil painting, ugly, blurry"
prompt_horse = "a traditional Chinese ink painting of a galloping horse, bold brushstrokes, expressive ink splatters"
prompt_boat = "a traditional Chinese ink painting of a single small boat on a quiet river, misty mountains background, ink wash, ancient poem landscape, high quality masterpiece"

clay_neg = "blurry, low quality, distorted, metal gloss, real human, realistic photochromatic, sketch"
prompt_ironman = "a 3D claymation style Iron Man superhero, smooth clay texture, vibrant colors, solid background"
prompt_fox = "a 3D claymation style cute little fox figurine, soft clay material, handcrafted texture, studio lighting"

# 1. 物理加载干净底模
pipe = StableDiffusionPipeline.from_pretrained(model_dir, torch_dtype=torch.float32, local_files_only=True).to("cuda")
pipe.unet.eval()

# 2. 根据不同模式进行地基注入
if mode == "full":
    full_path = "{full_path}"
    if os.path.exists(full_path):
        pipe.unet.load_state_dict(torch.load(full_path, map_location="cuda"), strict=False)
elif mode == "lora":
    lora_path = "{lora_path}"
    if os.path.exists(lora_path):
        import torch.nn as nn
        class DreamboothLoRALayer(nn.Module):
            def __init__(self, original_layer):
                super().__init__()
                self.original_layer = original_layer  
                self.rank = 8
                self.base_scale = 16.0 / 8
                self.current_scale = 0.3  # 严格锁定黄金点
                self.lora_down = nn.Linear(original_layer.in_features, 8, bias=False)
                self.lora_up = nn.Linear(8, original_layer.out_features, bias=False)
            def forward(self, x, *args, **kwargs):
                return self.original_layer(x, *args, **kwargs) + self.lora_up(self.lora_down(x)) * (self.base_scale * self.current_scale)

        for name, module in pipe.unet.named_modules():
            if "attn2" in name:
                for sub_name in ["to_q", "to_k", "to_v", "to_out.0"]:
                    if hasattr(module, sub_name):
                        orig_layer = getattr(module, sub_name)
                        if isinstance(orig_layer, nn.Linear):
                            lora_layer = DreamboothLoRALayer(orig_layer).to("cuda")
                            setattr(module, sub_name, lora_layer)
        
        raw_dict = torch.load(lora_path, map_location="cuda")
        # 🌟 关键修复：用双花括号转义字典推导式，防止被 .format() 误解析
        target_dict = {{k: v.to("cuda", dtype=torch.float32) for k, v in raw_dict.items() if "weight" in k}}
        pipe.unet.load_state_dict(target_dict, strict=False)

# 3. 严格锁定种子生成
def save_img(p, n, s, c, name):
    gen = torch.Generator("cuda").manual_seed(s)
    img = pipe(prompt=p, negative_prompt=n, num_inference_steps=35 if "ink" in name else 30, guidance_scale=c, generator=gen).images[0]
    img.save(os.path.join(cache_dir, name))

# 🌟 关键修复：修复这里的 f-string 嵌套花括号
save_img(prompt_horse, ink_neg, 40, 7.5, f"ink_horse_{{mode}}.png")
save_img(prompt_boat, ink_neg, 40, 7.5, f"ink_boat_{{mode}}.png")
save_img(prompt_ironman, clay_neg, 45, 8.5, f"clay_ironman_{{mode}}.png")
save_img(prompt_fox, clay_neg, 45, 8.5, f"clay_fox_{{mode}}.png")
print(f"✅ 独立进程模式 [{{mode}}] 渲染完成并已完全退出。")
"""

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 启动学术级多进程物理隔离渲染管线 (花括号Bug修复版)...")
    print("=" * 70)
    
    # 路径配置
    ink_full = "/root/autodl-tmp/outputs/weights_ink_full/pytorch_model.bin"
    ink_lora = "/root/autodl-tmp/outputs/weights_ink_lora/pytorch_lora_weights.bin"
    clay_full = "/root/autodl-tmp/outputs/weights_clay_full/pytorch_model.bin"
    clay_lora = "/root/autodl-tmp/outputs/weights_clay_lora/pytorch_lora_weights.bin"

    # ─── 进程 1：纯净 Base 渲染 ───
    # 此时不需要加载任何权重，直接跑原底模
    with open("temp_base.py", "w") as f:
        f.write(RENDER_TEMPLATE.format(mode="base", cache_dir=CACHE_DIR, full_path="", lora_path=""))
    os.system("python temp_base.py && rm temp_base.py")

    # ─── 进程 2：Full 全量渲染 (水墨) ───
    with open("temp_ink_full.py", "w") as f:
        f.write(RENDER_TEMPLATE.format(mode="full", cache_dir=CACHE_DIR, full_path=ink_full, lora_path=""))
    # 过滤掉粘土的生成，防止水墨全量模型画出畸变粘土
    os.system("sed -i '/clay_/d' temp_ink_full.py") 
    os.system("python temp_ink_full.py && rm temp_ink_full.py")

    # ─── 进程 3：LoRA 黄金点渲染 (水墨) ───
    with open("temp_ink_lora.py", "w") as f:
        f.write(RENDER_TEMPLATE.format(mode="lora", cache_dir=CACHE_DIR, full_path="", lora_path=ink_lora))
    os.system("sed -i '/clay_/d' temp_ink_lora.py")
    os.system("python temp_ink_lora.py && rm temp_ink_lora.py")
    
    # ─── 进程 4：Full 全量渲染 (粘土) ───
    with open("temp_clay_full.py", "w") as f:
        f.write(RENDER_TEMPLATE.format(mode="full", cache_dir=CACHE_DIR, full_path=clay_full, lora_path=""))
    # 过滤掉水墨的生成，防止粘土全量模型画出畸变水墨
    os.system("sed -i '/ink_/d' temp_clay_full.py")
    os.system("python temp_clay_full.py && rm temp_clay_full.py")
    
    # ─── 进程 5：LoRA 黄金点渲染 (粘土) ───
    with open("temp_clay_lora.py", "w") as f:
        f.write(RENDER_TEMPLATE.format(mode="lora", cache_dir=CACHE_DIR, full_path="", lora_path=clay_lora))
    os.system("sed -i '/ink_/d' temp_clay_lora.py")
    os.system("python temp_clay_lora.py && rm temp_clay_lora.py")

    print("\n🎉 [所有风格独立进程隔离渲染全部顺利结束！缓存单图已安全落盘。]")
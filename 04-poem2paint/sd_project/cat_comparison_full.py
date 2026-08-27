# -*- coding: utf-8 -*-
"""
课程设计项目：微调效果矩阵拼接脚本
描述：将水墨组和粘土组的微调前后对比图分别合并为 2x2 的可视化网格图，便于报告排版。
"""

import os
from PIL import Image, ImageDraw, ImageFont

def create_image_grid(image_paths, titles, output_path, size=512):
    """
    将4张图拼接为 2x2 矩阵，并在图片上方绘制子标题
    """
    # 创建画布 (每张图缩放为 size * size，加上中间和边缘留白，设定为 2*size + 30)
    grid_w = size * 2 + 30
    grid_h = size * 2 + 30
    # 使用白色背景
    grid_img = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))
    
    # 坐标映射账本
    positions = [
        (10, 10),                  # 左上 (Top-Left)
        (size + 20, 10),           # 右上 (Top-Right)
        (10, size + 20),           # 左下 (Bottom-Left)
        (size + 20, size + 20)     # 右下 (Bottom-Right)
    ]
    
    for i, path in enumerate(image_paths):
        if not os.path.exists(path):
            print(f"⚠️ 警告: 未找到文件 {path}，请检查路径。")
            continue
            
        # 打开并统一缩放图像尺寸
        img = Image.open(path).convert('RGB')
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        
        # 粘贴到大图指定位置
        pos = positions[i]
        grid_img.paste(img, pos)
        
        # 在子图左上角写上简单的文本标识（学术图表标准规范）
        draw = ImageDraw.Draw(grid_img)
        # 绘制背景半透明黑框，防止白底看不清字
        draw.rectangle([pos[0]+10, pos[1]+10, pos[0]+150, pos[1]+40], fill=(0, 0, 0, 180))
        draw.text((pos[0] + 20, pos[1] + 15), titles[i], fill=(255, 255, 255))

    # 保存最终网格图
    grid_img.save(output_path, quality=95)
    print(f"🎉 拼接成功！已导出至: {output_path}")

if __name__ == "__main__":
    # 🌟 请确保这些图片和脚本在同一个目录下，或者填写绝对路径
    
    # 1. 水墨画组拼接配置 (Base vs Full-Tuned)
    ink_images = [
        "base_boat.png",       # 左上
        "ink_tuned_boat.png",  # 右上
        "base_horse.png",      # 左下
        "ink_tuned_horse.png"  # 右下
    ]
    ink_titles = ["(a) Boat: Base", "(b) Boat: Full-Tuned", "(c) Horse: Base", "(d) Horse: Full-Tuned"]
    create_image_grid(ink_images, ink_titles, "ink_style_comparison_grid.png")

    # 2. 3D粘土材质组拼接配置 (Base vs Full-Tuned)
    clay_images = [
        "base_ironman.png",       # 左上
        "clay_full_tuned_ironman.png", # 右上
        "base_fox.png",           # 左下
        "clay_full_tuned_fox.png" # 右下
    ]
    clay_titles = ["(a) Ironman: Base", "(b) Ironman: Full-Tuned", "(c) Fennec Fox: Base", "(d) Fennec Fox: Full-Tuned"]
    create_image_grid(clay_images, clay_titles, "clay_style_comparison_grid.png")
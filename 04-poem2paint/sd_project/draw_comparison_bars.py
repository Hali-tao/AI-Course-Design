# -*- coding: utf-8 -*-
"""
课程设计项目：基于 Stable Diffusion v1.5 的多风格微调
任务阶段：任务 (6) PEFT 变革 ── 核心性能指标消融可视化 (纯英学术稳定版)
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# 🌟 使用纯英学术样式，彻底抛弃本地中文字体依赖
plt.style.use('ggplot') 
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False 

def draw_performance_ablation_charts():
    print("\n📊 正在启动任务6消融数据可视化引擎（纯英学术版）...")
    
    OUTPUT_DIR = "/root/autodl-tmp/outputs/parameter_study"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ─── 1. 核心消融数据 ───
    labels = ['Full Fine-Tuning\n(Baseline)', 'Dreambooth + LoRA\n(Ours)']
    vram_data = [16.92, 6.21]        # GB
    time_data = [342, 115]           # Seconds
    volume_data = [3.2 * 1024, 12.5]  # MB
    
    # ─── 2. 画布多子图拓扑架构设计 (1行3列) ───
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), dpi=200)
    colors = ['#E64B35', '#4DBBD5']  # 学术经典配色：高亮红 与 科技蓝
    
    # ─── 子图 1：VRAM 开销 ───
    bars1 = axes[0].bar(labels, vram_data, color=colors, width=0.4, edgecolor='black', linewidth=0.8)
    axes[0].set_title('Peak Training VRAM (Lower is Better)', fontsize=11, fontweight='bold', pad=15)
    axes[0].set_ylabel('VRAM Usage (GB)', fontsize=10)
    axes[0].set_ylim(0, 20)
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 0.4, f'{yval:.2f} GB', ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    
    # ─── 子图 2：训练速度 ───
    bars2 = axes[1].bar(labels, time_data, color=colors, width=0.4, edgecolor='black', linewidth=0.8)
    axes[1].set_title('Training Time per 100 Steps (Lower is Better)', fontsize=11, fontweight='bold', pad=15)
    axes[1].set_ylabel('Time Cost (Seconds)', fontsize=10)
    axes[1].set_ylim(0, 400)
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 8, f'{int(yval)}s', ha='center', va='bottom', fontsize=9.5, fontweight='bold')
        
    # ─── 子图 3：模型资产体积 ───
    bars3 = axes[2].bar(labels, volume_data, color=colors, width=0.4, edgecolor='black', linewidth=0.8)
    axes[2].set_title('Exported Model Size (Lower is Better)', fontsize=11, fontweight='bold', pad=15)
    axes[2].set_ylabel('Storage Size (MB)', fontsize=10)
    axes[2].set_ylim(0, 4000)
    for bar in bars3:
        yval = bar.get_height()
        label_str = f'3.20 GB\n({yval:.1f} MB)' if yval > 1000 else f'{yval:.2f} MB'
        axes[2].text(bar.get_x() + bar.get_width()/2.0, yval + 100, label_str, ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    # ─── 3. 全局细节美化 ───
    for ax in axes:
        ax.tick_params(axis='x', labelsize=9.5)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    
    # 保存高保真学术图表
    chart_save_path = os.path.join(OUTPUT_DIR, "peft_vs_full_ablation_chart.png")
    plt.savefig(chart_save_path, bbox_inches='tight')
    plt.close()
    
    print(f"🎉 纯英学术版消融柱状图已成功完美导出，警告彻底消除！\n💾 路径: {chart_save_path}")

if __name__ == "__main__":
    draw_performance_ablation_charts()
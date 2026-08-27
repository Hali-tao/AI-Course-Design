import matplotlib.pyplot as plt
import numpy as np

# =====================================================================
# 1. Environment & Style Configuration
# =====================================================================
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']  # Academic standard fonts
plt.rcParams['axes.unicode_minus'] = False                # Fix minus sign rendering
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# =====================================================================
# 2. Complete Data Matrix Integration
# =====================================================================
scenarios = [
    '50 Epochs\nFixed LR (0.005)', 
    '30 Epochs\nFixed LR (0.005)', 
    '30 Epochs\nDynamic LR'
]

# Order of models to display per group
models = [
    'Modified LeNet', 
    'MobileNetV2 (Scratch)', 
    'MobileNetV2 (Full-FT)', 
    'MobileNetV2 (Freeze)', 
    'MobileNetV2 (LoRA)'
]

# Accuracy Matrix (%) [Scenario 1, Scenario 2, Scenario 3]
acc_data = {
    'Modified LeNet': [80.08, 75.39, 51.95],
    'MobileNetV2 (Scratch)': [82.81, 77.73, 80.47],
    'MobileNetV2 (Full-FT)': [91.41, 90.62, 90.62],
    'MobileNetV2 (Freeze)': [91.02, 90.23, 86.72],
    'MobileNetV2 (LoRA)': [91.80, 91.41, 87.11]
}

# Training Time Matrix (Seconds) [Scenario 1, Scenario 2, Scenario 3]
time_data = {
    'Modified LeNet': [31.31, 33.08, 22.26],
    'MobileNetV2 (Scratch)': [188.26, 104.81, 68.23],
    'MobileNetV2 (Full-FT)': [408.50, 258.30, 253.20],
    'MobileNetV2 (Freeze)': [224.10, 150.80, 146.00],
    'MobileNetV2 (LoRA)': [257.80, 154.30, 148.40]
}

# Structural Color Palette
colors = ['#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b3']

# =====================================================================
# 3. Canvas Geometry Setup
# =====================================================================
x = np.arange(len(scenarios))
total_width = 0.75
bar_width = total_width / len(models)
offsets = np.linspace(-total_width/2 + bar_width/2, total_width/2 - bar_width/2, len(models))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5), dpi=300)

# =====================================================================
# 4. Left Subplot: Test Set Accuracy Comparison
# =====================================================================
for idx, model_name in enumerate(models):
    bars1 = ax1.bar(x + offsets[idx], acc_data[model_name], bar_width, 
                    label=model_name, color=colors[idx], alpha=0.9, edgecolor='white', linewidth=0.5)
    
    # Text labels over the accuracy bars
    for bar in bars1:
        yval = bar.get_height()
        ax1.annotate(f'{yval:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, yval),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=0)

ax1.set_title('(a) Test Set Final Accuracy across Experimental Conditions', fontsize=13, fontweight='bold', pad=15)
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(scenarios, fontsize=10, fontweight='bold')
ax1.set_ylim(0, 105)
ax1.grid(True, linestyle='--', alpha=0.5, axis='y')

# =====================================================================
# 5. Right Subplot: Total Training Time Comparison
# =====================================================================
for idx, model_name in enumerate(models):
    bars2 = ax2.bar(x + offsets[idx], time_data[model_name], bar_width, 
                    label=model_name, color=colors[idx], alpha=0.9, edgecolor='white', linewidth=0.5)
    
    # Text labels over the time bars
    for bar in bars2:
        yval = bar.get_height()
        ax2.annotate(f'{yval:.1f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, yval),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=30)

ax2.set_title('(b) Total Computational Runtime across Experimental Conditions', fontsize=13, fontweight='bold', pad=15)
ax2.set_ylabel('Training Duration (Seconds)', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(scenarios, fontsize=10, fontweight='bold')
ax2.set_ylim(0, 460)
ax2.grid(True, linestyle='--', alpha=0.5, axis='y')

# =====================================================================
# 6. Global Legend Layout & Saving Asset
# =====================================================================
# Shared legend centered seamlessly between title and graphs
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.96), 
           ncol=5, frameon=True, shadow=True, fontsize=11)

plt.suptitle('Cross-Chapter Model Optimization Performance & Resource Footprint Report', 
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout(rect=[0, 0, 1, 0.92])  # Allocates whitespace layout for the global legend block
plt.savefig('model_comprehensive_comparison.png', bbox_inches='tight')
plt.show()
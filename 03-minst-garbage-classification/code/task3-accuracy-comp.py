import matplotlib.pyplot as plt
import numpy as np

# Accuracy data before and after hyperparameter tuning (percentage)
strategies = ['Freeze Fine-Tuning', 'LoRA Fine-Tuning']
before_tuning = [23.4375, 31.2500]  # Epoch=25, LR=0.002
after_tuning = [48.0469, 50.0000]   # Epoch=50, LR=0.01

x = np.arange(len(strategies))
width = 0.35

# Figure setup
plt.figure(figsize=(9, 6), dpi=150)

# Draw grouped bars
rects1 = plt.bar(x - width/2, before_tuning, width,
                 label='Before Tuning (Ep=25, LR=0.002)',
                 color='#aec7e8', edgecolor='black', linewidth=0.8)
rects2 = plt.bar(x + width/2, after_tuning, width,
                 label='After Tuning (Ep=50, LR=0.01)',
                 color='#1f77b4', edgecolor='black', linewidth=0.8)

# Chart labels (all English)
plt.title('Test Accuracy Improvement After Hyperparameter Tuning',
          fontsize=13, fontweight='bold', pad=15)
plt.ylabel('Test Accuracy (%)', fontsize=11, labelpad=10)
plt.xticks(x, strategies, fontsize=11)
plt.ylim(0, 65)

# Horizontal grid only
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Legend
plt.legend(fontsize=10, loc='upper left', frameon=True, shadow=True)

# Add value labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('accuracy_comparison_chart.png', dpi=300)
plt.show()
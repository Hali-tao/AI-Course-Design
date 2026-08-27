import matplotlib.pyplot as plt

# 1. Training log data
episodes = [150, 300, 450, 600, 750, 900, 1050, 1200, 1350, 1500]
epsilons = [0.2550, 0.2167, 0.1566, 0.1331, 0.0962, 0.0817, 0.0591, 0.0502, 0.0363, 0.0308]
rewards = [-1062.80, -376.90, -171.00, -142.00, 64.70, -777.60, 110.50, 115.80, 112.70, 115.80]

# 2. Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['axes.unicode_minus'] = False  # Ensure minus sign renders correctly

# 3. Create a figure with 1 row and 2 columns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# --- Left Plot: Total Reward Convergence ---
ax1.plot(episodes, rewards, marker='o', color='#1f77b4', linewidth=2, label='Total Reward')
ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)  # Zero reward baseline

# Annotate the sudden drop at Episode 900 (Academic highlight)
ax1.annotate('Policy Oscillation', xy=(900, -777.60), xytext=(550, -850),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
             fontsize=10, color='crimson')

ax1.set_title('RL Training Reward Convergence', fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel('Training Episodes', fontsize=10)
ax1.set_ylabel('Total Reward per Episode', fontsize=10)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='lower right')

# --- Right Plot: Exploration Rate (Epsilon) Decay ---
ax2.plot(episodes, epsilons, marker='s', color='#e377c2', linewidth=2, label='Exploration Rate ($\epsilon$)')
ax2.set_title('Exploration Rate $\epsilon$ Decay Curve', fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel('Training Episodes', fontsize=10)
ax2.set_ylabel('Current $\epsilon$', fontsize=10)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper right')

# 4. Adjust layout and save high-resolution image
plt.tight_layout()
plt.savefig('rl_training_curves_en.png', dpi=300, bbox_inches='tight')
print("Chart successfully generated and saved as 'rl_training_curves_en.png'!")
plt.show()
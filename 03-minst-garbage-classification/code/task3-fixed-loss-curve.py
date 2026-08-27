import matplotlib.pyplot as plt

# Training loss data for 3 fine-tuning methods
epochs = list(range(1, 26))

lora_loss = [
    3.2449, 3.2604, 3.1879, 3.2913, 3.0994, 3.0724, 3.2054, 2.9270, 2.7711, 3.0840,
    2.4819, 2.7014, 2.5946, 2.7875, 2.6041, 2.7197, 2.4024, 2.6544, 2.6559, 2.4926,
    2.6489, 1.9939, 1.7261, 2.4981, 2.1050
]

freeze_loss = [
    3.2536, 3.2548, 3.1499, 3.0873, 3.0815, 2.9291, 3.0406, 2.8138, 2.8204, 2.8533,
    2.7439, 2.7097, 2.7154, 2.5736, 2.4780, 2.6093, 2.4023, 2.5783, 2.6538, 2.4600,
    2.4074, 2.3538, 2.7287, 2.3971, 2.4345
]

full_loss = [
    3.1706, 3.2046, 3.2121, 3.1044, 2.8677, 2.6847, 2.9350, 2.4116, 2.2922, 1.9210,
    1.7311, 1.9109, 1.6482, 1.7131, 1.7001, 1.3592, 1.1297, 1.4587, 1.3147, 0.9037,
    0.6996, 0.5612, 0.3764, 0.5348, 0.3363
]

# Figure initialization
plt.figure(figsize=(10, 6), dpi=150)

# Plot three loss curves
plt.plot(epochs, full_loss, label='Full Fine-Tuning', color='#2ca02c', marker='o', linewidth=2.5, markersize=5)
plt.plot(epochs, lora_loss, label='LoRA Fine-Tuning', color='#1f77b4', marker='s', linewidth=2.5, markersize=5)
plt.plot(epochs, freeze_loss, label='Freeze Fine-Tuning', color='#ff7f0e', marker='^', linewidth=2.5, markersize=5)

# Title & axis labels (all English, concise)
plt.title('Training Loss Convergence Curves of Three Fine-Tuning Strategies', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Epoch', fontsize=12, labelpad=10)
plt.ylabel('Training Loss', fontsize=12, labelpad=10)

# X-axis ticks
plt.xticks(range(1, 26, 2))
plt.xlim(0.5, 25.5)

# Grid
plt.grid(True, linestyle='--', alpha=0.5, which='both')

# Legend
plt.legend(fontsize=11, loc='lower left', frameon=True, shadow=True)

# Highlight unconverged region (pure English text)
plt.axvspan(20, 25, color='red', alpha=0.08, label='Unconverged Region')
plt.text(17.2, 2.82, 'LoRA & Freeze\nUnconverged\nRoom for Improvement',
         color='darkred', fontsize=10, weight='semibold',
         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

# Layout & output
plt.tight_layout()
plt.savefig('fine_tuning_loss_curves.png', dpi=300)
plt.show()
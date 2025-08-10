#!/usr/bin/env python3
"""
Test script to verify learning rate, entropy coefficient, and target KL scheduling.
"""

import numpy as np
import matplotlib.pyplot as plt
from schedule_utils import linear_schedule, exponential_schedule, cosine_schedule

# Test parameters from parameters.py
INITIAL_LR = 3e-4
FINAL_LR = 1e-4
INITIAL_ENT_COEF = 0.02
FINAL_ENT_COEF = 0.005
INITIAL_TARGET_KL = 0.05
FINAL_TARGET_KL = 0.01

# Create progress values from 1.0 (start) to 0.0 (end)
progress_values = np.linspace(1.0, 0.0, 100)

# Test all three schedule types
schedule_types = ["linear", "exponential", "cosine"]

# Create plots
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Hyperparameter Scheduling Curves', fontsize=16)

for i, schedule_type in enumerate(schedule_types):
    # Learning rate
    lr_schedule = locals()[f"{schedule_type}_schedule"](INITIAL_LR, FINAL_LR)
    lr_values = [lr_schedule(p) for p in progress_values]
    
    axes[0, i].plot(1 - progress_values, lr_values, 'b-', linewidth=2)
    axes[0, i].set_title(f'Learning Rate ({schedule_type})')
    axes[0, i].set_xlabel('Training Progress')
    axes[0, i].set_ylabel('Learning Rate')
    axes[0, i].grid(True, alpha=0.3)
    axes[0, i].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    # Entropy coefficient
    ent_schedule = locals()[f"{schedule_type}_schedule"](INITIAL_ENT_COEF, FINAL_ENT_COEF)
    ent_values = [ent_schedule(p) for p in progress_values]
    
    axes[1, i].plot(1 - progress_values, ent_values, 'g-', linewidth=2)
    axes[1, i].set_title(f'Entropy Coefficient ({schedule_type})')
    axes[1, i].set_xlabel('Training Progress')
    axes[1, i].set_ylabel('Entropy Coefficient')
    axes[1, i].grid(True, alpha=0.3)
    
    # Target KL
    kl_schedule = locals()[f"{schedule_type}_schedule"](INITIAL_TARGET_KL, FINAL_TARGET_KL)
    kl_values = [kl_schedule(p) for p in progress_values]
    
    axes[2, i].plot(1 - progress_values, kl_values, 'r-', linewidth=2)
    axes[2, i].set_title(f'Target KL ({schedule_type})')
    axes[2, i].set_xlabel('Training Progress')
    axes[2, i].set_ylabel('Target KL')
    axes[2, i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('scheduling_curves.png', dpi=150)
print("Scheduling curves saved to scheduling_curves.png")

# Print some example values
print("\nLinear Schedule Examples (at 0%, 25%, 50%, 75%, 100% progress):")
linear_lr = linear_schedule(INITIAL_LR, FINAL_LR)
linear_ent = linear_schedule(INITIAL_ENT_COEF, FINAL_ENT_COEF)
linear_kl = linear_schedule(INITIAL_TARGET_KL, FINAL_TARGET_KL)

for progress in [0.0, 0.25, 0.5, 0.75, 1.0]:
    progress_remaining = 1.0 - progress
    print(f"\nProgress: {progress*100:.0f}%")
    print(f"  LR: {linear_lr(progress_remaining):.2e}")
    print(f"  Entropy: {linear_ent(progress_remaining):.4f}")
    print(f"  Target KL: {linear_kl(progress_remaining):.4f}")
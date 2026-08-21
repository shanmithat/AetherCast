import os
import sys
import time
import random
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.insert(0, os.getcwd())

from aethercast.models.fno2d import FNO2d
from aethercast.utils.physics_loss import PhysicsInformedLoss
from aethercast.data import generate_synthetic_data

SEED = 42

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_model(lambda_phy=0.01, num_samples=256, device="cpu"):
    set_seeds(SEED)
    model = FNO2d().to(device)
    # Generate training data with train seed
    X, Y = generate_synthetic_data(num_samples, seed=SEED)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-4)
    
    mse_criterion = nn.MSELoss()
    pinn_criterion = PhysicsInformedLoss().to(device)
    batch_size = 32

    for epoch in range(30):
        model.train()
        indices = torch.randperm(len(X))
        for i in range(0, len(X), batch_size):
            batch_idx = indices[i:i+batch_size]
            bx = X[batch_idx].to(device)
            by = Y[batch_idx].to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = mse_criterion(out, by) + lambda_phy * pinn_criterion(out, bx)
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Physical Mismatch Verification Experiment on {device}...")
    
    # 1. Train FNO model
    print("Training base PI-FNO model...")
    model = train_model(lambda_phy=0.01, device=device)
    model.eval()
    
    # 2. Generate non-overlapping test set of 1,000 samples
    test_seed = SEED + 20000
    print(f"Generating independent test dataset with seed {test_seed}...")
    X_test, Y_test = generate_synthetic_data(1000, seed=test_seed)
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)
    
    # 3. Get valid base predictions
    print("Running base FNO projections...")
    with torch.inference_mode():
        Y_pred = model(X_test)
        
    pinn_criterion = PhysicsInformedLoss().to(device)
    
    # Base physics residual
    base_pde = pinn_criterion(Y_pred, X_test).item()
    gt_pde = pinn_criterion(Y_test, X_test).item()
    
    # --- Apply perturbations ---
    print("Applying perturbations to predictions...")
    
    # A. Spatial Displacement (Jitter)
    # Roll each frame of each trajectory randomly by 1-3 pixels
    Y_jitter = Y_pred.clone()
    for i in range(len(Y_jitter)):
        for t in range(Y_jitter.shape[1]):
            dx = random.randint(1, 3) * random.choice([-1, 1])
            dy = random.randint(1, 3) * random.choice([-1, 1])
            Y_jitter[i, t] = torch.roll(Y_jitter[i, t], shifts=(dy, dx), dims=(0, 1))
    jitter_pde = pinn_criterion(Y_jitter, X_test).item()
    
    # B. Amplitude Scaling (Time-varying amplitude scaling)
    # Scale each frame by a random multiplier to violate physical decay conservation
    Y_scaled = Y_pred.clone()
    for i in range(len(Y_scaled)):
        for t in range(Y_scaled.shape[1]):
            scale_factor = random.uniform(0.4, 1.6)
            Y_scaled[i, t] = Y_scaled[i, t] * scale_factor
    scaled_pde = pinn_criterion(Y_scaled, X_test).item()
    
    # C. Excessive Blurring (Excessive physical diffusion)
    # Apply a 5x5 box blur kernel to each frame
    blur_kernel = torch.ones(1, 1, 5, 5, device=device) / 25.0
    batch, time, height, width = Y_pred.shape
    Y_blur_in = Y_pred.reshape(-1, 1, height, width)
    Y_blur_padded = F.pad(Y_blur_in, (2, 2, 2, 2), mode='circular')
    Y_blur_out = F.conv2d(Y_blur_padded, blur_kernel)
    Y_blur = Y_blur_out.reshape(batch, time, height, width)
    blur_pde = pinn_criterion(Y_blur, X_test).item()
    
    # D. Nonphysical Temporal Perturbation (Shuffled time sequence)
    # Randomly permute the 24 frames of each trajectory
    Y_shuffled = Y_pred.clone()
    for i in range(len(Y_shuffled)):
        perm = torch.randperm(time)
        Y_shuffled[i] = Y_shuffled[i, perm]
    shuffled_pde = pinn_criterion(Y_shuffled, X_test).item()
    
    # Report Results
    print("\n--- Physical Mismatch Verification Results ---")
    print(f"Ground Truth (Reference) MSE-PDE           : {gt_pde:.4f}")
    print(f"PI-FNO Base Predictions MSE-PDE             : {base_pde:.4f}")
    print(f"Perturbation: Spatial Jitter MSE-PDE       : {jitter_pde:.4f}")
    print(f"Perturbation: Amplitude Scaling MSE-PDE    : {scaled_pde:.4f}")
    print(f"Perturbation: Blurring/Diffusion MSE-PDE   : {blur_pde:.4f}")
    print(f"Perturbation: Temporal Shuffling MSE-PDE   : {shuffled_pde:.4f}")
    
    results = [
        {"configuration": "Ground Truth Reference", "mse_pde": gt_pde},
        {"configuration": "PI-FNO Base Predictions", "mse_pde": base_pde},
        {"configuration": "Spatial Jitter (Displacement)", "mse_pde": jitter_pde},
        {"configuration": "Amplitude Scaling (Nonphysical Decay)", "mse_pde": scaled_pde},
        {"configuration": "Excessive Blurring (Diffusion Mismatch)", "mse_pde": blur_pde},
        {"configuration": "Temporal Shuffling (Broken Time)", "mse_pde": shuffled_pde}
    ]
    
    # Save to CSV
    os.makedirs("experiments/results", exist_ok=True)
    csv_path = "experiments/results/verification_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["configuration", "mse_pde"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Metrics saved to {csv_path}")
    
    # Generate verification bar chart
    plt.figure(figsize=(10, 6))
    configs = [r["configuration"] for r in results]
    pdes = [r["mse_pde"] for r in results]
    
    colors = ['#475569', '#3b82f6', '#f59e0b', '#ec4899', '#8b5cf6', '#ef4444']
    plt.bar(configs, pdes, color=colors, edgecolor='black', alpha=0.85)
    plt.yscale('log')
    plt.ylabel("MSE-PDE (Log Scale)", fontsize=11, fontweight="bold")
    plt.title("MSE-PDE Physical Mismatch Verification Sensitivity Analysis", fontsize=12, fontweight="bold", pad=15)
    plt.xticks(rotation=15, ha='right', fontsize=9)
    plt.grid(axis='y', linestyle='--', alpha=0.5, which='both')
    plt.tight_layout()
    
    plot_path = "experiments/results/verification_pde.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Verification plot saved to {plot_path}")

if __name__ == "__main__":
    main()

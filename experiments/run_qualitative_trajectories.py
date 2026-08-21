import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.insert(0, os.getcwd())

from aethercast.models.fno2d import FNO2d
from aethercast.data import generate_synthetic_data

SEED = 42

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_model(lambda_phy=0.01, epochs=30, num_samples=256, device="cpu"):
    set_seeds(SEED)
    model = FNO2d().to(device)
    X, Y = generate_synthetic_data(num_samples, seed=SEED)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    
    mse_criterion = nn.MSELoss()
    from aethercast.utils.physics_loss import PhysicsInformedLoss
    pinn_criterion = PhysicsInformedLoss().to(device)
    batch_size = 32

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X))
        for i in range(0, len(X), batch_size):
            batch_idx = indices[i:i+batch_size]
            bx = X[batch_idx].to(device)
            by = Y[batch_idx].to(device)

            optimizer.zero_grad()
            out = model(bx)
            
            mse_loss = mse_criterion(out, by)
            pinn_loss = pinn_criterion(out, bx)
            loss = mse_loss + lambda_phy * pinn_loss
            
            loss.backward()
            optimizer.step()
        scheduler.step()
        
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Experiment 5: Qualitative Trajectories on {device}...")
    
    # Train PI-FNO model
    print("Training PI-FNO model for trajectory visualization...")
    model = train_model(lambda_phy=0.01, device=device)
    model.eval()
    
    # Generate independent test set (non-overlapping seeds)
    test_seed = SEED + 20000
    print(f"Generating test set with seed {test_seed}...")
    X_val, Y_val = generate_synthetic_data(50, seed=test_seed)  # Generate 50 trajectories, pick 3
    
    # We choose test cases 0, 4, and 7 as representative advection cases
    indices_to_plot = [0, 4, 7]
    os.makedirs("experiments/results", exist_ok=True)
    
    # Step index mappings for minutes:
    # DT = 5 mins, so:
    # 0 mins = initial field (input index 0)
    # 30 mins = step 6 (index 5)
    # 60 mins = step 12 (index 11)
    # 120 mins = step 24 (index 23)
    step_indices = [5, 11, 23]
    time_labels = ["30 Min", "60 Min", "120 Min"]

    with torch.inference_mode():
        for idx in indices_to_plot:
            bx = X_val[idx:idx+1].to(device)
            by = Y_val[idx:idx+1].to(device)
            
            # Predict
            pred = model(bx).squeeze(0).cpu().numpy()  # shape (24, 32, 32)
            true = by.squeeze(0).cpu().numpy()          # shape (24, 32, 32)
            init = bx[0, ..., 0].cpu().numpy()           # shape (32, 32)
            
            # Setup Plotting Grid
            # Reference, PI-FNO, and Error at t=30, 60, 120
            # Column 1: t=0 (Input)
            # Column 2: t=30 Min
            # Column 3: t=60 Min
            # Column 4: t=120 Min
            
            fig, axes = plt.subplots(3, 4, figsize=(12, 9))
            
            # Row 0: Reference
            axes[0, 0].imshow(init, cmap="viridis", origin="lower")
            axes[0, 0].set_title("Input (t=0)", fontsize=10, fontweight="bold")
            
            for col_idx, step in enumerate(step_indices):
                axes[0, col_idx + 1].imshow(true[step], cmap="viridis", origin="lower")
                axes[0, col_idx + 1].set_title(f"Ref ({time_labels[col_idx]})", fontsize=10, fontweight="bold")
                
            # Row 1: PI-FNO
            axes[1, 0].imshow(init, cmap="viridis", origin="lower")
            axes[1, 0].set_title("Input (t=0)", fontsize=10, fontweight="bold")
            
            for col_idx, step in enumerate(step_indices):
                axes[1, col_idx + 1].imshow(pred[step], cmap="viridis", origin="lower")
                axes[1, col_idx + 1].set_title(f"PI-FNO ({time_labels[col_idx]})", fontsize=10, fontweight="bold")
                
            # Row 2: Error Fields (Absolute difference)
            # Center of column 0 on Row 2 can display wind vectors or remain empty
            axes[2, 0].axis("off")
            u_wind = bx[0, 0, 0, 1].item()
            v_wind = bx[0, 0, 0, 2].item()
            axes[2, 0].text(0.5, 0.5, f"Wind:\nU={u_wind:.1f}\nV={v_wind:.1f}", 
                            ha='center', va='center', fontsize=11, fontweight="bold",
                            bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray'))
            
            for col_idx, step in enumerate(step_indices):
                diff = np.abs(pred[step] - true[step])
                im = axes[2, col_idx + 1].imshow(diff, cmap="magma", origin="lower")
                axes[2, col_idx + 1].set_title(f"Abs Error ({time_labels[col_idx]})", fontsize=10, fontweight="bold")
                fig.colorbar(im, ax=axes[2, col_idx + 1], fraction=0.046, pad=0.04)

            # Labels and styling
            for r in range(3):
                for c in range(4):
                    if r == 2 and c == 0:
                        continue
                    axes[r, c].set_xticks([])
                    axes[r, c].set_yticks([])
            
            fig.suptitle(f"Qualitative Trajectory Comparison - Test Case {idx}", fontsize=14, fontweight="bold")
            plt.tight_layout()
            
            plot_path = f"experiments/results/trajectory_comparison_{idx}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Qualitative comparison plot saved to {plot_path}")

if __name__ == "__main__":
    main()

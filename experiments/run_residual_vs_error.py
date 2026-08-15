import os
import sys
import csv
import random
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

def train_model(lambda_phy=0.01, epochs=30, num_samples=256, device="cpu"):
    set_seeds(SEED)
    model = FNO2d().to(device)
    X, Y = generate_synthetic_data(num_samples)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    
    mse_criterion = nn.MSELoss()
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
    print(f"Running Experiment 4: Residual vs Error analysis on {device}...")
    
    # Train both models
    print("Training PI-FNO model...")
    model_pinn = train_model(lambda_phy=0.01, device=device)
    print("Training Data-Driven FNO model...")
    model_data = train_model(lambda_phy=0.0, device=device)
    
    # Generate test set of 1,000 samples
    set_seeds(SEED)
    X_val, Y_val = generate_synthetic_data(1000)
    
    pinn_criterion = PhysicsInformedLoss().to(device)
    
    model_pinn.eval()
    model_data.eval()
    
    results = []
    
    pinn_l2 = []
    pinn_pde = []
    data_l2 = []
    data_pde = []
    
    print("Evaluating models on test set...")
    with torch.inference_mode():
        for i in range(len(X_val)):
            bx = X_val[i:i+1].to(device)
            by = Y_val[i:i+1].to(device)
            
            # Evaluate PI-FNO
            out_pinn = model_pinn(bx)
            rel_l2_pinn = (torch.norm(out_pinn - by) / torch.norm(by)).item()
            mse_pde_pinn = pinn_criterion(out_pinn, bx).item()
            
            pinn_l2.append(rel_l2_pinn)
            pinn_pde.append(mse_pde_pinn)
            
            results.append({
                "sample_idx": i,
                "model_type": "PI-FNO",
                "rel_l2": rel_l2_pinn,
                "pde_residual": mse_pde_pinn
            })
            
            # Evaluate Data-Driven FNO
            out_data = model_data(bx)
            rel_l2_data = (torch.norm(out_data - by) / torch.norm(by)).item()
            mse_pde_data = pinn_criterion(out_data, bx).item()
            
            data_l2.append(rel_l2_data)
            data_pde.append(mse_pde_data)
            
            results.append({
                "sample_idx": i,
                "model_type": "Data-Driven FNO",
                "rel_l2": rel_l2_data,
                "pde_residual": mse_pde_data
            })

    # Save results to CSV
    os.makedirs("experiments/results", exist_ok=True)
    csv_path = "experiments/results/residual_vs_error.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_idx", "model_type", "rel_l2", "pde_residual"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Data saved to {csv_path}")

    # Generate scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(data_l2, data_pde, color="#f43f5e", alpha=0.5, s=15, label="Data-Driven FNO (λ = 0.0)")
    plt.scatter(pinn_l2, pinn_pde, color="#10b981", alpha=0.5, s=15, label="PI-FNO (λ = 0.01)")
    
    plt.title("Mean Squared PDE Residual (MSE-PDE) vs. Relative L2 Error", fontsize=12, fontweight="bold")
    plt.xlabel("Relative L2 Prediction Error", fontsize=10)
    plt.ylabel("Mean Squared PDE Residual (MSE-PDE)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    
    # Save plot to PNG
    plot_path = "experiments/results/pde_vs_l2.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Scatter plot saved to {plot_path}")

if __name__ == "__main__":
    main()

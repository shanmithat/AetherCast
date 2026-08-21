import os
import sys
import csv
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to sys.path
sys.path.insert(0, os.getcwd())

from aethercast.models.fno2d import FNO2d
from aethercast.utils.physics_loss import PhysicsInformedLoss
from aethercast.data import generate_synthetic_data

SEEDS = [42, 100, 2026, 7, 999]

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_model(lambda_phy=0.01, epochs=30, num_samples=256, seed=42, device="cpu"):
    set_seeds(seed)
    model = FNO2d().to(device)
    X, Y = generate_synthetic_data(num_samples, seed=seed)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    
    mse_criterion = nn.MSELoss()
    pinn_criterion = PhysicsInformedLoss().to(device)
    batch_size = min(32, num_samples)

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

def evaluate_model_samples(model, X_val, Y_val, device):
    model.eval()
    mse_list = []
    rel_l2_list = []
    pinn_list = []
    
    pinn_criterion = PhysicsInformedLoss().to(device)
    
    with torch.inference_mode():
        for i in range(len(X_val)):
            bx = X_val[i:i+1].to(device)
            by = Y_val[i:i+1].to(device)
            out = model(bx)
            
            mse = F.mse_loss(out, by).item()
            rel_l2 = (torch.norm(out - by) / torch.norm(by)).item()
            pinn = pinn_criterion(out, bx).item()
            
            mse_list.append(mse)
            rel_l2_list.append(rel_l2)
            pinn_list.append(pinn)
            
    return mse_list, rel_l2_list, pinn_list

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Training Size Sweep on {device} across 5 seeds...")
    
    sizes = [64, 128, 256, 512, 1024]
    results = []
    
    for size in sizes:
        print(f"\nEvaluating training size = {size}...")
        
        all_mses, all_l2s, all_pdes = [], [], []
        
        for seed in SEEDS:
            # 1. Train model
            model = train_model(lambda_phy=0.01, epochs=30, num_samples=size, seed=seed, device=device)
            
            # 2. Generate independent test set
            test_seed = seed + 20000
            X_val, Y_val = generate_synthetic_data(1000, seed=test_seed)
            
            # 3. Evaluate
            mses, l2s, pdes = evaluate_model_samples(model, X_val, Y_val, device)
            all_mses.extend(mses)
            all_l2s.extend(l2s)
            all_pdes.extend(pdes)
            
        results.append({
            "size": size,
            "mse_mean": np.mean(all_mses), "mse_std": np.std(all_mses),
            "rel_l2_mean": np.mean(all_l2s), "rel_l2_std": np.std(all_l2s),
            "pinn_mean": np.mean(all_pdes), "pinn_std": np.std(all_pdes)
        })
        
        print(f"Size {size} complete: MSE = {np.mean(all_mses):.6f} ± {np.std(all_mses):.6f}")

    # Ensure results directory exists
    os.makedirs("experiments/results", exist_ok=True)
    csv_path = "experiments/results/training_size.csv"
    
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["size", "mse_mean", "mse_std", "rel_l2_mean", "rel_l2_std", "pinn_mean", "pinn_std"])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Sweep results successfully saved to {csv_path}")

if __name__ == "__main__":
    main()

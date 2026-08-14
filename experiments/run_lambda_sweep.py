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

def evaluate_model(model, X_val, Y_val, device):
    model.eval()
    mse_list = []
    rel_l2_list = []
    pinn_list = []
    
    pinn_criterion = PhysicsInformedLoss().to(device)
    
    with torch.inference_mode():
        for i in range(0, len(X_val), 32):
            bx = X_val[i:i+32].to(device)
            by = Y_val[i:i+32].to(device)
            out = model(bx)
            
            for j in range(len(bx)):
                single_out = out[j:j+1]
                single_by = by[j:j+1]
                single_bx = bx[j:j+1]
                
                mse = F.mse_loss(single_out, single_by).item()
                rel_l2 = (torch.norm(single_out - single_by) / torch.norm(single_by)).item()
                pinn = pinn_criterion(single_out, single_bx).item()
                
                mse_list.append(mse)
                rel_l2_list.append(rel_l2)
                pinn_list.append(pinn)
            
    return {
        "mse_mean": np.mean(mse_list), "mse_std": np.std(mse_list),
        "rel_l2_mean": np.mean(rel_l2_list), "rel_l2_std": np.std(rel_l2_list),
        "pinn_mean": np.mean(pinn_list), "pinn_std": np.std(pinn_list)
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Lambda Weight Sweep on {device}...")
    
    # Generate fixed 1,000-sample test set using SEED
    set_seeds(SEED)
    X_val, Y_val = generate_synthetic_data(1000)
    
    lambdas = [0.0, 0.001, 0.01, 0.05, 0.1]
    results = []
    
    for l in lambdas:
        print(f"Training FNO with lambda_phy = {l}...")
        model = train_model(lambda_phy=l, num_samples=256, device=device)
        metrics = evaluate_model(model, X_val, Y_val, device)
        
        results.append({
            "lambda": l,
            "mse_mean": metrics["mse_mean"], "mse_std": metrics["mse_std"],
            "rel_l2_mean": metrics["rel_l2_mean"], "rel_l2_std": metrics["rel_l2_std"],
            "pinn_mean": metrics["pinn_mean"], "pinn_std": metrics["pinn_std"]
        })
        
        print(f"Lambda {l} complete: MSE = {metrics['mse_mean']:.6f} ± {metrics['mse_std']:.6f}, PINN = {metrics['pinn_mean']:.6f} ± {metrics['pinn_std']:.6f}")

    # Ensure results directory exists
    os.makedirs("experiments/results", exist_ok=True)
    csv_path = "experiments/results/lambda_sweep.csv"
    
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lambda", "mse_mean", "mse_std", "rel_l2_mean", "rel_l2_std", "pinn_mean", "pinn_std"])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Sweep results successfully saved to {csv_path}")

if __name__ == "__main__":
    main()

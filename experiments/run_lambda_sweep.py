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

SEEDS = [42, 100, 2026]  # 3 seeds for quick reliable sweep on CPU

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_model(X, Y, lambda_phy=0.01, epochs=15, seed=42, device="cpu"):
    set_seeds(seed)
    model = FNO2d().to(device)
    
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

def evaluate_model_samples(model, X_val, Y_val, device):
    model.eval()
    mse_list = []
    rel_l2_list = []
    pinn_list = []
    
    pinn_criterion = PhysicsInformedLoss().to(device)
    batch_size = 128
    
    with torch.inference_mode():
        for i in range(0, len(X_val), batch_size):
            bx = X_val[i:i+batch_size].to(device)
            by = Y_val[i:i+batch_size].to(device)
            out = model(bx)
            
            # Sample-wise MSE
            mse_batch = F.mse_loss(out, by, reduction='none').mean(dim=(1, 2, 3))
            mse_list.extend(mse_batch.cpu().numpy().tolist())
            
            # Sample-wise Relative L2
            diff_norm = torch.norm((out - by).flatten(start_dim=1), p=2, dim=1)
            by_norm = torch.norm(by.flatten(start_dim=1), p=2, dim=1)
            rel_l2_batch = diff_norm / by_norm
            rel_l2_list.extend(rel_l2_batch.cpu().numpy().tolist())
            
            # Vectorized physics loss evaluation keeping batch dimension
            batch_len, time_len, height, width = out.shape
            cx = bx[:, 0, 0, 1].view(-1, 1, 1, 1)
            cy = bx[:, 0, 0, 2].view(-1, 1, 1, 1)
            
            du_dt = (out[:, 1:] - out[:, :-1]) / pinn_criterion.dt
            u_slice = out[:, :-1].reshape(batch_len * (time_len - 1), 1, height, width)
            u_padded = F.pad(u_slice, (1, 1, 1, 1), mode='circular')
            
            du_dx = F.conv2d(u_padded, pinn_criterion.weight_dx).reshape(batch_len, time_len - 1, height, width)
            du_dy = F.conv2d(u_padded, pinn_criterion.weight_dy).reshape(batch_len, time_len - 1, height, width)
            laplacian = F.conv2d(u_padded, pinn_criterion.weight_laplacian).reshape(batch_len, time_len - 1, height, width)
            u_slice_3d = u_slice.reshape(batch_len, time_len - 1, height, width)
            
            residual = du_dt + (cx * du_dx) + (cy * du_dy) - (pinn_criterion.D * laplacian) + (pinn_criterion.lambda_d * u_slice_3d)
            pinn_batch = torch.mean(residual ** 2, dim=(1, 2, 3))
            pinn_list.extend(pinn_batch.cpu().numpy().tolist())
            
    return mse_list, rel_l2_list, pinn_list

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Optimized Lambda Weight Sweep on {device} across 3 seeds...")
    
    # 1. Pre-generate datasets once per seed
    train_datasets = {}
    val_datasets = {}
    for seed in SEEDS:
        print(f"Pre-generating datasets for seed {seed}...")
        train_datasets[seed] = generate_synthetic_data(256, seed=seed)
        val_datasets[seed] = generate_synthetic_data(1000, seed=seed + 20000)
        
    lambdas = [0.0, 0.001, 0.01, 0.05, 0.1]
    results = []
    
    for l in lambdas:
        print(f"\nEvaluating lambda = {l}...")
        
        all_mses, all_l2s, all_pdes = [], [], []
        
        for seed in SEEDS:
            X_train, Y_train = train_datasets[seed]
            X_val, Y_val = val_datasets[seed]
            
            # 2. Train model reusing pre-generated training set
            model = train_model(X_train, Y_train, lambda_phy=l, epochs=15, seed=seed, device=device)
            
            # 3. Evaluate reusing pre-generated validation set
            mses, l2s, pdes = evaluate_model_samples(model, X_val, Y_val, device)
            all_mses.extend(mses)
            all_l2s.extend(l2s)
            all_pdes.extend(pdes)
            
        results.append({
            "lambda": l,
            "mse": f"{np.mean(all_mses):.6f} ± {np.std(all_mses):.6f}",
            "rel_l2": f"{np.mean(all_l2s):.6f} ± {np.std(all_l2s):.6f}",
            "pinn": f"{np.mean(all_pdes):.6f} ± {np.std(all_pdes):.6f}"
        })

    # Ensure results directory exists
    os.makedirs("experiments/results", exist_ok=True)
    csv_path = "experiments/results/lambda_sweep.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lambda", "mse", "rel_l2", "pinn"])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nLambda sweep results successfully saved to {csv_path}")

if __name__ == "__main__":
    main()

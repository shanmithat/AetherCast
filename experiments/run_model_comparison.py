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
from aethercast.solvers import run_discrete_transport_reference

SEEDS = [42, 100, 2026, 7, 999]

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_model(lambda_phy=0.01, num_samples=256, seed=42, device="cpu"):
    set_seeds(seed)
    model = FNO2d().to(device)
    # Generate training data using specific seed
    X, Y = generate_synthetic_data(num_samples, seed=seed)
    
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

def evaluate_model_samples(model, X_val, Y_val, device):
    model.eval()
    mse_list = []
    rel_l2_list = []
    pinn_list = []
    time_list = []
    
    pinn_criterion = PhysicsInformedLoss().to(device)
    
    # Warm-up pass to ensure PyTorch initialization overhead isn't counted
    with torch.inference_mode():
        _ = model(X_val[0:1].to(device))
        if device.type == "cuda":
            torch.cuda.synchronize()
            
    with torch.inference_mode():
        for i in range(len(X_val)):
            bx = X_val[i:i+1].to(device)
            by = Y_val[i:i+1].to(device)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_start = time.perf_counter()
            out = model(bx)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            
            mse = F.mse_loss(out, by).item()
            rel_l2 = (torch.norm(out - by) / torch.norm(by)).item()
            pinn = pinn_criterion(out, bx).item()
            
            mse_list.append(mse)
            rel_l2_list.append(rel_l2)
            pinn_list.append(pinn)
            time_list.append(t_end - t_start)
            
    return mse_list, rel_l2_list, pinn_list, time_list

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running FNO vs Classical Solver Benchmarks on {device} across 5 seeds...")
    
    # Pre-train models once per seed to avoid redundant training
    models_pinn = {}
    models_data = {}
    for seed in SEEDS:
        print(f"Training models for seed {seed}...")
        models_pinn[seed] = train_model(lambda_phy=0.01, num_samples=256, seed=seed, device=device)
        models_data[seed] = train_model(lambda_phy=0.0, num_samples=256, seed=seed, device=device)
        
    sizes = [100, 500, 1000]
    results = []
    
    pinn_criterion = PhysicsInformedLoss().to(device)
    
    for size in sizes:
        print(f"\nEvaluating test set size = {size}...")
        
        pinn_mses, pinn_l2s, pinn_pdes, pinn_times = [], [], [], []
        data_mses, data_l2s, data_pdes, data_times = [], [], [], []
        fd_pdes, fd_times = [], []
        
        for seed in SEEDS:
            model_pinn = models_pinn[seed]
            model_data = models_data[seed]
            
            # Generate non-overlapping test set
            test_seed = seed + 20000
            X_val, Y_val = generate_synthetic_data(size, seed=test_seed)
            
            # PI-FNO
            pmse, pl2, ppde, ptime = evaluate_model_samples(model_pinn, X_val, Y_val, device)
            pinn_mses.extend(pmse)
            pinn_l2s.extend(pl2)
            pinn_pdes.extend(ppde)
            pinn_times.extend(ptime)
            
            # Data-Driven FNO
            dmse, dl2, dpde, dtime = evaluate_model_samples(model_data, X_val, Y_val, device)
            data_mses.extend(dmse)
            data_l2s.extend(dl2)
            data_pdes.extend(dpde)
            data_times.extend(dtime)
            
            # Classical solver (Discrete Transport Reference)
            for i in range(len(X_val)):
                bx = X_val[i:i+1]
                rain = bx[0, ..., 0].numpy()
                u = bx[0, 0, 0, 1].item()
                v = bx[0, 0, 0, 2].item()
                
                t_start = time.perf_counter()
                out_fd = run_discrete_transport_reference(rain, u, v)
                t_end = time.perf_counter()
                
                fd_times.append(t_end - t_start)
                
                out_fd_tensor = torch.tensor(np.array([out_fd]), dtype=torch.float32, device=device)
                fd_pdes.append(pinn_criterion(out_fd_tensor, bx.to(device)).item())
                
        # Aggregate and report metrics
        results.append({
            "test_size": size, "model": "PI-FNO",
            "mse": f"{np.mean(pinn_mses):.3f} ± {np.std(pinn_mses):.3f}",
            "rel_l2": f"{np.mean(pinn_l2s):.3f} ± {np.std(pinn_l2s):.3f}",
            "pinn": f"{np.mean(pinn_pdes):.2f} ± {np.std(pinn_pdes):.2f}",
            "latency": f"{np.mean(pinn_times) * 1000:.3f} ms"
        })
        
        results.append({
            "test_size": size, "model": "Data-Driven FNO",
            "mse": f"{np.mean(data_mses):.3f} ± {np.std(data_mses):.3f}",
            "rel_l2": f"{np.mean(data_l2s):.3f} ± {np.std(data_l2s):.3f}",
            "pinn": f"{np.mean(data_pdes):.2f} ± {np.std(data_pdes):.2f}",
            "latency": f"{np.mean(data_times) * 1000:.3f} ms"
        })
        
        results.append({
            "test_size": size, "model": "Discrete Transport Reference",
            "mse": "reference", "rel_l2": "reference",
            "pinn": f"{np.mean(fd_pdes):.2f} ± {np.std(fd_pdes):.2f}",
            "latency": f"{np.mean(fd_times) * 1000:.3f} ms"
        })

    # Ensure results directory exists
    os.makedirs("experiments/results", exist_ok=True)
    csv_path = "experiments/results/model_comparison.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["test_size", "model", "mse", "rel_l2", "pinn", "latency"])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nComparison results successfully saved to {csv_path}")

if __name__ == "__main__":
    main()

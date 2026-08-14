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
    X, Y = generate_synthetic_data(num_samples)
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

def evaluate_model(model, X_val, Y_val, device):
    model.eval()
    mse_list = []
    rel_l2_list = []
    pinn_list = []
    time_list = []
    
    pinn_criterion = PhysicsInformedLoss().to(device)
    
    # Warm-up pass to ensure PyTorch and CUDA compilation overhead isn't counted
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
            
    return {
        "mse": np.mean(mse_list), "mse_std": np.std(mse_list),
        "rel_l2": np.mean(rel_l2_list), "rel_l2_std": np.std(rel_l2_list),
        "pinn": np.mean(pinn_list), "pinn_std": np.std(pinn_list),
        "latency_ms": np.mean(time_list) * 1000
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running FNO vs Classical Solver Benchmarks on {device}...")
    
    # Train configurations
    model_pinn = train_model(lambda_phy=0.01, num_samples=256, device=device)
    model_data = train_model(lambda_phy=0.0, num_samples=256, device=device)
    
    sizes = [100, 500, 1000]
    results = []
    
    pinn_criterion = PhysicsInformedLoss().to(device)
    
    for size in sizes:
        print(f"\nEvaluating test set size = {size}...")
        set_seeds(SEED)
        X_val, Y_val = generate_synthetic_data(size)
        
        # 1. PI-FNO (PINN)
        metrics_pinn = evaluate_model(model_pinn, X_val, Y_val, device)
        results.append({
            "test_size": size, "model": "PI-FNO",
            "mse": f"{metrics_pinn['mse']:.3f} ± {metrics_pinn['mse_std']:.3f}",
            "rel_l2": f"{metrics_pinn['rel_l2']:.3f} ± {metrics_pinn['rel_l2_std']:.3f}",
            "pinn": f"{metrics_pinn['pinn']:.2f} ± {metrics_pinn['pinn_std']:.2f}",
            "latency": f"{metrics_pinn['latency_ms']:.3f} ms"
        })
        
        # 2. Data-Driven FNO
        metrics_data = evaluate_model(model_data, X_val, Y_val, device)
        results.append({
            "test_size": size, "model": "Data-Driven FNO",
            "mse": f"{metrics_data['mse']:.3f} ± {metrics_data['mse_std']:.3f}",
            "rel_l2": f"{metrics_data['rel_l2']:.3f} ± {metrics_data['rel_l2_std']:.3f}",
            "pinn": f"{metrics_data['pinn']:.2f} ± {metrics_data['pinn_std']:.2f}",
            "latency": f"{metrics_data['latency_ms']:.3f} ms"
        })
        
        # 3. Classical Solver
        # Run step-by-step numpy-based advection
        latency_fd = []
        pinn_fd = []
        for i in range(size):
            bx = X_val[i:i+1]
            rain = bx[0, ..., 0].numpy()
            u = bx[0, 0, 0, 1].item()
            v = bx[0, 0, 0, 2].item()
            
            t_start = time.perf_counter()
            out_fd = run_discrete_transport_reference(rain, u, v)
            t_end = time.perf_counter()
            
            latency_fd.append(t_end - t_start)
            
            # Compute PDE residual of FD output
            out_fd_tensor = torch.tensor(np.array([out_fd]), dtype=torch.float32, device=device)
            bx_device = bx.to(device)
            pinn_fd.append(pinn_criterion(out_fd_tensor, bx_device).item())
            
        results.append({
            "test_size": size, "model": "FD Solver",
            "mse": "reference", "rel_l2": "reference",
            "pinn": f"{np.mean(pinn_fd):.2f} ± {np.std(pinn_fd):.2f}",
            "latency": f"{np.mean(latency_fd) * 1000:.3f} ms"
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

import os
import sys
import time
import random
import csv
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

# Define wind configurations
ID_WIND = (-10.0, 10.0)
OOD_WIND = [(-25.0, -15.0), (15.0, 25.0)]

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_model(lambda_phy=0.01, num_samples=256, seed=42, device="cpu"):
    set_seeds(seed)
    model = FNO2d().to(device)
    # Generate training data with ID wind range and specific seed
    X, Y = generate_synthetic_data(num_samples, seed=seed, wind_range=ID_WIND)
    
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
    print(f"Running Out-of-Distribution (OOD) Generalization Experiment on {device}...")
    
    # Store individual runs
    runs_id_pinn = {"mse": [], "rel_l2": [], "pinn": []}
    runs_id_data = {"mse": [], "rel_l2": [], "pinn": []}
    runs_ood_pinn = {"mse": [], "rel_l2": [], "pinn": []}
    runs_ood_data = {"mse": [], "rel_l2": [], "pinn": []}
    
    for run_idx, seed in enumerate(SEEDS):
        print(f"\n--- Seed {seed} (Run {run_idx+1}/5) ---")
        
        # 1. Train models on ID wind range
        print("Training FNO configurations...")
        model_pinn = train_model(lambda_phy=0.01, num_samples=256, seed=seed, device=device)
        model_data = train_model(lambda_phy=0.0, num_samples=256, seed=seed, device=device)
        
        # 2. Generate independent test sets (non-overlapping seeds)
        test_seed = seed + 20000
        print("Generating test sets (ID & OOD)...")
        X_id, Y_id = generate_synthetic_data(200, seed=test_seed, wind_range=ID_WIND)
        X_ood, Y_ood = generate_synthetic_data(200, seed=test_seed + 5000, wind_range=OOD_WIND)
        
        # 3. Evaluate ID Set
        print("Evaluating on ID Test Set...")
        id_pinn_mse, id_pinn_l2, id_pinn_pde = evaluate_model(model_pinn, X_id, Y_id, device)
        id_data_mse, id_data_l2, id_data_pde = evaluate_model(model_data, X_id, Y_id, device)
        
        runs_id_pinn["mse"].extend(id_pinn_mse)
        runs_id_pinn["rel_l2"].extend(id_pinn_l2)
        runs_id_pinn["pinn"].extend(id_pinn_pde)
        
        runs_id_data["mse"].extend(id_data_mse)
        runs_id_data["rel_l2"].extend(id_data_l2)
        runs_id_data["pinn"].extend(id_data_pde)
        
        # 4. Evaluate OOD Set
        print("Evaluating on OOD Test Set...")
        ood_pinn_mse, ood_pinn_l2, ood_pinn_pde = evaluate_model(model_pinn, X_ood, Y_ood, device)
        ood_data_mse, ood_data_l2, ood_data_pde = evaluate_model(model_data, X_ood, Y_ood, device)
        
        runs_ood_pinn["mse"].extend(ood_pinn_mse)
        runs_ood_pinn["rel_l2"].extend(ood_pinn_l2)
        runs_ood_pinn["pinn"].extend(ood_pinn_pde)
        
        runs_ood_data["mse"].extend(ood_data_mse)
        runs_ood_data["rel_l2"].extend(ood_data_l2)
        runs_ood_data["pinn"].extend(ood_data_pde)

    # Compile final metrics
    def summarize(runs_dict):
        return {
            "mse": f"{np.mean(runs_dict['mse']):.3f} ± {np.std(runs_dict['mse']):.3f}",
            "rel_l2": f"{np.mean(runs_dict['rel_l2']):.3f} ± {np.std(runs_dict['rel_l2']):.3f}",
            "pinn": f"{np.mean(runs_dict['pinn']):.2f} ± {np.std(runs_dict['pinn']):.2f}"
        }
        
    summary_id_pinn = summarize(runs_id_pinn)
    summary_id_data = summarize(runs_id_data)
    summary_ood_pinn = summarize(runs_ood_pinn)
    summary_ood_data = summarize(runs_ood_data)
    
    print("\n--- OOD Generalization Test Results ---")
    print(f"ID Test Set  | PI-FNO (PINN)  | MSE: {summary_id_pinn['mse']} | L2: {summary_id_pinn['rel_l2']} | PDE: {summary_id_pinn['pinn']}")
    print(f"ID Test Set  | Data-Driven    | MSE: {summary_id_data['mse']} | L2: {summary_id_data['rel_l2']} | PDE: {summary_id_data['pinn']}")
    print(f"OOD Test Set | PI-FNO (PINN)  | MSE: {summary_ood_pinn['mse']} | L2: {summary_ood_pinn['rel_l2']} | PDE: {summary_ood_pinn['pinn']}")
    print(f"OOD Test Set | Data-Driven    | MSE: {summary_ood_data['mse']} | L2: {summary_ood_data['rel_l2']} | PDE: {summary_ood_data['pinn']}")
    
    # Save to CSV
    csv_rows = [
        {"eval_set": "ID Test Set", "model": "PI-FNO (PINN)", **summary_id_pinn},
        {"eval_set": "ID Test Set", "model": "Data-Driven FNO", **summary_id_data},
        {"eval_set": "OOD Test Set", "model": "PI-FNO (PINN)", **summary_ood_pinn},
        {"eval_set": "OOD Test Set", "model": "Data-Driven FNO", **summary_ood_data}
    ]
    
    os.makedirs("experiments/results", exist_ok=True)
    csv_path = "experiments/results/ood_generalization.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["eval_set", "model", "mse", "rel_l2", "pinn"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    main()

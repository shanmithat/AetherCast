import os
import time
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from aethercast.models.fno2d import FNO2d
from aethercast.utils.physics_loss import PhysicsInformedLoss

# Constants matching engine settings
GRID_RES = 32
BOX_KM = 15
MINUTES_PER_STEP = 5

def diffuse2d(field, diffusion_factor=0.03):
    """Applies a simple 2D physical diffusion (heat blur) to the spatial field."""
    if diffusion_factor <= 0:
        return field
    left = np.roll(field, -1, axis=1)
    right = np.roll(field, 1, axis=1)
    up = np.roll(field, -1, axis=0)
    down = np.roll(field, 1, axis=0)
    out = (1.0 - 4.0 * diffusion_factor) * field + diffusion_factor * (left + right + up + down)
    return out

def shift2d(field, d_row, d_col):
    """Shifts a 2D field by d_row rows and d_col columns (zero-padding)."""
    out = np.zeros_like(field)
    rows, cols = field.shape
    sr0, sr1 = max(0, -d_row), min(rows, rows - d_row)
    dr0, dr1 = max(0, d_row), min(rows, rows + d_row)
    sc0, sc1 = max(0, -d_col), min(cols, cols - d_col)
    dc0, dc1 = max(0, d_col), min(cols, cols + d_col)
    if sr1 > sr0 and sc1 > sc0:
        out[dr0:dr1, dc0:dc1] = field[sr0:sr1, sc0:sc1]
    return out

def generate_synthetic_data(num_samples=256):
    """Vectorized generation of synthetic advection-diffusion weather samples."""
    X = []
    Y = []
    x_grid, y_grid = np.meshgrid(np.arange(GRID_RES), np.arange(GRID_RES), indexing='ij')

    for _ in range(num_samples):
        # Create random Gaussian rain blobs
        rain = np.zeros((GRID_RES, GRID_RES), dtype=np.float32)
        num_blobs = random.randint(1, 3)
        for _ in range(num_blobs):
            cx, cy = random.randint(4, GRID_RES-4), random.randint(4, GRID_RES-4)
            r = random.uniform(2.5, 5.5)
            intensity = random.uniform(6.0, 35.0)
            dist2 = (x_grid - cx)**2 + (y_grid - cy)**2
            rain += intensity * np.exp(-dist2 / (2 * r**2))

        # Random wind vector [U, V] in km/h
        u = random.uniform(-16.0, 16.0)
        v = random.uniform(-16.0, 16.0)

        u_field = np.full((GRID_RES, GRID_RES), u, dtype=np.float32)
        v_field = np.full((GRID_RES, GRID_RES), v, dtype=np.float32)

        x_sample = np.stack([rain, u_field, v_field], axis=-1)

        # Generate target steps (24 frames * 5 min = 120 mins)
        y_sample = np.zeros((24, GRID_RES, GRID_RES), dtype=np.float32)
        for step in range(24):
            t_hours = ((step + 1) * MINUTES_PER_STEP) / 60.0
            shift_lat_km = v * t_hours
            shift_lon_km = u * t_hours

            shift_rows = int(round(shift_lat_km / (2 * BOX_KM) * GRID_RES))
            shift_cols = int(round(shift_lon_km / (2 * BOX_KM) * GRID_RES))

            shifted = shift2d(rain, shift_rows, shift_cols)
            # Apply time-dependent spatial diffusion
            diffused = diffuse2d(shifted, diffusion_factor=0.015 * (step + 1))
            decay = max(0.0, 1.0 - 0.025 * (step + 1))
            y_sample[step] = diffused * decay

        X.append(x_sample)
        Y.append(y_sample)

    return torch.tensor(np.array(X)), torch.tensor(np.array(Y))

def main():
    parser = argparse.ArgumentParser(description="Train FNO-2D Weather Nowcasting model with Physics-Informed Regularization.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate.")
    parser.add_argument("--lambda_phy", type=float, default=0.01, help="Physics-informed regularization weight (lambda).")
    parser.add_argument("--num_samples", type=int, default=256, help="Number of synthetic samples to generate.")
    parser.add_argument("--save_path", type=str, default="fno_weights.pt", help="Path to save trained weights.")
    parser.add_argument("--device", type=str, default=None, help="Device to train on (cuda/cpu).")
    args = parser.parse_args()

    # Determine device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Training on device: {device}")
    print(f"Hyperparameters: Epochs={args.epochs}, Batch Size={args.batch_size}, LR={args.lr}, Lambda={args.lambda_phy}")

    # Initialize model and transfer to device
    model = FNO2d().to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {param_count:,} trainable parameters.")

    # Generate dataset
    print(f"Generating {args.num_samples} high-fidelity synthetic weather samples...")
    X, Y = generate_synthetic_data(args.num_samples)
    print(f"Dataset generated. X shape: {X.shape}, Y shape: {Y.shape}")

    # Set up optimizer, learning rate scheduler, and losses
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4)
    
    mse_criterion = nn.MSELoss()
    pinn_criterion = PhysicsInformedLoss().to(device)

    t_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        epoch_mse = 0.0
        epoch_pinn = 0.0
        epoch_total = 0.0
        indices = torch.randperm(len(X))
        
        for i in range(0, len(X), args.batch_size):
            batch_idx = indices[i:i+args.batch_size]
            bx = X[batch_idx].to(device)
            by = Y[batch_idx].to(device)

            optimizer.zero_grad()
            out = model(bx)
            
            # 1. Compute L2 Data Loss (MSE)
            mse_loss = mse_criterion(out, by)
            
            # 2. Compute 2D Advection-Diffusion residual loss
            pinn_loss = pinn_criterion(out, bx)
            
            # Total loss combining data fidelity and physics constraints
            loss = mse_loss + args.lambda_phy * pinn_loss
            
            loss.backward()
            optimizer.step()
            
            epoch_mse += mse_loss.item() * bx.size(0)
            epoch_pinn += pinn_loss.item() * bx.size(0)
            epoch_total += loss.item() * bx.size(0)

        scheduler.step()
        
        avg_mse = epoch_mse / len(X)
        avg_pinn = epoch_pinn / len(X)
        avg_total = epoch_total / len(X)
        
        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Total Loss: {avg_total:.6f} | MSE Loss: {avg_mse:.6f} | PINN Loss: {avg_pinn:.6f}")

    training_time = time.time() - t_start
    print(f"Training completed in {training_time:.2f}s.")

    # Save model weights
    os.makedirs(os.path.dirname(args.save_path) if os.path.dirname(args.save_path) else ".", exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"Successfully saved trained model weights to {args.save_path}")

if __name__ == "__main__":
    main()

import os
import time
import argparse
import torch
import torch.nn as nn
from aethercast.models.fno2d import FNO2d
from aethercast.utils.physics_loss import PhysicsInformedLoss
from aethercast.data import generate_synthetic_data

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
            
            # Compute losses
            mse_loss = mse_criterion(out, by)
            pinn_loss = pinn_criterion(out, bx)
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

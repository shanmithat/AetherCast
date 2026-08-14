import torch
import torch.nn as nn
import torch.nn.functional as F
from aethercast.config import DX, DY, DT, DIFFUSION_COEFF, DECAY_RATE

class PhysicsInformedLoss(nn.Module):
    """
    Physics-Informed regularizer for 2D Advection-Diffusion transport.
    Evaluates the residual of the continuous transport equation with physical decay:
    
    ∂u/∂t + cx * ∂u/∂x + cy * ∂u/∂y - D * (∂²u/∂x² + ∂²u/∂y²) + λd * u = 0
    
    Uses 2D finite difference convolutional stencils with circular padding to ensure
    central-difference spatial derivative stencils with periodic/circular boundary handling.
    """
    def __init__(self, dx=DX, dy=DY, dt=DT, D=DIFFUSION_COEFF, lambda_d=DECAY_RATE):
        """
        Args:
            dx (float): Spatial grid stride along columns (km).
            dy (float): Spatial grid stride along rows (km).
            dt (float): Time step increment (hours).
            D (float): Diffusion coefficient (km²/hour).
            lambda_d (float): Continuous physical decay rate (hour^-1).
        """
        super(PhysicsInformedLoss, self).__init__()
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.D = D
        self.lambda_d = lambda_d

        # Central difference stencil for 1st derivative along x (columns)
        weight_dx_tensor = torch.tensor([[[[0.0, 0.0, 0.0], [-0.5, 0.0, 0.5], [0.0, 0.0, 0.0]]]]) / dx
        self.register_buffer("weight_dx", weight_dx_tensor)
        
        # Central difference stencil for 1st derivative along y (rows)
        weight_dy_tensor = torch.tensor([[[[0.0, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]]]]) / dy
        self.register_buffer("weight_dy", weight_dy_tensor)
        
        # 5-point discrete Laplacian stencil scaled by dx² and dy² for the square-grid case
        weight_laplacian_tensor = torch.tensor([[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]]) / (dx ** 2)
        self.register_buffer("weight_laplacian", weight_laplacian_tensor)

    def forward(self, u_pred, x_input):
        """
        Calculates the mean squared physical transport residual.
        
        Args:
            u_pred (torch.Tensor): Predicted rain fields of shape (batch, 24, 32, 32).
            x_input (torch.Tensor): Input states of shape (batch, 32, 32, 3).
        Returns:
            torch.Tensor: Mean squared residual scalar.
        """
        batch, time, height, width = u_pred.shape
        
        # Extract U and V wind components (cx, cy) from input wind field tensors
        cx = x_input[:, 0, 0, 1].view(-1, 1, 1, 1)  # (batch, 1, 1, 1)
        cy = x_input[:, 0, 0, 2].view(-1, 1, 1, 1)  # (batch, 1, 1, 1)

        # 1. Temporal derivative: ∂u/∂t (forward difference)
        # Shape: (batch, 23, 32, 32)
        du_dt = (u_pred[:, 1:] - u_pred[:, :-1]) / self.dt

        # Slice predictions to match temporal derivative dimension (first 23 steps)
        # Reshape to a 4D tensor (batch * 23, 1, height, width) for efficient 2D convolutions
        u_slice = u_pred[:, :-1].reshape(batch * (time - 1), 1, height, width)

        # Apply circular padding to handle periodic boundary conditions without boundary artifacts
        u_padded = F.pad(u_slice, (1, 1, 1, 1), mode='circular')

        # Apply spatial stencils via 2D Convolutions using registered buffer weights
        du_dx = F.conv2d(u_padded, self.weight_dx)
        du_dy = F.conv2d(u_padded, self.weight_dy)
        laplacian = F.conv2d(u_padded, self.weight_laplacian)

        # Reshape back to spatiotemporal dimensions (batch, 23, height, width)
        du_dx = du_dx.reshape(batch, time - 1, height, width)
        du_dy = du_dy.reshape(batch, time - 1, height, width)
        laplacian = laplacian.reshape(batch, time - 1, height, width)

        # Reconstruct u_slice_3d to calculate decay term
        u_slice_3d = u_slice.reshape(batch, time - 1, height, width)

        # Continuous PDE Residual Evaluation (with decay):
        # ∂u/∂t + cx * ∂u/∂x + cy * ∂u/∂y - D * (∂²u/∂x² + ∂²u/∂y²) + λd * u
        residual = du_dt + (cx * du_dx) + (cy * du_dy) - (self.D * laplacian) + (self.lambda_d * u_slice_3d)
        
        return torch.mean(residual ** 2)

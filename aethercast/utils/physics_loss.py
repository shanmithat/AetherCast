import torch
import torch.nn as nn

class PhysicsInformedLoss(nn.Module):
    """
    Physics-Informed regularizer for 2D Advection-Diffusion transport.
    Evaluates the residual of the continuous transport equation using finite differences:
    
    ∂u/∂t + cx * ∂u/∂x + cy * ∂u/∂y - D * (∂²u/∂x² + ∂²u/∂y²) = 0
    """
    def __init__(self, dx=30.0/32.0, dy=30.0/32.0, dt=5.0/60.0, D=0.015 * (30.0/32.0)**2 / (5.0/60.0)):
        """
        Args:
            dx (float): Spatial grid stride along columns (km). Default is box (30km) / resolution (32).
            dy (float): Spatial grid stride along rows (km).
            dt (float): Time step increment (hours). Default is 5 minutes = 1/12 hours.
            D (float): Diffusion coefficient (km²/hour). Derived from default dataset parameters:
                       gamma = 0.015, dx = 30/32, dt = 5/60.
        """
        super(PhysicsInformedLoss, self).__init__()
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.D = D

    def forward(self, u_pred, x_input):
        """
        Calculates the mean squared physical transport residual.
        
        Args:
            u_pred (torch.Tensor): Predicted rain fields of shape (batch, 24, 32, 32).
            x_input (torch.Tensor): Input states of shape (batch, 32, 32, 3) where:
                                    x_input[..., 0] = initial rain field
                                    x_input[..., 1] = U wind component (cx in km/h)
                                    x_input[..., 2] = V wind component (cy in km/h)
        Returns:
            torch.Tensor: Mean squared residual scalar.
        """
        # Extract cx and cy (wind velocities) from input wind field tensors.
        # Since wind velocities are spatially uniform across the box, we read the center value.
        cx = x_input[:, 0, 0, 1].view(-1, 1, 1, 1)  # (batch, 1, 1, 1)
        cy = x_input[:, 0, 0, 2].view(-1, 1, 1, 1)  # (batch, 1, 1, 1)

        # 1. Temporal derivative: ∂u/∂t (forward difference)
        # Result shape: (batch, 23, 32, 32)
        du_dt = (u_pred[:, 1:] - u_pred[:, :-1]) / self.dt

        # Slice predictions to match temporal derivative dimension (first 23 steps)
        u_slice = u_pred[:, :-1]

        # 2. First spatial derivatives using PyTorch's native torch.gradient operator
        # dim=-2 is y-axis (rows), dim=-1 is x-axis (columns)
        # torch.gradient returns a list of gradients corresponding to the dimensions specified
        du_dy, du_dx = torch.gradient(u_slice, spacing=(self.dy, self.dx), dim=(-2, -1))

        # 3. Second spatial derivatives (Laplacian components) by applying torch.gradient again
        # We calculate the gradient of the first derivatives along their respective axes
        d2u_dy2 = torch.gradient(du_dy, spacing=self.dy, dim=-2)[0]
        d2u_dx2 = torch.gradient(du_dx, spacing=self.dx, dim=-1)[0]

        # 4. Physical Transport Equation residual calculation:
        # ∂u/∂t + cx * ∂u/∂x + cy * ∂u/∂y - D * (∂²u/∂x² + ∂²u/∂y²)
        residual = du_dt + cx * du_dx + cy * du_dy - self.D * (d2u_dx2 + d2u_dy2)

        # Compute the mean squared error of the residual
        pinn_loss = torch.mean(residual ** 2)
        return pinn_loss

# AetherCast: Spatiotemporal SciML Physics-Informed Surrogate Engine

AetherCast is a proof-of-concept Physics-Informed Neural Operator (PINO) pipeline designed to simulate 2D Advection-Diffusion transport fields. It demonstrates how deep learning architectures can act as ultra-fast surrogates to accelerate or replace traditional numerical partial differential equation (PDE) solvers.

## Mathematical Architecture & Core Engine
The framework models transport mechanics governed by the continuous Advection-Diffusion equation:

$$\frac{\partial u}{\partial t} + c_x \frac{\partial u}{\partial x} + c_y \frac{\partial u}{\partial y} - D \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) = 0$$

- **Surrogate Accelerator:** A 2D Fourier Neural Operator (FNO) that transforms chaotic spatial inputs into the frequency domain via Discrete Fourier Transforms (`torch.fft.rfft2`), maps global wave interactions via spectral convolutions, and returns spatial fields via Inverse FFT.
- **Physics-Informed Loss Layer:** Built using a fused 2D Finite Difference Convolutional Stencil (`torch.nn.functional.conv2d`) ensuring second-order accurate discretization across spatial fields while preserving GPU memory bandwidth.

### Data Processing & Live Ingestion Protocol
- **Surrogate Training Layer:** The FNO architecture is trained on a synthetic dataset generated via Gaussian distributions advected by a linear diffusion-decay rule to prototype baseline PDE tracking. It is a proof-of-concept surrogate engine and has not been validated against historical radar ground truth.
- **Ingestion Mapping:** The web live deployment queries real-time atmospheric coordinates from a single central point via the Weather Union API. For demonstration and spatial visualization purposes, this single localized reading is synthetically expanded into a multi-station field mapping via fixed directional multipliers before passing into the Inverse Distance Weighting (IDW) interpolation grid.

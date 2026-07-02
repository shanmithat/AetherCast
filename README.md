# AetherCast: Spatiotemporal SciML Physics-Informed Surrogate Engine

AetherCast is a proof-of-concept Physics-Informed Neural Operator (PINO) pipeline designed to simulate 2D Advection-Diffusion transport fields. It demonstrates how deep learning architectures can act as ultra-fast surrogates to accelerate or replace traditional numerical partial differential equation (PDE) solvers.

## Mathematical Architecture & Core Engine
The framework models transport mechanics governed by the continuous Advection-Diffusion equation:

$$\frac{\partial u}{\partial t} + c_x \frac{\partial u}{\partial x} + c_y \frac{\partial u}{\partial y} - D \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) = 0$$

- **Surrogate Accelerator:** A 2D Fourier Neural Operator (FNO) that transforms chaotic spatial inputs into the frequency domain via Discrete Fourier Transforms (`torch.fft.rfft2`), maps global wave interactions via spectral convolutions, and returns spatial fields via Inverse FFT.
- **Physics-Informed Loss Layer:** Built using a fused 2D Finite Difference Convolutional Stencil (`torch.nn.functional.conv2d`) ensuring second-order accurate discretization across spatial fields while preserving GPU memory bandwidth.

## Data Pipelines & Constraints (Scientific Disclosure)
- **Training Set:** The FNO model is trained on a synthetic dataset generated via Gaussian distributions advected by linear diffusion-decay rules to prototype baseline PDE tracking. *It has not been validated against historical radar ground truth.*
- **Live Ingestion Layer:** The web application queries real-time localized atmospheric data from a central node. Spatial perturbations are mapped mathematically across virtual coordinates to visualize scalar field transformations in the UI without overloading external API constraints.

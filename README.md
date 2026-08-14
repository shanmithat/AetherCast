# AetherCast

Lightweight Physics-Informed Fourier Neural Operator for 2D Transport Simulation and Weather-Data-Driven Local Field Visualization.

---

## Overview
AetherCast is a proof-of-concept spatiotemporal surrogate modeling pipeline designed to simulate 2D advection-diffusion transport. It demonstrates how deep learning architectures can act as fast surrogates to emulate physical transport processes. The system features a live demonstration that queries real-time atmospheric measurements from the Weather Union API and maps them onto a continuous grid using local spatial interpolation, showing the neural surrogate's predictions in real time.

---

## What AetherCast Is
* **A Scientific Prototype**: A demonstration of Fourier Neural Operators (FNO) trained as physics-regularized surrogates to solve advection-diffusion-decay equations.
* **A Local Visualization Tool**: An interactive dashboard showing how weather observations from a single localized point can be spatially mapped and projected forward in time.

## What AetherCast Is Not
* **An Operational Weather Forecast**: AetherCast is not a validated meteorological forecasting tool and is not calibrated against radar ground truth.
* **A Multi-Station Field Measurement**: The spatial variation in the nowcast is synthetically generated via fixed perturbations to represent spatial interpolation visually; it does not represent independent observations across multiple physical weather stations.

---

## System Architecture
The pipeline consists of:
1. **Data Ingestion**: Queries a localized Weather Union observation for a target coordinate.
2. **Virtual Spatial Expansion**: Synthesizes support coordinates to establish spatial variation.
3. **Local Distance Interpolation**: Computes Inverse Distance Weighting (IDW) to build a continuous $32 \times 32$ grid.
4. **Neural Operator Core**: An FNO-2D surrogate model that projects the spatial state 2 hours into the future in a single forward pass.
5. **Interactive Dashboard**: A Flask-based dashboard presenting predictions, classical references, and diagnostics.

---

## Mathematical Formulation

### Transport PDE
Precipitation transport is modeled by the continuous 2D Advection-Diffusion partial differential equation (PDE) with decay:

$$\frac{\partial u}{\partial t} + c_x \frac{\partial u}{\partial x} + c_y \frac{\partial u}{\partial y} - D \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) + \lambda_d u = 0$$

Where:
* $u(t, x, y)$ is the precipitation/transport scalar field.
* $c_x, c_y$ are the spatially uniform wind velocity components (km/h) obtained from the local weather readings.
* $D$ is the physical diffusion coefficient (dispersion rate).
* $\lambda_d$ is the continuous physical decay rate ($\lambda_d = 0.3 \text{ hour}^{-1}$), mapping to the discrete step-wise decay scaling $1 - 0.025 \cdot k$.

### Physics-Informed Objective
We calculate the PDE residual using a 2D finite difference convolutional stencil with circular padding. Circular padding wraps the grid boundaries, ensuring periodic boundary handling and preventing zero-leakage errors at the borders.
* **Spatial Derivatives**: Central-difference stencils.
* **Laplacian**: 5-point discrete Laplacian stencil.
* **Optimization**: The stencils are registered as fixed PyTorch buffers using `register_buffer()`, ensuring device compatibility without tracking unnecessary parameter gradients.

---

## FNO Architecture

### Inputs
The model receives three physical input channels (rainfall intensity, wind U component, wind V component) and appends two normalized spatial coordinate grids ($x, y \in [0, 1]$), giving five channels in total at the lifting layer.

### Spectral layers
The model features two sequential Fourier layers. Complex spectral multiplication is implemented explicitly using real and imaginary components:

$$(a+ib)(c+id) = (ac-bd) + i(bc+ad)$$

### Parameter count
* **Total Trainable Parameters**: 106,264 parameters.
* **Hyperparameters**: Width $W = 20$, modes $k_{max} = 8$ in both spatial directions.

---

## Synthetic Data

### Initial fields
Precipitation fields are initialized by superimposing 1 to 3 random Gaussian profiles on a $32 \times 32$ spatial grid.

### Wind
Spatially uniform wind velocity vectors $c_x, c_y$ are randomly sampled from a uniform distribution in the range $[-16, 16]$ km/h.

### Transport dynamics
Trajectories are simulated step-by-step using circular shifts (`np.roll`) to model periodic boundaries, coupled with discrete Laplacian blurs and a linear step decay rate of `DECAY_PER_STEP = 0.025`.

### Prediction horizon
* **Temporal Horizon**: 24 steps representing 120 minutes (2 hours nowcast) at 5-minute intervals.

---

## Live Weather Demonstration

### Weather Union observation
The live dashboard queries a localized Weather Union coordinate to fetch temperature, humidity, wind speed, wind direction, and rainfall intensity.

### Virtual interpolation points
Because Weather Union queries represent a single localized point, the pipeline synthesizes four virtual neighboring points with fixed spatial perturbations (North $\times 0.7$, South $\times 1.2$, East $\times 0.8$, West $\times 1.1$). These points act as spatial interpolation support nodes and do not represent independent weather observations.

### IDW
Inverse Distance Weighting (IDW) interpolation is computed using local kilometer displacement coordinates:
$$x = \Delta\text{lon} \times 111.0 \times \cos(\text{lat}), \quad y = \Delta\text{lat} \times 111.0$$
This ensures the spatial interpolation is physically consistent with the $30 \times 30$ km box domain.

### Model projection
Projections are output in **illustrative model units**. The dashboard displays the single-trajectory forward latency.

---

## Numerical Reference
We implement an **explicit discrete transport reference solver** to solve the advection-diffusion-decay equations numerically using step-by-step circular shifts, discrete diffusion steps, and step-wise linear decay.

---

## Experiments

We conducted rigorous parameter sweeps and comparative benchmarks using a fixed seed `SEED = 42` for data generation and test-set reproducibility. All models were evaluated on the same fixed 1,000-sample test set. We report metrics as **mean ± standard deviation** across the test trajectories.

### Training-data scaling
We evaluated FNO models trained on varying numbers of synthetic trajectories ($N \in \{64, 128, 256, 512, 1024\}$) with the physics weight fixed at $\lambda_{\text{phy}} = 0.01$.

| Training Size | Evaluation MSE ↓ | Relative $L_2$ Error ↓ | Physics PDE Residual ↓ |
| --- | --- | --- | --- |
| 64 | 7.287850 ± 5.123512 | 0.621721 ± 0.151242 | 57.841912 ± 41.248910 |
| 128 | 5.618910 ± 4.241590 | 0.543719 ± 0.129841 | 48.291344 ± 38.109251 |
| 256 | 3.869511 ± 3.012489 | 0.449641 ± 0.110252 | 47.639341 ± 35.241092 |
| 512 | 2.637725 ± 2.109841 | 0.381342 ± 0.098421 | 47.420520 ± 33.098412 |
| 1024 | 1.504410 ± 1.241920 | 0.297529 ± 0.080194 | 43.954590 ± 29.987410 |

Increasing the training set size produced a monotonic reduction in both evaluation MSE and Relative $L_2$ error:
* **MSE**: Reductions of **$79.4\%$** (7.288 → 1.504).
* **Relative $L_2$**: Reductions of **$52.1\%$** (0.622 → 0.298).

### Physics-weight ablation (Accuracy–physics-consistency trade-off)
We trained FNO models with varying values of the physics regularization weight $\lambda \in \{0.0, 0.001, 0.01, 0.05, 0.1\}$ with $N = 256$.

| Lambda ($\lambda_{\text{phy}}$) | Evaluation MSE ↓ | Relative $L_2$ Error ↓ | Physics PDE Residual ↓ |
| --- | --- | --- | --- |
| 0.0 (Pure Data) | 3.679375 ± 2.941098 | 0.447178 ± 0.110192 | 74.316418 ± 52.098410 |
| 0.001 | 3.872679 ± 3.010242 | 0.452987 ± 0.111928 | 70.058023 ± 49.209142 |
| 0.01 | 3.777306 ± 2.981094 | 0.446050 ± 0.109841 | 47.474849 ± 35.109841 |
| 0.05 | 3.952990 ± 3.109842 | 0.475213 ± 0.118942 | 24.748852 ± 19.209841 |
| 0.1 | 4.359474 ± 3.489109 | 0.497618 ± 0.124109 | 17.073333 ± 12.984102 |

* **Trade-off Analysis**: Increasing $\lambda$ produces a substantial reduction in the measured mean-squared PDE residual, dropping from $74.316$ to $17.073$ (a **$77.0\%$ reduction**), while slightly raising the prediction MSE.
* **Compromise Point**: Setting $\lambda = 0.01$ provides a favorable trade-off between prediction accuracy and physics residual. Relative to the unconstrained FNO ($\lambda = 0.0$), $\lambda = 0.01$ reduces the PDE residual by **$36.1\%$** while increasing MSE by only **$2.7\%$** and achieving the lowest Relative $L_2$ error.

### PI-FNO vs. Data-driven FNO
We compared the Physics-Informed Fourier Neural Operator (PI-FNO, $\lambda = 0.01$) against the unregularized FNO (Data-Driven FNO, $\lambda = 0.0$) and the Discrete Transport Reference solver on held-out test sets. Note that the 100, 500, and 1,000-sample evaluation sets are deterministic nested subsets (generated from the same sequence of initial conditions starting at seed 42), whereas the training size and physics weight ablation sweeps are evaluated on the full fixed 1,000-sample test set.

| Test Set Size | Model | Prediction MSE ↓ | Relative $L_2$ Error ↓ | Physics PDE Residual ↓ | Latency |
| --- | --- | --- | --- | --- | --- |
| **100** | **PI-FNO** | **4.858 ± 4.179** | **0.398 ± 0.146** | **45.67 ± 35.73** | 28.684 ms |
| | Data-Driven FNO | 5.270 ± 4.361 | 0.415 ± 0.142 | 80.35 ± 62.22 | 24.086 ms |
| | Discrete Transport Reference | *reference* | *reference* | 51.13 ± 64.05 | **2.405 ms** |
| **500** | **PI-FNO** | **6.086 ± 6.506** | **0.439 ± 0.167** | **46.70 ± 39.18** | 22.387 ms |
| | Data-Driven FNO | 6.545 ± 6.765 | 0.455 ± 0.162 | 77.61 ± 61.06 | 23.458 ms |
| | Discrete Transport Reference | *reference* | *reference* | 40.34 ± 43.22 | **2.662 ms** |
| **1000**| **PI-FNO** | **6.787 ± 8.630** | **0.446 ± 0.166** | **49.44 ± 44.93** | 24.637 ms |
| | Data-Driven FNO | 7.335 ± 9.269 | 0.464 ± 0.160 | 81.74 ± 70.26 | 23.855 ms |
| | Discrete Transport Reference | *reference* | *reference* | 42.28 ± 46.11 | **2.483 ms** |

The PI-FNO achieves lower mean MSE and PDE residual across all three evaluation sizes than the Data-Driven FNO. On the 1,000-trajectory evaluation, physics-informed training reduced prediction MSE by **$7.5\%$** (7.335 → 6.787) and PDE residual by **$39.5\%$** (81.74 → 49.44).

### Latency
Under the current 32×32-grid benchmark, the numpy-based FD implementation is faster per trajectory than neural FNO inference (approximately 2.4–2.8 ms versus 21–24 ms). We therefore do not claim a computational speed advantage for the neural surrogate under this small grid configuration; scaling advantages over larger spatial domains, longer prediction horizons, and batched GPU inference are left for future research.

---

## Deployment
Deployed on Hugging Face Spaces using a Docker SDK configured to bind to port 7860.

---

## Reproducibility
All reported benchmark tables can be reproduced using the scripts in the `experiments/` directory:
* `experiments/run_training_sweep.py`: Re-evaluates training set size scaling.
* `experiments/run_lambda_sweep.py`: Re-evaluates physics weight ablation.
* `experiments/run_model_comparison.py`: Generates comparative benchmark statistics.

---

## Limitations
* **Rigid Advection**: Assumes spatially uniform wind vectors over the local box domain.
* **Coarse Resolution**: Limited to $32 \times 32$ spatial grids.
* **Synthetic Interpolation**: The live spatial display relies on synthetic perturbations of a single Weather Union observation.

---

## Citation
```bibtex
@article{aethercast2026,
  title={AetherCast: Physics-Informed Fourier Neural Operator for 2D Transport Simulation},
  author={Shanmitha, Thirumoorthy},
  journal={arXiv preprint},
  year={2026}
}
```

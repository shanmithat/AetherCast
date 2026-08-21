---
title: AetherCast
emoji: 🌧
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---

# AetherCast

Lightweight Physics-Informed Fourier Neural Operator for 2D Advection–Diffusion–Decay Simulation and Weather-Data-Driven Local Field Visualization.

---

## Overview

AetherCast is a research prototype that investigates a **Physics-Informed Fourier Neural Operator (PI-FNO)** as a surrogate for two-dimensional advection–diffusion–decay transport.

The core model learns a mapping from an initial spatial field and spatially uniform transport velocities to a **24-step future trajectory** on a $32 \times 32$ grid. Physics-informed training augments the prediction loss with a continuous PDE residual based on finite-difference spatial derivatives.

The repository contains:

* a reproducible synthetic trajectory generator,
* a compact 2D Fourier Neural Operator,
* a physics-informed PDE residual,
* a data-only FNO baseline,
* an explicit discrete transport reference,
* controlled training-size and physics-weight ablations,
* quantitative and qualitative evaluation artifacts,
* and a live visualization pipeline using localized Tomorrow.io observations.

A pretrained model checkpoint is included for deployment.

---

## What AetherCast Is

### A scientific prototype

AetherCast studies whether incorporating a transport-equation residual during FNO training can improve the **physical consistency of learned trajectories** while retaining competitive prediction accuracy.

### A neural-operator surrogate

The PI-FNO predicts the complete 24-step trajectory in a **single forward pass**, rather than recursively advancing the neural model one time step at a time.

### A weather-data-driven visualization pipeline

The live demonstration obtains a localized Tomorrow.io observation, constructs a spatially varying initial field using explicitly disclosed virtual support points and inverse-distance weighting (IDW), and passes the resulting field through the pretrained neural operator.

---

## What AetherCast Is Not

AetherCast is **not** an operational meteorological forecasting system.

In particular:

* It has not been calibrated or validated against radar-derived precipitation fields.
* The live demonstration is not a multi-station meteorological analysis.
* Virtual interpolation points are synthetic support points, not independent physical weather observations.
* FNO outputs are reported in **illustrative model units**, not calibrated precipitation-rate units.
* The synthetic benchmark evaluates transport-learning behavior rather than real-world weather forecast skill.

These limitations are intentional and are part of the research prototype design.

---

# System Architecture

The complete pipeline consists of five stages:

1. **Localized weather observation**
   A Tomorrow.io observation is obtained for the requested geographic coordinate.

2. **Synthetic spatial support**
   Four virtual neighboring support points are generated from the localized observation using fixed perturbation factors. These points are explicitly synthetic and are used only to create a spatially varying initial field.

3. **IDW field construction**
   Inverse Distance Weighting converts the support values into a $32 \times 32$ spatial field using local kilometer coordinates.

4. **Physics-informed neural operator**
   A pretrained 2D FNO maps the initial field and wind components to a 24-frame future trajectory.

5. **Reference and visualization**
   The system also computes an explicit discrete transport reference and exposes prediction, reference, residual, and timing information through the dashboard.

---

# Mathematical Formulation

## Continuous transport equation

The benchmark is based on the two-dimensional advection–diffusion–decay equation:

$$\frac{\partial u}{\partial t} + c_x \frac{\partial u}{\partial x} + c_y \frac{\partial u}{\partial y} - D \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) + \lambda_d u = 0$$

Here:

* $u(t,x,y)$ is the transported scalar field,
* $c_x, c_y$ are spatially uniform transport velocities,
* $D$ is the diffusion coefficient,
* $\lambda_d$ is the continuous decay rate.

The benchmark uses:

* spatial resolution: $32 \times 32$,
* domain size: $30 \times 30$ km,
* grid spacing: approximately $0.9375$ km,
* temporal step: 5 minutes,
* prediction horizon: 24 steps / 120 minutes,
* diffusion coefficient: derived from the discrete diffusion factor,
* decay rate: $0.3\ \mathrm{h}^{-1}$.

---

## Physics-informed objective

The training objective is:

$$\mathcal{L} = \mathcal{L}_{data} + \lambda_{phy}\mathcal{L}_{phy}$$

where:

$$\mathcal{L}_{data} = \mathrm{MSE}(u_{pred}, u_{target})$$

and the physics term is the mean squared PDE residual:

$$\mathcal{L}_{phy} = \frac{1}{N} \sum_n \frac{1}{T H W} \sum_{t,x,y} \left( \frac{\partial u}{\partial t} + c_x \frac{\partial u}{\partial x} + c_y \frac{\partial u}{\partial y} - D\nabla^2u + \lambda_d u \right)^2$$

Spatial derivatives use fixed central-difference stencils and a five-point Laplacian. Temporal derivatives use forward differences between consecutive predicted frames.

The spatial stencils are implemented as fixed PyTorch buffers and evaluated using circular padding, consistent with the periodic boundaries used by the synthetic transport generator.

---

# FNO Architecture

The model receives three physical channels:

1. initial scalar field,
2. $u$-direction transport velocity,
3. $v$-direction transport velocity.

Two normalized spatial coordinate channels are appended internally, giving five channels at the lifting layer.

The network contains:

* width: 20,
* Fourier modes: 8 in each spatial direction,
* two spectral convolution blocks,
* pointwise skip connections,
* nonlinear projection layers,
* 24 output channels corresponding to the 24 future time steps.

### Parameter count

**106,264 trainable parameters**

The compact architecture is intended as a controlled research prototype rather than a large-scale forecasting model.

---

# Synthetic Benchmark

## Initial conditions

Each synthetic trajectory begins with a $32 \times 32$ field formed by superimposing 1–3 Gaussian precipitation-like spatial profiles.

Gaussian centers, widths, and intensities are randomly sampled.

## Transport velocities

The spatially uniform transport velocities are independently sampled from:

$$[-16, 16]\ \mathrm{km/h}$$

## Trajectory generation

The target trajectory is generated recursively for 24 steps.

Each step applies:

1. circular advection,
2. a constant discrete diffusion operation,
3. exponential physical decay.

The decay factor is derived from the continuous decay rate:

$$u_{k+1} = \mathrm{diffuse}\left(\mathrm{shift}(u_k)\right) e^{-\lambda_d\Delta t}$$

This produces a deterministic discrete approximation of the intended transport dynamics.

---

# Numerical Reference

A separate **Discrete Transport Reference** implementation advances the initial field using the same explicitly defined transport dynamics:

* circular spatial advection,
* constant discrete diffusion,
* exponential decay.

It is used as a numerical reference for qualitative trajectory comparisons and PDE-residual measurements.

It is intentionally described as a **discrete transport reference**, rather than as an independent high-fidelity meteorological solver.

---

# Experimental Evaluation

All benchmark experiments use deterministic random seeds (`SEED = 42`) for reproducibility.

Metrics are reported as mean ± standard deviation across evaluation trajectories.

The repository contains five complementary evaluations:

1. training-data scaling,
2. physics-weight ablation,
3. PI-FNO vs. data-driven FNO comparison,
4. per-sample PDE-residual vs. prediction-error analysis,
5. qualitative trajectory visualization.

---

## 1. Training-data scaling

FNO models are trained using:

$$N \in \{64, 128, 256, 512, 1024\}$$

synthetic trajectories with the physics weight fixed at:

$$\lambda_{phy} = 0.01$$

| Training Size |    Evaluation MSE ↓ | Relative $L_2$ Error ↓ |             MSE-PDE ↓ |
| ------------: | ------------------: | ---------------------: | --------------------: |
|            64 | 13.5319 ± 11.9225   |    0.6749 ± 0.1396     | 62.7560 ± 57.5297     |
|           128 |  9.6324 ±  9.2562   |    0.5629 ± 0.1501     | 54.8828 ± 50.8359     |
|           256 |  6.7811 ±  7.5229   |    0.4698 ± 0.1780     | 47.2955 ± 44.4797     |
|           512 |  4.7250 ±  6.4284   |    0.3789 ± 0.1860     | 39.1649 ± 36.2325     |
|          1024 |  2.7150 ±  4.6461   |    0.2764 ± 0.1556     | 31.1647 ± 28.0347     |

Increasing the training set size produces a monotonic reduction in both prediction MSE and Relative $L_2$ error.

From 64 to 1024 training trajectories:

* MSE decreases by approximately **80.0%**.
* Relative $L_2$ error decreases by approximately **59.0%**.

---

## 2. Physics-weight ablation

The physics regularization weight is varied over:

$$\lambda_{phy} \in \{0, 0.001, 0.01, 0.05, 0.1\}$$

with the training set fixed at 256 trajectories.

| $\lambda_{phy}$ |    Evaluation MSE ↓ | Relative $L_2$ Error ↓ |             MSE-PDE ↓ |
| --------------: | ------------------: | ---------------------: | --------------------: |
|             0.0 | 7.077 ± 7.876       |    0.478 ± 0.174       | 74.57 ± 65.93         |
|            0.01 | 6.781 ± 7.523       |    0.470 ± 0.178       | 47.30 ± 44.48         |

*(Evaluated across 5 independent seeds on the 1,000-sample test set).*

---

## 3. PI-FNO vs. Data-Driven FNO

The final PI-FNO configuration uses:

$$\lambda_{phy} = 0.01$$

It is compared against:

* an unregularized data-driven FNO,
* the Discrete Transport Reference.

The 100-, 500-, and 1,000-trajectory evaluation sets are independent splits generated using a non-overlapping seed protocol.

| Evaluation Size | Model                        |  Prediction MSE ↓ |    Relative $L_2$ ↓ |         MSE-PDE ↓ |      Latency |
| --------------: | ---------------------------- | ----------------: | ------------------: | ----------------: | -----------: |
|             100 | **PI-FNO**                   | **7.484 ± 9.090** |   **0.466 ± 0.178** | **52.51 ± 51.78** |    23.044 ms |
|                 | Data-Driven FNO              |     7.779 ± 9.616 |       0.473 ± 0.171 |     81.41 ± 73.37 |    22.384 ms |
|                 | Discrete Transport Reference |         reference |           reference |     44.71 ± 47.38 | **2.358 ms** |
|             500 | **PI-FNO**                   | **7.050 ± 8.090** |   **0.472 ± 0.177** | **48.86 ± 48.06** |    22.697 ms |
|                 | Data-Driven FNO              |     7.346 ± 8.425 |       0.480 ± 0.172 |     76.61 ± 70.06 |    23.112 ms |
|                 | Discrete Transport Reference |         reference |           reference |     40.97 ± 43.39 | **2.345 ms** |
|            1000 | **PI-FNO**                   | **6.781 ± 7.523** |   **0.470 ± 0.178** | **47.30 ± 44.48** |    24.279 ms |
|                 | Data-Driven FNO              |     7.077 ± 7.876 |       0.478 ± 0.174 |     74.57 ± 65.93 |    23.297 ms |
|                 | Discrete Transport Reference |         reference |           reference |     39.46 ± 41.07 | **2.327 ms** |

Across all three evaluation sizes, PI-FNO obtains lower **mean prediction MSE, Relative $L_2$ error, and PDE residual** than the data-driven FNO.

On the 1,000-trajectory evaluation:

* prediction MSE decreases by approximately **4.2%**,
* PDE residual decreases by approximately **36.6%**.

The experiment supports the interpretation that the physics residual acts as a regularizer that improves physical consistency while preserving competitive prediction accuracy.

---

## 4. PDE Residual vs. Prediction Error

The repository contains a per-sample comparison of Relative $L_2$ prediction error and PDE residual for all 1,000 evaluation trajectories under both FNO configurations.

![MSE-PDE vs. L2 Error Scatter Plot](experiments/results/pde_vs_l2.png)

The PI-FNO distribution is concentrated in a lower PDE-residual regime than the data-driven FNO, while maintaining comparable prediction error.

This analysis complements the aggregate benchmark by showing the sample-level relationship between prediction accuracy and physical consistency.

---

## 5. Qualitative Trajectories

Representative trajectories are provided at 30-, 60-, and 120-minute horizons.

Each visualization compares:

* initial field,
* Discrete Transport Reference,
* PI-FNO prediction,
* absolute spatial prediction error.

### Trajectory Case 0

Wind:

$$U = -14.3\ \mathrm{km/h}, \qquad V = -5.8\ \mathrm{km/h}$$

![Trajectory Comparison Case 0](experiments/results/trajectory_comparison_0.png)

### Trajectory Case 4

Wind:

$$U = 4.7\ \mathrm{km/h}, \qquad V = -2.1\ \mathrm{km/h}$$

![Trajectory Comparison Case 4](experiments/results/trajectory_comparison_4.png)

### Trajectory Case 7

Wind:

$$U = -3.2\ \mathrm{km/h}, \qquad V = 12.4\ \mathrm{km/h}$$

![Trajectory Comparison Case 7](experiments/results/trajectory_comparison_7.png)

The qualitative examples show that the PI-FNO reproduces the dominant transport structure over the two-hour horizon, with errors concentrated around moving high-gradient regions.

---

# Inference Latency

Latency is measured per trajectory.

Under the current $32 \times 32$ CPU benchmark, the discrete transport reference is faster than neural inference:

* Discrete reference: approximately **2.3 ms**
* FNO inference: approximately **23–24 ms**

Therefore, AetherCast **does not claim a computational speed advantage at this small grid size**.

Potential scaling advantages of neural operators for larger spatial domains, longer horizons, and batched GPU workloads are left as future work.

---

# Live Weather Demonstration

## Tomorrow.io API

The live pipeline queries a localized Tomorrow.io observation for:

* temperature,
* humidity,
* wind speed,
* wind direction,
* rainfall intensity.

## Virtual spatial support

Because the live pipeline obtains a localized observation rather than a dense multi-station field, four synthetic neighboring support points are constructed using fixed perturbations:

* North × 0.7
* South × 1.2
* East × 0.8
* West × 1.1

These points are **not independent weather observations**.

They are used solely to create spatial support for the IDW visualization.

## IDW interpolation

The support observations are mapped to local kilometer coordinates and interpolated onto the $32 \times 32$ grid using inverse-distance weighting.

For a local coordinate displacement:

$$x = \Delta lon \times 111 \times \cos(lat),$$

$$y = \Delta lat \times 111.$$

## Neural projection

The resulting field, together with the derived wind components, is passed through the pretrained PI-FNO.

Neural projections are displayed in **illustrative model units** and should not be interpreted as calibrated rainfall-rate predictions.

---

# Repository Structure

```text
.
├── aethercast/
│   ├── config.py
│   ├── data.py
│   ├── physics.py
│   ├── solvers.py
│   ├── train.py
│   ├── models/
│   │   ├── fno2d.py
│   │   └── layers.py
│   └── utils/
│       └── physics_loss.py
│
├── experiments/
│   ├── run_training_sweep.py
│   ├── run_lambda_sweep.py
│   ├── run_model_comparison.py
│   ├── run_residual_vs_error.py
│   ├── run_qualitative_trajectories.py
│   └── results/
│
├── app.py
├── streamlit_app.py
├── templates/
│   └── index.html
├── fno_weights.pt
├── requirements.txt
├── Dockerfile
└── README.md
```

---

# Reproducibility

All benchmark experiments use `SEED = 42`.

The experiment scripts explicitly seed:

* Python `random`,
* NumPy,
* PyTorch,
* CUDA when available.

The repository contains the generated CSV tables and qualitative figures used in the README.

## Re-run experiments

```bash
python experiments/run_training_sweep.py
python experiments/run_lambda_sweep.py
python experiments/run_model_comparison.py
python experiments/run_residual_vs_error.py
python experiments/run_qualitative_trajectories.py
```

---

# Deployment

A pretrained checkpoint is included:

```text
fno_weights.pt
```

The deployment applications load the saved state dictionary directly rather than retraining the model at startup.

The repository includes a Docker configuration for deployment to Hugging Face Spaces.

---

# Limitations

The current prototype has several important limitations:

1. **Synthetic benchmark dynamics**
   Training and evaluation trajectories are generated from a controlled synthetic transport process rather than observational weather datasets.

2. **Uniform transport velocity**
   The benchmark assumes spatially uniform transport velocities over the entire $30 \times 30$ km domain.

3. **Periodic boundaries**
   The synthetic benchmark uses circular spatial boundaries.

4. **Coarse spatial resolution**
   The model operates on a $32 \times 32$ grid.

5. **Synthetic live spatial expansion**
   The live visualization constructs virtual spatial support from a single localized weather observation.

6. **No radar validation**
   The system has not been evaluated against radar-derived precipitation ground truth.

7. **Model-unit output**
   Neural predictions are not calibrated to physical rainfall-rate units.

8. **Small-scale latency benchmark**
   The current CPU benchmark does not demonstrate a neural inference speed advantage over the simple discrete reference.

These limitations define the scope of the current research prototype and motivate future evaluation on observational weather datasets and larger operator-learning problems.

---

# Future Work

Potential extensions include:

* evaluation against radar-derived precipitation fields,
* multi-station observational input,
* spatially varying wind fields,
* non-periodic boundary conditions,
* larger spatial domains and resolutions,
* longer prediction horizons,
* uncertainty estimation,
* GPU and batched inference benchmarks,
* and comparison against stronger numerical and neural forecasting baselines.

---

# Citation

```bibtex
@article{aethercast2026,
  title={AetherCast: Physics-Informed Fourier Neural Operator for 2D Transport Simulation},
  author={Shanmitha, Thirumoorthy},
  journal={arXiv preprint},
  year={2026}
}
```

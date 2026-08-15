# AetherCast Experiments Directory

This folder contains the training, benchmarking, and parameter sweep scripts to evaluate and compare the Physics-Informed Fourier Neural Operator (PI-FNO) against purely data-driven FNO and classical discrete transport baselines.

## Directory Structure

```text
experiments/
├── README.md
├── run_training_sweep.py         # Size sweep (N = 64 to 1024, lambda = 0.01)
├── run_lambda_sweep.py           # Regularization weight sweep (lambda = 0 to 0.1, size = 256)
├── run_model_comparison.py       # Comparative benchmark (PI-FNO, Data-Driven FNO, Reference)
├── run_residual_vs_error.py      # Experiment 4: MSE-PDE vs. Relative L2 scatter plot
├── run_qualitative_trajectories.py # Experiment 5: Qualitative advection-diffusion trajectories
└── results/
    ├── training_size.csv         # Output results for training size scaling
    ├── lambda_sweep.csv          # Output results for regularization sweeps
    ├── model_comparison.csv      # Output results for model comparisons (mean ± std)
    ├── residual_vs_error.csv     # Raw L2 vs. PDE scatter values
    ├── pde_vs_l2.png             # Experiment 4 scatter plot figure
    └── trajectory_comparison_*.png # Experiment 5 trajectory grid figures
```

## Reproducibility Instructions

All experiments use a fixed seed `SEED = 42` for data generation and model parameter initialization to guarantee exact reproducibility of reported results.

### 1. Run FNO Training Size Sweep
Runs FNO training over sizes of `[64, 128, 256, 512, 1024]` with $\lambda = 0.01$ and evaluates them on a fixed 1,000-sample test set:
```bash
python experiments/run_training_sweep.py
```

### 2. Run Physics Regularization Weight Sweep
Runs FNO training over physics weights $\lambda \in \{0.0, 0.001, 0.01, 0.05, 0.1\}$ with $N = 256$ and evaluates them on the same fixed 1,000-sample test set:
```bash
python experiments/run_lambda_sweep.py
```

### 3. Run Comparative Benchmarks
Trains a PI-FNO ($\lambda = 0.01$) and a purely data-driven FNO ($\lambda = 0.0$) on 256 samples, and runs them alongside the Discrete Transport Reference solver on held-out test sets of 100, 500, and 1,000 samples:
```bash
python experiments/run_model_comparison.py
```
This script computes mean and standard deviation (mean ± std) for all metrics, and runs warm-up runs with CUDA synchronization (`torch.cuda.synchronize()`) for accurate hardware latency profiling.

### 4. Run Scatter Analysis (Experiment 4)
Computes the prediction error vs. physical violation scatter distribution across all 1,000 test samples:
```bash
python experiments/run_residual_vs_error.py
```

### 5. Run Qualitative Trajectory Plotter (Experiment 5)
Saves comparison grids of Reference, PI-FNO, and spatial absolute error fields across 3 representative weather sequences:
```bash
python experiments/run_qualitative_trajectories.py
```

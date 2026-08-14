import random
import numpy as np
import torch
from aethercast.config import (
    GRID_RES, BOX_KM, MINUTES_PER_STEP, PREDICTION_STEPS,
    DECAY_PER_STEP, DIFFUSION_PER_STEP, WIND_MIN, WIND_MAX
)

def diffuse2d(field, diffusion_factor=0.015):
    """Applies a 2D discrete diffusion step to the spatial field with periodic boundaries."""
    if diffusion_factor <= 0:
        return field
    left = np.roll(field, -1, axis=1)
    right = np.roll(field, 1, axis=1)
    up = np.roll(field, -1, axis=0)
    down = np.roll(field, 1, axis=0)
    out = (1.0 - 4.0 * diffusion_factor) * field + diffusion_factor * (left + right + up + down)
    return out

def shift2d(field, d_row, d_col):
    """Applies a 2D shift to the spatial field with periodic (circular) boundary conditions."""
    out = np.roll(field, d_row, axis=0)
    out = np.roll(out, d_col, axis=1)
    return out

def generate_synthetic_data(num_samples=256):
    """
    Generates synthetic advection-diffusion weather trajectories.
    Each sample represents a 3D input state (rain field, wind U, wind V) and
    a 24-step advected, diffused, and decaying target sequence.
    """
    X = []
    Y = []
    x_grid, y_grid = np.meshgrid(np.arange(GRID_RES), np.arange(GRID_RES), indexing='ij')

    for _ in range(num_samples):
        # Create 1 to 3 random Gaussian rain blobs
        rain = np.zeros((GRID_RES, GRID_RES), dtype=np.float32)
        num_blobs = random.randint(1, 3)
        for _ in range(num_blobs):
            cx, cy = random.randint(4, GRID_RES-4), random.randint(4, GRID_RES-4)
            r = random.uniform(2.5, 5.5)
            intensity = random.uniform(6.0, 35.0)
            dist2 = (x_grid - cx)**2 + (y_grid - cy)**2
            rain += intensity * np.exp(-dist2 / (2 * r**2))

        # Random wind vector [U, V] in km/h
        u = random.uniform(WIND_MIN, WIND_MAX)
        v = random.uniform(WIND_MIN, WIND_MAX)

        u_field = np.full((GRID_RES, GRID_RES), u, dtype=np.float32)
        v_field = np.full((GRID_RES, GRID_RES), v, dtype=np.float32)

        x_sample = np.stack([rain, u_field, v_field], axis=-1)

        # Generate target steps (24 frames * 5 min = 120 mins)
        y_sample = np.zeros((PREDICTION_STEPS, GRID_RES, GRID_RES), dtype=np.float32)
        for step in range(PREDICTION_STEPS):
            t_hours = ((step + 1) * MINUTES_PER_STEP) / 60.0
            shift_lat_km = v * t_hours
            shift_lon_km = u * t_hours

            # Shift mapping
            shift_rows = int(round(shift_lat_km / (2 * BOX_KM) * GRID_RES))
            shift_cols = int(round(shift_lon_km / (2 * BOX_KM) * GRID_RES))

            shifted = shift2d(rain, shift_rows, shift_cols)
            # Apply time-dependent spatial diffusion
            diffused = diffuse2d(shifted, diffusion_factor=DIFFUSION_PER_STEP * (step + 1))
            decay = max(0.0, 1.0 - DECAY_PER_STEP * (step + 1))
            y_sample[step] = diffused * decay

        X.append(x_sample)
        Y.append(y_sample)

    return torch.tensor(np.array(X)), torch.tensor(np.array(Y))

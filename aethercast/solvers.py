import math
import numpy as np
from aethercast.config import PREDICTION_STEPS, DT, BOX_KM, GRID_RES, DECAY_RATE, DIFFUSION_PER_STEP
from aethercast.data import shift2d, diffuse2d

def run_discrete_transport_reference(rain_field, u, v):
    """
    Simulates advection-diffusion-decay dynamics over 24 temporal intervals.
    Serves as the explicit discrete transport reference solver.
    """
    grids = []
    current_field = rain_field.copy()
    
    # Shift in km per single step of size DT
    shift_lat_km = v * DT
    shift_lon_km = u * DT
    shift_rows = int(round(shift_lat_km / (2 * BOX_KM) * GRID_RES))
    shift_cols = int(round(shift_lon_km / (2 * BOX_KM) * GRID_RES))
    
    # Decay factor based on continuous decay rate: exp(-lambda_d * dt)
    decay_factor = math.exp(-DECAY_RATE * DT)

    for step in range(PREDICTION_STEPS):
        # 1. Advection (circular shift)
        shifted = shift2d(current_field, shift_rows, shift_cols)
        # 2. Diffusion (constant diffusion factor)
        diffused = diffuse2d(shifted, diffusion_factor=DIFFUSION_PER_STEP)
        # 3. True Exponential Decay
        current_field = diffused * decay_factor
        grids.append(current_field)
        
    return grids

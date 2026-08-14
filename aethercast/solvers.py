import numpy as np
from aethercast.config import PREDICTION_STEPS, MINUTES_PER_STEP, BOX_KM, GRID_RES, DECAY_PER_STEP, DIFFUSION_PER_STEP
from aethercast.data import shift2d, diffuse2d

def run_discrete_transport_reference(rain_field, u, v):
    """
    Simulates advection-diffusion-decay dynamics over 24 temporal intervals.
    Serves as the explicit discrete transport reference solver.
    """
    grids = []
    for step in range(PREDICTION_STEPS):
        t_hours = ((step + 1) * MINUTES_PER_STEP) / 60.0
        shift_lat_km = v * t_hours
        shift_lon_km = u * t_hours

        # Convert kilometer shift to grid indices
        shift_rows = int(round(shift_lat_km / (2 * BOX_KM) * GRID_RES))
        shift_cols = int(round(shift_lon_km / (2 * BOX_KM) * GRID_RES))

        # Perform periodic circular shift
        shifted = shift2d(rain_field, shift_rows, shift_cols)
        
        # Apply time-dependent spatial diffusion step
        diffused = diffuse2d(shifted, diffusion_factor=DIFFUSION_PER_STEP * (step + 1))
        
        # Apply linear decay step
        decay = max(0.0, 1.0 - DECAY_PER_STEP * (step + 1))
        grids.append(diffused * decay)
        
    return grids

import os

# Central Configuration for AetherCast Engine
GRID_RES = 32                 # Output grid resolution (32x32)
BOX_KM = 15                   # Half-width (km) of the sampling/nowcast box (30 km domain)
MINUTES_PER_STEP = 5          # 5 minutes per forecast step
PREDICTION_STEPS = 24         # 24 steps = 120 minutes (2 hour nowcast horizon)
CACHE_TTL_SECONDS = 300       # 5 minutes caching for Weather Union API responses

# Transport Dynamics Constants
DECAY_PER_STEP = 0.025        # Multiplicative decay factor per 5-minute step
DIFFUSION_PER_STEP = 0.015     # Spatial diffusion (dispersion) factor per step
WIND_MIN = -16.0              # Min wind velocity (km/h)
WIND_MAX = 16.0              # Max wind velocity (km/h)

# Loss and Physical Bounds
LAMBDA_PHY = 0.01             # Weight of the physics residual in PINN loss
DX = 30.0 / GRID_RES          # Grid column spacing: 30 km / 32 = 0.9375 km
DY = 30.0 / GRID_RES          # Grid row spacing: 30 km / 32 = 0.9375 km
DT = MINUTES_PER_STEP / 60.0  # Time step: 5 mins = 0.0833 hours

# Continuous PDE transport parameters
# D = diffusion_factor * dx^2 / dt
# Using diffusion_factor = 0.015, dx = 0.9375, dt = 0.0833 -> D ≈ 0.1582 km²/h
DIFFUSION_COEFF = DIFFUSION_PER_STEP * (DX ** 2) / DT
DECAY_RATE = DECAY_PER_STEP * 12.0  # lambda_d: continuous decay per hour = 0.025 * 12 = 0.3 h^-1

# API Ingestion
WU_BASE_URL = "https://www.weatherunion.com/gw/weather/external/v0/get_weather_data"
WEATHER_UNION_API_KEY = os.environ.get("WEATHER_UNION_API_KEY")

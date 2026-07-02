import os
import time
import math
import random
import requests
import numpy as np
import threading
from flask import Flask, jsonify, render_template, request

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# =====================================================================
# CONFIG & GLOBALS
# =====================================================================
GRID_RES = 32                 # output grid resolution (32x32)
BOX_KM = 15                   # half-width (km) of the sampling/forecast box
CACHE_TTL_SECONDS = 300       # 5 minutes caching for Weather Union API
MINUTES_PER_STEP = 5          # 24 steps * 5 min = 120 min (2 hour) horizon
WU_BASE_URL = "https://www.weatherunion.com/gw/weather/external/v0/get_weather_data"

# Default API key provided by the user
WEATHER_UNION_API_KEY = os.environ.get("WEATHER_UNION_API_KEY")

# Station Cache: (round(lat,2), round(lon,2), demo_mode) -> (timestamp, [stations], coverage_found, meteorology_dict)
_station_cache = {}

# FNO Training and Engine status
fno_status = {
    "status": "initializing",  # "initializing", "training", "ready", "failed"
    "epoch": 0,
    "total_epochs": 30,
    "loss_history": [],
    "device": "cpu",
    "param_count": 0,
    "training_time": 0.0,
    "model": None
}

# =====================================================================
# 1. FOURIER NEURAL OPERATOR (FNO-2D) IN PYTORCH
# =====================================================================
if TORCH_AVAILABLE:
    try:
        from aethercast.models.fno2d import FNO2d
        from aethercast.models.layers import SpectralConv2d
    except ImportError:
        from models.fno2d import FNO2d
        from models.layers import SpectralConv2d

# =====================================================================
# 2. VECTORIZED SYNTHETIC DATA GENERATION & TRAINING LOOP
# =====================================================================
def diffuse2d(field, diffusion_factor=0.03):
    """Applies a simple 2D physical diffusion (heat blur) to the spatial field."""
    if diffusion_factor <= 0:
        return field
    left = np.roll(field, -1, axis=1)
    right = np.roll(field, 1, axis=1)
    up = np.roll(field, -1, axis=0)
    down = np.roll(field, 1, axis=0)
    # 2D discrete diffusion equation solver step
    out = (1.0 - 4.0 * diffusion_factor) * field + diffusion_factor * (left + right + up + down)
    return out

def generate_synthetic_data(num_samples=256):
    """Vectorized, fast generation of synthetic advection-diffusion weather samples."""
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
        u = random.uniform(-16.0, 16.0)
        v = random.uniform(-16.0, 16.0)

        u_field = np.full((GRID_RES, GRID_RES), u, dtype=np.float32)
        v_field = np.full((GRID_RES, GRID_RES), v, dtype=np.float32)

        x_sample = np.stack([rain, u_field, v_field], axis=-1)

        # Generate target steps (24 frames * 5 min = 120 mins)
        y_sample = np.zeros((24, GRID_RES, GRID_RES), dtype=np.float32)
        for step in range(24):
            t_hours = ((step + 1) * MINUTES_PER_STEP) / 60.0
            shift_lat_km = v * t_hours
            shift_lon_km = u * t_hours

            # Shift mapping
            shift_rows = int(round(shift_lat_km / (2 * BOX_KM) * GRID_RES))
            shift_cols = int(round(shift_lon_km / (2 * BOX_KM) * GRID_RES))

            shifted = shift2d(rain, shift_rows, shift_cols)
            # Apply time-dependent spatial diffusion (blobs expand & dissolve)
            diffused = diffuse2d(shifted, diffusion_factor=0.015 * (step + 1))
            decay = max(0.0, 1.0 - 0.025 * (step + 1))
            y_sample[step] = diffused * decay

        X.append(x_sample)
        Y.append(y_sample)

    return torch.tensor(np.array(X)), torch.tensor(np.array(Y))

def calibrate_fno_model():
    """Background calibration thread to train the FNO on startup with high accuracy."""
    global fno_status
    if not TORCH_AVAILABLE:
        fno_status["status"] = "failed"
        print("[FNO Calibration ERROR]: PyTorch is not available.")
        return

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fno_status["device"] = "CUDA (GPU)" if device.type == "cuda" else "CPU (Fallback)"
        fno_status["status"] = "training"

        # Initialize Model
        model = FNO2d().to(device)
        fno_status["param_count"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Generate larger, high-fidelity dataset
        X, Y = generate_synthetic_data(256)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
        # Cosine Annealing scheduler to smoothly decay LR and improve generalization accuracy
        epochs = fno_status["total_epochs"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
        
        try:
            from aethercast.utils.physics_loss import PhysicsInformedLoss
        except ImportError:
            from utils.physics_loss import PhysicsInformedLoss

        mse_criterion = nn.MSELoss()
        pinn_criterion = PhysicsInformedLoss().to(device)
        lambda_phy = 0.01
        batch_size = 32

        t_start = time.time()
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            indices = torch.randperm(len(X))
            
            for i in range(0, len(X), batch_size):
                batch_idx = indices[i:i+batch_size]
                bx = X[batch_idx].to(device)
                by = Y[batch_idx].to(device)

                optimizer.zero_grad()
                out = model(bx)
                mse_loss = mse_criterion(out, by)
                pinn_loss = pinn_criterion(out, bx)
                loss = mse_loss + lambda_phy * pinn_loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * bx.size(0)

            scheduler.step()
            avg_loss = epoch_loss / len(X)
            fno_status["epoch"] = epoch + 1
            fno_status["loss_history"].append(avg_loss)
            
        fno_status["training_time"] = time.time() - t_start
        fno_status["model"] = model
        fno_status["status"] = "ready"
        print(f"[FNO Calibration] FNO successfully calibrated in {fno_status['training_time']:.2f}s on {device}. Final Loss: {fno_status['loss_history'][-1]:.6f}")

    except Exception as e:
        print(f"[FNO Calibration ERROR]: {e}")
        fno_status["status"] = "failed"

# =====================================================================
# 3. WEATHER UNION DATA LAYER & SMART SAMPLING (Rate-Limit Friendly)
# =====================================================================
def km_to_deg_lat(km):
    return km / 111.0

def km_to_deg_lon(km, at_lat):
    return km / (111.0 * math.cos(math.radians(at_lat)))

def fetch_station(lat, lon):
    """Hits Weather Union API for a single coordinate."""
    if not WEATHER_UNION_API_KEY:
        return None
    headers = {"x-zomato-api-key": WEATHER_UNION_API_KEY}
    params = {"latitude": lat, "longitude": lon}
    try:
        resp = requests.get(WU_BASE_URL, headers=headers, params=params, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "200" or data.get("status") == 200:
                return data.get("locality_weather_data", {})
    except Exception as e:
        print(f"[WU API Fetch Error] at ({lat}, {lon}): {e}")
    return None

def sample_stations(center_lat, center_lon, demo_mode=False):
    """
    Query only the center point to save API hits (1 call per search).
    Generates surrounding virtual points to build the 2D field.
    Includes 5-minute caching mechanism.
    """
    cache_key = (round(center_lat, 2), round(center_lon, 2), demo_mode)
    now = time.time()
    
    # Check cache
    cached = _station_cache.get(cache_key)
    if cached and (now - cached[0] < CACHE_TTL_SECONDS):
        # Return stations, coverage_found, meteorology_dict, time_remaining
        time_remaining = int(CACHE_TTL_SECONDS - (now - cached[0]))
        return cached[1], cached[2], cached[3], time_remaining, True

    stations = []
    coverage_found = False
    
    # Default meteorology dictionary
    met = {
        "temperature": 24.5,
        "humidity": 65.0,
        "wind_speed": 8.0,
        "wind_direction": 120.0,
        "rain_intensity": 0.0,
        "api_pulled": False
    }

    if demo_mode:
        # Generate dynamic demo rain field parameters centered at user target
        met["rain_intensity"] = 18.5
        met["wind_speed"] = 14.5
        met["wind_direction"] = 135.0  # blowing southeast
        met["temperature"] = 22.8
        met["humidity"] = 88.0
        coverage_found = True
    else:
        # Fetch actual Weather Union API (Single Pull)
        data = fetch_station(center_lat, center_lon)
        met["api_pulled"] = True
        
        if data:
            coverage_found = True
            # Fill met values, replacing nulls with reasonable defaults
            met["temperature"] = data.get("temperature") if data.get("temperature") is not None else 25.0
            met["humidity"] = data.get("humidity") if data.get("humidity") is not None else 70.0
            met["wind_speed"] = data.get("wind_speed") if data.get("wind_speed") is not None else 5.0
            met["wind_direction"] = data.get("wind_direction") if data.get("wind_direction") is not None else 90.0
            met["rain_intensity"] = data.get("rain_intensity") if data.get("rain_intensity") is not None else 0.0
        else:
            # Out of coverage area
            coverage_found = False
            # Flat defaults
            met["rain_intensity"] = 0.0
            met["wind_speed"] = 0.0
            met["wind_direction"] = 0.0

    # Build spatial grid based on meteorology center values using Virtual Expansion
    # Center station
    stations.append({
        "lat": center_lat, "lon": center_lon,
        "rain_intensity": met["rain_intensity"],
        "wind_speed": met["wind_speed"],
        "wind_direction": met["wind_direction"]
    })

    # Add 4 virtual stations to populate a 2D field for spatial interpolation
    dlat = km_to_deg_lat(BOX_KM / 1.5)
    dlon = km_to_deg_lon(BOX_KM / 1.5, center_lat)
    
    offsets = [
        (dlat, 0, 0.7),    # North (slightly less rain)
        (-dlat, 0, 1.2),   # South (slightly more rain)
        (0, dlon, 0.8),    # East
        (0, -dlon, 1.1)    # West
    ]

    for lat_off, lon_off, rain_mult in offsets:
        # Perturb rain intensity slightly to make the spatial field interesting
        p_rain = met["rain_intensity"] * rain_mult if met["rain_intensity"] > 0 else 0.0
        stations.append({
            "lat": center_lat + lat_off,
            "lon": center_lon + lon_off,
            "rain_intensity": p_rain,
            "wind_speed": met["wind_speed"],
            "wind_direction": met["wind_direction"]
        })

    # Cache response
    _station_cache[cache_key] = (now, stations, coverage_found, met)
    return stations, coverage_found, met, CACHE_TTL_SECONDS, False

# =====================================================================
# 4. SPATIAL INTERPOLATION & PHYSICAL ADVECTION
# =====================================================================
def idw_interpolate(grid_lats, grid_lons, stations, power=2):
    field = np.zeros((GRID_RES, GRID_RES))
    s_lat = np.array([s["lat"] for s in stations])
    s_lon = np.array([s["lon"] for s in stations])
    s_val = np.array([s["rain_intensity"] for s in stations])

    for i in range(GRID_RES):
        for j in range(GRID_RES):
            d = np.hypot(grid_lats[i] - s_lat, grid_lons[j] - s_lon) + 1e-6
            w = 1.0 / (d ** power)
            field[i, j] = np.sum(w * s_val) / np.sum(w)
    return field

def mean_wind_vector(stations):
    u_sum, v_sum = 0.0, 0.0
    for s in stations:
        speed = s["wind_speed"]
        rad = math.radians(s["wind_direction"])
        # Wind blowing FROM direction -> vector points in opposite direction
        u_sum += -speed * math.sin(rad)
        v_sum += -speed * math.cos(rad)
    n = len(stations)
    return u_sum / n, v_sum / n

def shift2d(field, d_row, d_col):
    out = np.zeros_like(field)
    rows, cols = field.shape
    sr0, sr1 = max(0, -d_row), min(rows, rows - d_row)
    dr0, dr1 = max(0, d_row), min(rows, rows + d_row)
    sc0, sc1 = max(0, -d_col), min(cols, cols - d_col)
    dc0, dc1 = max(0, d_col), min(cols, cols + d_col)
    if sr1 > sr0 and sc1 > sc0:
        out[dr0:dr1, dc0:dc1] = field[sr0:sr1, sc0:sc1]
    return out

def compute_advection_grids(rain_field, u, v):
    """Simulates physical advection-diffusion steps over the 24 intervals."""
    advection_grids = []
    for step in range(24):
        t_hours = ((step + 1) * MINUTES_PER_STEP) / 60.0
        shift_lat_km = v * t_hours
        shift_lon_km = u * t_hours

        # Convert shift to grid indices
        shift_rows = int(round(shift_lat_km / (2 * BOX_KM) * GRID_RES))
        shift_cols = int(round(shift_lon_km / (2 * BOX_KM) * GRID_RES))

        shifted = shift2d(rain_field, shift_rows, shift_cols)
        # Apply physical diffusion
        diffused = diffuse2d(shifted, diffusion_factor=0.015 * (step + 1))
        decay = max(0.0, 1.0 - 0.03 * (step + 1))
        advection_grids.append(np.round(diffused * decay, 1).tolist())
    return advection_grids

# =====================================================================
# 5. FLASK WEB APP & API ROUTES
# =====================================================================
app = Flask(__name__)

from jinja2 import ChoiceLoader, FileSystemLoader
app.jinja_loader = ChoiceLoader([
    FileSystemLoader('templates'),
    FileSystemLoader('.')
])

import traceback
@app.errorhandler(Exception)
def handle_exception(e):
    # Pass through HTTP errors
    tb = traceback.format_exc()
    return f"<h1>Internal Server Error (500)</h1><p>An unhandled exception occurred:</p><pre>{tb}</pre>", 500

@app.route('/api/engine_status')
def engine_status():
    """Returns FNO calibration state."""
    status_copy = {k: v for k, v in fno_status.items() if k != "model"}
    return jsonify(status_copy)

@app.route('/api/predict_target')
def predict_target():
    """Performs Weather Union search and outputs comparative forecast."""
    try:
        lat = float(request.args.get('lat', 12.9716))  # Default: Bangalore
        lon = float(request.args.get('lon', 77.5946))
        demo_mode = request.args.get('demo', 'false').lower() == 'true'
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "Invalid coordinates"}), 400

    # Fetch/sample weather stations (incorporating Cache + Single Pull + Virtual Expansion)
    stations, coverage_found, met, cache_ttl, was_cached = sample_stations(lat, lon, demo_mode)

    # Compute Interpolated Initial State
    dlat = km_to_deg_lat(BOX_KM)
    dlon = km_to_deg_lon(BOX_KM, lat)
    grid_lats = np.linspace(lat - dlat, lat + dlat, GRID_RES)
    grid_lons = np.linspace(lon - dlon, lon + dlon, GRID_RES)

    rain_field = idw_interpolate(grid_lats, grid_lons, stations)
    u, v = mean_wind_vector(stations)

    # 1. Classical Physics Advection Forecast
    t0_adv = time.time()
    grids_advection = compute_advection_grids(rain_field, u, v)
    t_adv_ms = (time.time() - t0_adv) * 1000

    # 2. FNO Neural Operator Forecast
    grids_fno = []
    fno_active = False
    fno_time_ms = 0.0

    if fno_status["status"] == "ready" and fno_status["model"] is not None:
        try:
            t0_fno = time.time()
            model = fno_status["model"]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Form input: Shape (GRID_RES, GRID_RES, 3) -> rain, u_field, v_field
            u_field = np.full((GRID_RES, GRID_RES), u, dtype=np.float32)
            v_field = np.full((GRID_RES, GRID_RES), v, dtype=np.float32)
            input_np = np.stack([rain_field, u_field, v_field], axis=-1)
            
            # Prepare tensor (batch, height, width, channels) -> shape (1, 32, 32, 3)
            in_tensor = torch.tensor(input_np, dtype=torch.float32, device=device).unsqueeze(0)
            
            model.eval()
            with torch.inference_mode():
                fno_out = model(in_tensor).squeeze(0).cpu().numpy()  # shape (24, 32, 32)
                
            # Post-process: Clamp negatives to 0
            fno_out = np.maximum(0.0, fno_out)
            grids_fno = np.round(fno_out, 1).tolist()
            fno_time_ms = (time.time() - t0_fno) * 1000
            fno_active = True
        except Exception as e:
            print(f"[FNO Inference Error]: {e}")
            grids_fno = grids_advection
    else:
        # Fallback to advection if neural model is still training
        grids_fno = grids_advection

    return jsonify({
        "status": "success",
        "coverage_found": coverage_found,
        "meteorology": met,
        "cache": {
            "was_cached": was_cached,
            "expires_in_sec": cache_ttl
        },
        "benchmarks": {
            "fno_active": fno_active,
            "fno_device": fno_status["device"],
            "fno_time_ms": fno_time_ms,
            "advection_time_ms": t_adv_ms,
            "param_count": fno_status["param_count"]
        },
        "grids_fno": grids_fno,
        "grids_advection": grids_advection
    })

# =====================================================================
# 6. PREMIUM DASHBOARD INTERFACE (AETHERCAST)
# =====================================================================
@app.route('/')
def home():
    return render_template('index.html')

# =====================================================================
# SERVER STARTUP
# =====================================================================
if __name__ == '__main__':
    if TORCH_AVAILABLE:
        # Start PyTorch FNO model calibration in a daemon background thread
        calibration_thread = threading.Thread(target=calibrate_fno_model, daemon=True)
        calibration_thread.start()
        print("[STARTUP]: Spawning background FNO calibration thread.")
    else:
        fno_status["status"] = "failed"
        print("[WARNING]: PyTorch is not installed. FNO engine will remain inactive, falling back to advection.")

    port = int(os.environ.get("PORT", 5000))
    print(f"[STARTUP]: Mounting AetherCast Portal at http://0.0.0.0:{port}/")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
import os
import time
import requests
import numpy as np
from flask import Flask, jsonify, render_template, request

# Import unified configuration, physics, and model structures from aethercast
from aethercast.config import (
    GRID_RES, BOX_KM, CACHE_TTL_SECONDS, MINUTES_PER_STEP
)
from aethercast.physics import (
    sample_stations, idw_interpolate, mean_wind_vector, km_to_deg_lat, km_to_deg_lon
)
from aethercast.solvers import run_discrete_transport_reference

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from aethercast.models.fno2d import FNO2d
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# Engine status dict to return diagnostics via UI
fno_status = {
    "status": "initializing",  # "initializing", "ready", "failed"
    "device": "cpu",
    "param_count": 0,
    "model": None
}

# =====================================================================
# LOAD PRE-TRAINED MODEL WEIGHTS
# =====================================================================
def load_neural_operator():
    global fno_status
    if not TORCH_AVAILABLE:
        fno_status["status"] = "failed"
        print("[STARTUP WARNING]: PyTorch not available. FNO neural model is disabled.")
        return

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fno_status["device"] = "CUDA (GPU)" if device.type == "cuda" else "CPU (Fallback)"
        
        model = FNO2d().to(device)
        fno_status["param_count"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        weights_path = "fno_weights.pt"
        if os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            model.eval()
            fno_status["model"] = model
            fno_status["status"] = "ready"
            print(f"[STARTUP]: Successfully loaded pre-trained PI-FNO weights from {weights_path} ({fno_status['param_count']:,} parameters).")
        else:
            fno_status["status"] = "failed"
            print(f"[STARTUP ERROR]: Pre-trained weights file {weights_path} not found.")
    except Exception as e:
        fno_status["status"] = "failed"
        print(f"[STARTUP ERROR]: Failed to load weights: {e}")

# =====================================================================
# FLASK WEB APP & API ROUTES
# =====================================================================
app = Flask(__name__)

# Loader configuration for template directory
from jinja2 import ChoiceLoader, FileSystemLoader
app.jinja_loader = ChoiceLoader([
    FileSystemLoader('templates'),
    FileSystemLoader('.')
])

import traceback
@app.errorhandler(Exception)
def handle_exception(e):
    tb = traceback.format_exc()
    return f"<h1>Internal Server Error (500)</h1><p>An unhandled exception occurred:</p><pre>{tb}</pre>", 500

@app.route('/api/engine_status')
def engine_status():
    """Returns FNO model registration details and status."""
    status_copy = {k: v for k, v in fno_status.items() if k != "model"}
    return jsonify(status_copy)

@app.route('/api/predict_target')
def predict_target():
    """Performs weather interpolation and returns model projections."""
    try:
        lat = float(request.args.get('lat', 12.9716))  # Default: Bangalore
        lon = float(request.args.get('lon', 77.5946))
        demo_mode = request.args.get('demo', 'false').lower() == 'true'
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "Invalid coordinates"}), 400

    # Fetch and sample localized Tomorrow.io observations
    stations, coverage_found, met, cache_ttl, was_cached = sample_stations(lat, lon, demo_mode)

    # Compute interpolated rainfall intensity grid in km coordinates
    rain_field = idw_interpolate(lat, lon, stations)
    u, v = mean_wind_vector(stations)

    # 1. Classical Discrete Transport Projection Reference
    t0_adv = time.time()
    grids_advection = run_discrete_transport_reference(rain_field, u, v)
    grids_advection_list = [np.round(g, 1).tolist() for g in grids_advection]
    t_adv_ms = (time.time() - t0_adv) * 1000

    # 2. FNO Neural Operator Projection
    grids_fno = []
    fno_active = False
    fno_time_ms = 0.0

    if fno_status["status"] == "ready" and fno_status["model"] is not None:
        try:
            t0_fno = time.time()
            model = fno_status["model"]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Form input: Shape (GRID_RES, GRID_RES, 3) -> rain, wind U, wind V
            u_field = np.full((GRID_RES, GRID_RES), u, dtype=np.float32)
            v_field = np.full((GRID_RES, GRID_RES), v, dtype=np.float32)
            input_np = np.stack([rain_field, u_field, v_field], axis=-1)
            
            # Prepare tensor (batch, height, width, channels) -> shape (1, 32, 32, 3)
            in_tensor = torch.tensor(input_np, dtype=torch.float32, device=device).unsqueeze(0)
            
            model.eval()
            with torch.inference_mode():
                # Perform synchronization for accurate CUDA timing measurements
                if device.type == "cuda":
                    torch.cuda.synchronize()
                
                t_inf_start = time.perf_counter()
                fno_out = model(in_tensor).squeeze(0).cpu().numpy()  # shape (24, 32, 32)
                
                if device.type == "cuda":
                    torch.cuda.synchronize()
                fno_time_ms = (time.perf_counter() - t_inf_start) * 1000
                
            # Clamp negatives
            fno_out = np.maximum(0.0, fno_out)
            grids_fno = np.round(fno_out, 1).tolist()
            fno_active = True
        except Exception as e:
            print(f"[FNO Inference Error]: {e}")
            grids_fno = grids_advection_list
    else:
        # Fallback to advection if model not loaded
        grids_fno = grids_advection_list

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
        "grids_advection": grids_advection_list
    })

@app.route('/')
def home():
    return render_template('index.html')

# =====================================================================
# SERVER STARTUP
# =====================================================================
# Load model weights synchronously on startup
load_neural_operator()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"[STARTUP]: Mounting AetherCast Portal at http://0.0.0.0:{port}/")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
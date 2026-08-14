import os
import time
import math
import requests
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Import unified configuration, physics, and model structures from aethercast
from aethercast.config import (
    GRID_RES, BOX_KM, MINUTES_PER_STEP, PREDICTION_STEPS, WU_BASE_URL, WEATHER_UNION_API_KEY
)
from aethercast.physics import (
    sample_stations, idw_interpolate, mean_wind_vector, geocode_city if 'geocode_city' in globals() else None
)
from aethercast.solvers import run_discrete_transport_reference

# Set page config
st.set_page_config(
    page_title="AetherCast // FNO Nowcasting",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import PyTorch safely
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from aethercast.models.fno2d import FNO2d
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Helper for geolocation (using Nominatim openstreetmap API)
def geocode_location(city_name):
    try:
        url = f"https://nominatim.openstreetmap.org/search?format=json&q={requests.utils.quote(city_name)}&limit=1"
        res = requests.get(url, headers={'User-Agent': 'AetherCastStreamlit/2.0'}, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if data:
                return float(data[0]["lat"]), float(data[0]["lon"]), data[0]["display_name"].split(',')[0]
    except Exception:
        pass
    return None

# Load pre-trained model weights from disk
@st.cache_resource
def load_fno_model():
    if not TORCH_AVAILABLE:
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FNO2d().to(device)
    weights_path = "fno_weights.pt"
    if os.path.exists(weights_path):
        try:
            state = torch.load(weights_path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            model.eval()
            return model
        except Exception as e:
            st.sidebar.error(f"Error loading {weights_path}: {e}")
    else:
        st.sidebar.warning(f"Weights file {weights_path} not found. Running advection fallback only.")
    return None

# =====================================================================
# CUSTOM COLOR MAPPING
# =====================================================================
WEATHER_COLORS = ["#0f172a", "#068cc8", "#06c8c0", "#1cc806", "#c8b706", "#ef4444"]
bounds = [0.0, 0.05, 2.0, 10.0, 25.0, 35.0, 100.0]
cmap = mcolors.ListedColormap(WEATHER_COLORS)
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# =====================================================================
# STREAMLIT UI
# =====================================================================
st.title("⛈️ AETHERCAST // Neural Weather Nowcasting Portal")
st.write("2D Fourier Neural Operator weather projections powered by Weather Union API.")

# Load FNO Model
model = load_fno_model()

# Sidebar inputs
st.sidebar.header("🎯 Target Parameters")
search_query = st.sidebar.text_input("Enter City/Locality Name", placeholder="e.g. Pune, Indiranagar")

st.sidebar.write("Or select a coverage preset:")
presets = {
    "Bangalore": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Pune": (18.5204, 73.8567),
    "Mumbai": (19.0760, 72.8777)
}
preset_selection = st.sidebar.selectbox("Presets list", ["None"] + list(presets.keys()))

demo_mode = st.sidebar.checkbox("Demo Rain Mode", value=True, help="Simulate a rain field if selected locality is out of coverage.")

# Setup Session State for location
if "lat" not in st.session_state:
    st.session_state.lat = None
    st.session_state.lon = None
    st.session_state.city = ""

# Trigger Search / Presets
if search_query:
    res = geocode_location(search_query)
    if res:
        st.session_state.lat, st.session_state.lon, st.session_state.city = res
elif preset_selection != "None":
    lat, lon = presets[preset_selection]
    st.session_state.lat, st.session_state.lon, st.session_state.city = lat, lon, preset_selection

# Check if location loaded
if st.session_state.lat is None:
    st.info("👋 Welcome! Please search for a city/locality in the sidebar or select a preset to initialize the nowcasting pipeline.")
else:
    # Fetch API weather stats
    with st.spinner("Fetching Locality Weather Data..."):
        stations, coverage_found, met, cache_ttl, was_cached = sample_stations(st.session_state.lat, st.session_state.lon, demo_mode)
        if coverage_found and not demo_mode:
            st.sidebar.success("Weather Union API: Station data loaded successfully.")
        elif demo_mode:
            st.sidebar.success("Demo Mode: Synthesizing rainfall cells.")
        else:
            st.sidebar.warning("Locality outside Weather Union coverage. Showing flat forecast.")

    # Compute Wind Vector U and V
    u, v = mean_wind_vector(stations)

    # Spatial fields interpolation
    rain_field = idw_interpolate(st.session_state.lat, st.session_state.lon, stations)

    # Run Forecast Models
    grids_adv = run_discrete_transport_reference(rain_field, u, v)
    
    grids_fno = None
    fno_time_ms = 0.0
    if model is not None and TORCH_AVAILABLE:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            u_field = np.full((GRID_RES, GRID_RES), u, dtype=np.float32)
            v_field = np.full((GRID_RES, GRID_RES), v, dtype=np.float32)
            input_np = np.stack([rain_field, u_field, v_field], axis=-1)
            in_tensor = torch.tensor(input_np, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Run inference timing with CUDA synchronization if GPU is active
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_start = time.perf_counter()
            with torch.inference_mode():
                fno_out = model(in_tensor).squeeze(0).cpu().numpy()
            if device.type == "cuda":
                torch.cuda.synchronize()
            fno_time_ms = (time.perf_counter() - t_start) * 1000
            
            grids_fno = np.maximum(0.0, fno_out)
        except Exception as e:
            st.sidebar.error(f"Inference error: {e}")
            
    if grids_fno is None:
         grids_fno = grids_adv

    # Main dashboard grid layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📍 Active Location")
        st.write(f"**City/Area**: {st.session_state.city}")
        st.write(f"**Coordinates**: {st.session_state.lat:.4f}° N, {st.session_state.lon:.4f}° E")
        
        st.subheader("🌡️ Meteorological Stats")
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Temperature", f"{met['temperature']:.1f} °C")
        m_col2.metric("Humidity", f"{met['humidity']:.1f} %")
        
        m_col3, m_col4 = st.columns(2)
        m_col3.metric("Rain Rate", f"{met['rain_intensity']:.2f} mm/h")
        m_col4.metric("Wind Speed", f"{met['wind_speed']:.1f} km/h")
        
        # Wind Direction textual description
        deg = met["wind_direction"]
        dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        dir_name = dirs[int(round(((deg % 360) / 45.0))) % 8]
        st.write(f"**Wind Angle**: {deg}° ({dir_name})")

    with col2:
        st.subheader("⛈️ 2-Hour Rain Nowcast Timeline")
        
        # Step slider scrubber
        step = st.slider("Forecast Timeline Horizon", min_value=1, max_value=24, value=1, format="+%d steps", help="Each step represents 5 minutes.")
        minutes = step * 5
        st.write(f"**Horizon Target**: +{minutes} Minutes")
        
        view_toggle = st.radio("Select Model View Mode", ["FNO Neural Projection", "Classical Physics Solver"], horizontal=True)
        active_grids = grids_fno if view_toggle == "FNO Neural Projection" else grids_adv
        frame = active_grids[step - 1]
        
        # Render Heatmap Canvas
        fig, ax = plt.subplots(figsize=(4.5, 4.5), facecolor='none')
        ax.imshow(frame, cmap=cmap, norm=norm, origin='lower')
        ax.axis('off')
        st.pyplot(fig, clear_figure=True)
        
        # Dynamic Text Summary
        max_rain = np.max(frame)
        global_max = np.max(active_grids)
        global_peak_step = np.argmax([np.max(g) for g in active_grids])
        
        st.write("---")
        st.subheader("📝 Forecast Outlook Summary")
        
        if global_max <= 0.05:
            if met["temperature"] >= 35:
                st.info("🌡️ **2-Hour Outlook**: Extremely hot and dry. No precipitation expected.")
            else:
                st.info("☀️ **2-Hour Outlook**: Dry conditions with clear weather.")
        else:
            peak_m = (global_peak_step + 1) * 5
            rain_desc = "light drizzle" if global_max < 2 else "moderate rain showers" if global_max < 10 else "heavy precipitation"
            st.info(f"⛈️ **2-Hour Outlook**: Precipitation peaks in {peak_m} minutes with {rain_desc} (max {global_max:.1f} mm/h).")

        # Step specific text
        if max_rain <= 0.05:
            st.write(f"**At +{minutes} Mins**: Clear with no rainfall.")
        elif max_rain < 2:
            st.write(f"**At +{minutes} Mins**: Light drizzle falling (under 2.0 mm/h).")
        elif max_rain < 10:
            st.write(f"**At +{minutes} Mins**: Moderate rainfall predicted.")
        else:
            st.error(f"⚠️ **At +{minutes} Mins**: Heavy downpours expected! Commuters proceed with caution.")

    # Sidelined Tech diagnostics
    st.write("---")
    with st.expander("🛠️ Neural Engine Diagnostics & Model Comparisons"):
        st.subheader("Speed Benchmarks")
        dev = "GPU (CUDA Target)" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "CPU (Fallback)"
        st.write(f"**Inference Device**: {dev}")
        if model is not None:
            st.write(f"**Single-Trajectory Forward Latency**: {fno_time_ms:.3f} ms")
        
        # Compare volume dynamics chart
        st.subheader("Regional Rain Volume Dynamics Curve")
        fno_sums = [np.mean(g) for g in grids_fno]
        adv_sums = [np.mean(g) for g in grids_adv]
        chart_data = {
            "Timeline": [f"+{(i+1)*5}m" for i in range(24)],
            "FNO Neural": fno_sums,
            "Physics Solver": adv_sums
        }
        st.line_chart(chart_data, x="Timeline", y=["FNO Neural", "Physics Solver"])

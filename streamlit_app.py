import os
import time
import math
import random
import requests
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# =====================================================================
# CONFIG & GLOBALS
# =====================================================================
GRID_RES = 32
BOX_KM = 15
MINUTES_PER_STEP = 5
WU_BASE_URL = "https://www.weatherunion.com/gw/weather/external/v0/get_weather_data"
WEATHER_UNION_API_KEY = os.environ.get("WEATHER_UNION_API_KEY")

# =====================================================================
# FNO MODEL
# =====================================================================
if TORCH_AVAILABLE:
    try:
        from aethercast.models.fno2d import FNO2d
        from aethercast.models.layers import SpectralConv2d
    except ImportError:
        from models.fno2d import FNO2d
        from models.layers import SpectralConv2d

# =====================================================================
# PHYSICAL ADVECTION-DIFFUSION UTILITIES
# =====================================================================
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

def diffuse2d(field, diffusion_factor=0.03):
    if diffusion_factor <= 0:
        return field
    left = np.roll(field, -1, axis=1)
    right = np.roll(field, 1, axis=1)
    up = np.roll(field, -1, axis=0)
    down = np.roll(field, 1, axis=0)
    out = (1.0 - 4.0 * diffusion_factor) * field + diffusion_factor * (left + right + up + down)
    return out

def generate_synthetic_data(num_samples=128):
    X, Y = [], []
    x_grid, y_grid = np.meshgrid(np.arange(GRID_RES), np.arange(GRID_RES), indexing='ij')
    for _ in range(num_samples):
        rain = np.zeros((GRID_RES, GRID_RES), dtype=np.float32)
        num_blobs = random.randint(1, 3)
        for _ in range(num_blobs):
            cx, cy = random.randint(4, GRID_RES-4), random.randint(4, GRID_RES-4)
            r = random.uniform(2.5, 5.5)
            intensity = random.uniform(6.0, 35.0)
            dist2 = (x_grid - cx)**2 + (y_grid - cy)**2
            rain += intensity * np.exp(-dist2 / (2 * r**2))
        u = random.uniform(-16.0, 16.0)
        v = random.uniform(-16.0, 16.0)
        u_field = np.full((GRID_RES, GRID_RES), u, dtype=np.float32)
        v_field = np.full((GRID_RES, GRID_RES), v, dtype=np.float32)
        x_sample = np.stack([rain, u_field, v_field], axis=-1)
        
        y_sample = np.zeros((24, GRID_RES, GRID_RES), dtype=np.float32)
        for step in range(24):
            t_hours = ((step + 1) * MINUTES_PER_STEP) / 60.0
            shift_lat_km = v * t_hours
            shift_lon_km = u * t_hours
            shift_rows = int(round(shift_lat_km / (2 * BOX_KM) * GRID_RES))
            shift_cols = int(round(shift_lon_km / (2 * BOX_KM) * GRID_RES))
            shifted = shift2d(rain, shift_rows, shift_cols)
            diffused = diffuse2d(shifted, diffusion_factor=0.015 * (step + 1))
            decay = max(0.0, 1.0 - 0.025 * (step + 1))
            y_sample[step] = diffused * decay
        X.append(x_sample)
        Y.append(y_sample)
    return torch.tensor(np.array(X)), torch.tensor(np.array(Y))

# Cache training calibration
@st.cache_resource
def train_and_cache_fno():
    if not TORCH_AVAILABLE:
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FNO2d().to(device)
    X, Y = generate_synthetic_data(128)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-4)
    criterion = nn.MSELoss()
    
    # Progress monitor in streamlit
    progress_bar = st.progress(0, text="Calibrating Neural Operator Model...")
    for epoch in range(30):
        model.train()
        epoch_loss = 0
        indices = torch.randperm(len(X))
        for i in range(0, len(X), 32):
            batch_idx = indices[i:i+32]
            bx = X[batch_idx].to(device)
            by = Y[batch_idx].to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * bx.size(0)
        scheduler.step()
        progress_bar.progress((epoch+1)/30, text=f"Calibrating Neural Operator Model: Epoch {epoch+1}/30")
    
    progress_bar.empty()
    return model

# =====================================================================
# GEOLOCATION & STATIONS DATA
# =====================================================================
def geocode_city(city_name):
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

def fetch_station_data(lat, lon):
    if not WEATHER_UNION_API_KEY:
        return None
    headers = {"x-zomato-api-key": WEATHER_UNION_API_KEY}
    params = {"latitude": lat, "longitude": lon}
    try:
        resp = requests.get(WU_BASE_URL, headers=headers, params=params, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") in ["200", 200]:
                return data.get("locality_weather_data", {})
    except Exception:
        pass
    return None

def build_weather_field(center_lat, center_lon, met, demo_mode=False):
    stations = []
    stations.append({
        "lat": center_lat, "lon": center_lon,
        "rain_intensity": met["rain_intensity"],
        "wind_speed": met["wind_speed"],
        "wind_direction": met["wind_direction"]
    })
    dlat = BOX_KM / 166.5
    dlon = BOX_KM / (166.5 * math.cos(math.radians(center_lat)))
    offsets = [(dlat, 0, 0.7), (-dlat, 0, 1.2), (0, dlon, 0.8), (0, -dlon, 1.1)]
    for lat_off, lon_off, rain_mult in offsets:
        p_rain = met["rain_intensity"] * rain_mult if met["rain_intensity"] > 0 else 0.0
        stations.append({
            "lat": center_lat + lat_off, "lon": center_lon + lon_off,
            "rain_intensity": p_rain, "wind_speed": met["wind_speed"],
            "wind_direction": met["wind_direction"]
        })
    return stations

def idw_interpolate(center_lat, center_lon, stations):
    dlat = BOX_KM / 111.0
    dlon = BOX_KM / (111.0 * math.cos(math.radians(center_lat)))
    grid_lats = np.linspace(center_lat - dlat, center_lat + dlat, GRID_RES)
    grid_lons = np.linspace(center_lon - dlon, center_lon + dlon, GRID_RES)
    field = np.zeros((GRID_RES, GRID_RES))
    s_lat = np.array([s["lat"] for s in stations])
    s_lon = np.array([s["lon"] for s in stations])
    s_val = np.array([s["rain_intensity"] for s in stations])
    for i in range(GRID_RES):
        for j in range(GRID_RES):
            d = np.hypot(grid_lats[i] - s_lat, grid_lons[j] - s_lon) + 1e-6
            w = 1.0 / (d ** 2)
            field[i, j] = np.sum(w * s_val) / np.sum(w)
    return field

def run_classical_advection(rain_field, u, v):
    grids = []
    for step in range(24):
        t_hours = ((step + 1) * MINUTES_PER_STEP) / 60.0
        shift_lat_km = v * t_hours
        shift_lon_km = u * t_hours
        shift_rows = int(round(shift_lat_km / (2 * BOX_KM) * GRID_RES))
        shift_cols = int(round(shift_lon_km / (2 * BOX_KM) * GRID_RES))
        shifted = shift2d(rain_field, shift_rows, shift_cols)
        diffused = diffuse2d(shifted, diffusion_factor=0.015 * (step + 1))
        decay = max(0.0, 1.0 - 0.03 * (step + 1))
        grids.append(diffused * decay)
    return grids

def run_fno_inference(model, rain_field, u, v):
    if model is None or not TORCH_AVAILABLE:
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    u_field = np.full((GRID_RES, GRID_RES), u, dtype=np.float32)
    v_field = np.full((GRID_RES, GRID_RES), v, dtype=np.float32)
    input_np = np.stack([rain_field, u_field, v_field], axis=-1)
    in_tensor = torch.tensor(input_np, dtype=torch.float32, device=device).unsqueeze(0)
    model.eval()
    with torch.inference_mode():
        fno_out = model(in_tensor).squeeze(0).cpu().numpy()
    return np.maximum(0.0, fno_out)

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

# Load FNO
model = train_and_cache_fno()

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
    res = geocode_city(search_query)
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
        met = {
            "temperature": 24.5, "humidity": 65.0, "wind_speed": 8.0, "wind_direction": 120.0, "rain_intensity": 0.0
        }
        
        if demo_mode:
            met["rain_intensity"] = 18.5
            met["wind_speed"] = 14.5
            met["wind_direction"] = 135.0
            met["temperature"] = 22.8
            met["humidity"] = 88.0
            st.sidebar.success("Demo Mode: Synthesizing rainfall cells.")
        else:
            wu_data = fetch_station_data(st.session_state.lat, st.session_state.lon)
            if wu_data:
                met["temperature"] = wu_data.get("temperature") or 25.0
                met["humidity"] = wu_data.get("humidity") or 70.0
                met["wind_speed"] = wu_data.get("wind_speed") or 5.0
                met["wind_direction"] = wu_data.get("wind_direction") or 90.0
                met["rain_intensity"] = wu_data.get("rain_intensity") or 0.0
                st.sidebar.success("Weather Union API: Station data loaded successfully.")
            else:
                st.sidebar.warning("Locality outside Weather Union coverage. Showing flat forecast.")

    # Compute Wind Vector
    rad = math.radians(met["wind_direction"])
    u = -met["wind_speed"] * math.sin(rad)
    v = -met["wind_speed"] * math.cos(rad)

    # Spatial fields interpolation
    stations = build_weather_field(st.session_state.lat, st.session_state.lon, met, demo_mode)
    rain_field = idw_interpolate(st.session_state.lat, st.session_state.lon, stations)

    # Run Forecast Models
    grids_adv = run_classical_advection(rain_field, u, v)
    grids_fno = run_fno_inference(model, rain_field, u, v)
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

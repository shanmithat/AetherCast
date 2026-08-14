import math
import numpy as np
import requests
import time
from aethercast.config import GRID_RES, BOX_KM, WU_BASE_URL, WEATHER_UNION_API_KEY, CACHE_TTL_SECONDS

# Cache station pulls: (lat, lon, demo) -> (timestamp, stations_list, coverage_found, met_dict)
_station_cache = {}

def km_to_deg_lat(km):
    """Converts kilometers to degrees of latitude."""
    return km / 111.0

def km_to_deg_lon(km, at_lat):
    """Converts kilometers to degrees of longitude at a specific latitude."""
    return km / (111.0 * math.cos(math.radians(at_lat)))

def fetch_station(lat, lon):
    """Hits the Weather Union API for a single location coordinate."""
    if not WEATHER_UNION_API_KEY:
        return None
    headers = {"x-zomato-api-key": WEATHER_UNION_API_KEY}
    params = {"latitude": lat, "longitude": lon}
    try:
        resp = requests.get(WU_BASE_URL, headers=headers, params=params, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") in [200, "200"]:
                return data.get("locality_weather_data", {})
    except Exception as e:
        print(f"[WU API Fetch Error] at ({lat}, {lon}): {e}")
    return None

def sample_stations(center_lat, center_lon, demo_mode=False):
    """
    Query only the center coordinate to respect API rate limits.
    Synthesizes four surrounding virtual interpolation points with fixed perturbations.
    """
    cache_key = (round(center_lat, 2), round(center_lon, 2), demo_mode)
    now = time.time()
    
    cached = _station_cache.get(cache_key)
    if cached and (now - cached[0] < CACHE_TTL_SECONDS):
        time_remaining = int(CACHE_TTL_SECONDS - (now - cached[0]))
        return cached[1], cached[2], cached[3], time_remaining, True

    stations = []
    coverage_found = False
    
    met = {
        "temperature": 24.5,
        "humidity": 65.0,
        "wind_speed": 8.0,
        "wind_direction": 120.0,
        "rain_intensity": 0.0,
        "api_pulled": False
    }

    if demo_mode:
        met["rain_intensity"] = 18.5
        met["wind_speed"] = 14.5
        met["wind_direction"] = 135.0
        met["temperature"] = 22.8
        met["humidity"] = 88.0
        coverage_found = True
    else:
        data = fetch_station(center_lat, center_lon)
        met["api_pulled"] = True
        
        if data:
            coverage_found = True
            met["temperature"] = data.get("temperature") if data.get("temperature") is not None else 25.0
            met["humidity"] = data.get("humidity") if data.get("humidity") is not None else 70.0
            met["wind_speed"] = data.get("wind_speed") if data.get("wind_speed") is not None else 5.0
            met["wind_direction"] = data.get("wind_direction") if data.get("wind_direction") is not None else 90.0
            met["rain_intensity"] = data.get("rain_intensity") if data.get("rain_intensity") is not None else 0.0
        else:
            coverage_found = False
            met["rain_intensity"] = 0.0
            met["wind_speed"] = 0.0
            met["wind_direction"] = 0.0

    # Center localized Weather Union observation
    stations.append({
        "lat": center_lat, "lon": center_lon,
        "rain_intensity": met["rain_intensity"],
        "wind_speed": met["wind_speed"],
        "wind_direction": met["wind_direction"]
    })

    # Add 4 virtual interpolation points with fixed synthetic spatial perturbations.
    # These virtual points act as spatial support points to map the initial grid field.
    dlat = km_to_deg_lat(BOX_KM / 1.5)
    dlon = km_to_deg_lon(BOX_KM / 1.5, center_lat)
    
    offsets = [
        (dlat, 0, 0.7),    # North
        (-dlat, 0, 1.2),   # South
        (0, dlon, 0.8),    # East
        (0, -dlon, 1.1)    # West
    ]

    for lat_off, lon_off, rain_mult in offsets:
        p_rain = met["rain_intensity"] * rain_mult if met["rain_intensity"] > 0 else 0.0
        stations.append({
            "lat": center_lat + lat_off,
            "lon": center_lon + lon_off,
            "rain_intensity": p_rain,
            "wind_speed": met["wind_speed"],
            "wind_direction": met["wind_direction"]
        })

    _station_cache[cache_key] = (now, stations, coverage_found, met)
    return stations, coverage_found, met, CACHE_TTL_SECONDS, False

def idw_interpolate(center_lat, center_lon, stations, power=2):
    """
    Computes Inverse Distance Weighting (IDW) interpolation.
    Uses local kilometer coordinates to ensure physical consistency with the 30 km box domain.
    """
    field = np.zeros((GRID_RES, GRID_RES))
    
    # Local grid offsets in kilometers
    y_coords = np.linspace(-BOX_KM, BOX_KM, GRID_RES)  # rows
    x_coords = np.linspace(-BOX_KM, BOX_KM, GRID_RES)  # cols
    
    s_y = []
    s_x = []
    s_val = []
    
    for s in stations:
        dy = (s["lat"] - center_lat) * 111.0
        dx = (s["lon"] - center_lon) * 111.0 * math.cos(math.radians(center_lat))
        s_y.append(dy)
        s_x.append(dx)
        s_val.append(s["rain_intensity"])
        
    s_y = np.array(s_y)
    s_x = np.array(s_x)
    s_val = np.array(s_val)
    
    for i in range(GRID_RES):
        for j in range(GRID_RES):
            d = np.hypot(x_coords[j] - s_x, y_coords[i] - s_y) + 1e-6
            w = 1.0 / (d ** power)
            field[i, j] = np.sum(w * s_val) / np.sum(w)
            
    return field

def mean_wind_vector(stations):
    """Computes mean wind velocity U and V components across the grid."""
    u_sum, v_sum = 0.0, 0.0
    for s in stations:
        speed = s["wind_speed"]
        rad = math.radians(s["wind_direction"])
        u_sum += -speed * math.sin(rad)
        v_sum += -speed * math.cos(rad)
    n = len(stations)
    return u_sum / n, v_sum / n

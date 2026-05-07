#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
import shutil
import s3fs
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import requests
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.tri import Triangulation
from pyproj import Transformer
from shapely.geometry import box

# -------------------------------------------------
# PATHS
# -------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
DATA_DIR = ROOT / "data"
SHAPE_DIR = DATA_DIR / "shapes"
RAW_DIR = DATA_DIR / "raw"

RAW_DIR.mkdir(parents=True, exist_ok=True)

COUNTY_SHP = SHAPE_DIR / "county" / "c_16ap26.shp"
CWA_SHP = SHAPE_DIR / "cwa" / "w_16ap26.shp"
STATE_SHP = SHAPE_DIR / "state" / "s_16ap26.shp"

MANUAL_EXCLUDES_JSON = DATA_DIR / "manual_station_excludes.json"
OUT_MANIFEST = DOCS_DIR / "latest_station_maps.json"

# -------------------------------------------------
# MAP CONFIG
# -------------------------------------------------
PLOT_BBOX = (-92.5, 28.5, -87.0, 31.8)

REGIONS: dict[str, dict[str, Any]] = {
    "full": {
        "label": "Full LIX Area",
        "bbox": PLOT_BBOX,
        "suffix": "",
    },
    "baton_rouge_metro": {
        "label": "Baton Rouge Metro",
        "bbox": (-91.55, 30.20, -90.85, 30.75),
        "suffix": "_baton_rouge_metro",
    },
    "new_orleans_metro": {
        "label": "New Orleans Metro",
        "bbox": (-90.60, 29.55, -89.55, 30.35),
        "suffix": "_new_orleans_metro",
    },
    "southwest_ms": {
        "label": "Southwest MS",
        "bbox": (-91.25, 30.80, -89.75, 31.85),
        "suffix": "_southwest_ms",
    },
    "coastal_ms": {
        "label": "Coastal MS",
        "bbox": (-89.75, 30.05, -88.25, 30.70),
        "suffix": "_coastal_ms",
    },
    "northshore": {
        "label": "Northshore of Lake Pontchartrain",
        "bbox": (-90.45, 30.15, -89.45, 30.75),
        "suffix": "_northshore",
    },
}

INDEX_BBOX = (-95.0, 28.2, -84.0, 35.0)
TARGET_CRS = 3857

TITLE_OFFICE = "National Weather Service New Orleans/Baton Rouge Louisiana"
LOCAL_TZ = ZoneInfo("America/Chicago")
OBS_WINDOW_END_LOCAL_HOUR = 6

CITIES = [
    {"name": "Baton Rouge", "lat": 30.4515, "lon": -91.0},
    {"name": "New Orleans", "lat": 29.9511, "lon": -90.0715},
    {"name": "Gulfport", "lat": 30.3674, "lon": -89.0928},
    {"name": "McComb", "lat": 31.2438, "lon": -90.4532},
    {"name": "Houma", "lat": 29.5958, "lon": -90.7195},
]

DATASETS: dict[str, dict[str, Any]] = {
    "precip_24h": {
        "label": "24 Hour Rainfall",
        "csv": "station_precip_24h_latest.csv",
        "value_col": "precip_in",
        "units": "Inches",
        "title_suffix": "24 Hour Rainfall",
        "png": "lix_station_precip_24h_latest.png",
        "levels": [0.01, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 2.50, 3.00, 4.00, 5.00, 6.00, 8.00, 10.00, 15.00, 30.00],
        "colors": [
            "#67c6e5", "#6e9ad0", "#4b4aa7", "#57ea58", "#52b852",
            "#4d8c50", "#eceb59", "#efd27a", "#eda24f", "#ff4b4b",
            "#c7484d", "#a44a50", "#e43ee0", "#9362d6", "#d9d9d9", "#bcbcbc",
        ],
        "value_fmt": "{:.2f}",
        "tri_edge_km": 95.0,
        "neighbor_radius_km": 90.0,
        "neighbor_min": 3,
        "buddy_threshold": 4.0,
        "desc": "MRMS 24-Hour Pass 2 QPE Gridded Background + Station Observations",
    },
    "air_temp_latest": {
        "label": "Current Temperature",
        "csv": "station_air_temp_latest.csv",
        "value_col": "air_temp_f",
        "units": "Degrees F",
        "title_suffix": "Current Temperatures",
        "png": "lix_station_air_temp_latest.png",
        "levels": [35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
        "colors": [
            "#4b4aa7", "#6e9ad0", "#67c6e5", "#1db4b0", "#007f5f",
            "#3aa655", "#8ccf7e", "#c7e9ad", "#f2e788", "#f2c66d",
            "#f59d3d", "#f04e37", "#cc1f1a",
        ],
        "value_fmt": "{:.0f}",
        "tri_edge_km": 95.0,
        "neighbor_radius_km": 90.0,
        "neighbor_min": 3,
        "buddy_threshold": 12.0,
        "desc": "URMA Gridded Background + Station Observations",
    },
    "air_temp_daily_min": {
        "label": "Daily Minimum Temperature",
        "csv": "station_air_temp_daily_min_latest.csv",
        "value_col": "air_temp_min_f",
        "units": "Degrees F",
        "title_suffix": "24 Hour Low Temperatures",
        "png": "lix_station_air_temp_daily_min_latest.png",
        "levels": [35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90],
        "colors": [
            "#4b4aa7", "#6e9ad0", "#67c6e5", "#1db4b0", "#007f5f",
            "#3aa655", "#8ccf7e", "#c7e9ad", "#f2e788", "#f2c66d", "#f59d3d",
        ],
        "value_fmt": "{:.0f}",
        "tri_edge_km": 95.0,
        "neighbor_radius_km": 90.0,
        "neighbor_min": 3,
        "buddy_threshold": 12.0,
        "desc": "URMA 24-Hour Gridded Minimum + Station Observations",
    },
    "air_temp_daily_max": {
        "label": "Daily Maximum Temperature",
        "csv": "station_air_temp_daily_max_latest.csv",
        "value_col": "air_temp_max_f",
        "units": "Degrees F",
        "title_suffix": "24 Hour High Temperatures",
        "png": "lix_station_air_temp_daily_max_latest.png",
        "levels": [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
        "colors": [
            "#4b4aa7", "#6e9ad0", "#67c6e5", "#1db4b0", "#007f5f",
            "#3aa655", "#8ccf7e", "#c7e9ad", "#f2e788", "#f2c66d",
            "#f59d3d", "#f04e37",
        ],
        "value_fmt": "{:.0f}",
        "tri_edge_km": 95.0,
        "neighbor_radius_km": 90.0,
        "neighbor_min": 3,
        "buddy_threshold": 12.0,
        "desc": "URMA 24-Hour Gridded Maximum + Station Observations",
    },
}

# -------------------------------------------------
# URMA FETCHING & PROCESSING (Temperature)
# -------------------------------------------------
def download_urma_hour(dt: datetime) -> Path | None:
    dt = dt.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    date_str = dt.strftime("%Y%m%d")
    hour_str = dt.strftime("%H")
    file_name = f"urma2p5.t{hour_str}z.2dvaranl_ndfd.grb2_wexp"

    url = (
        f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/urma/prod/"
        f"urma2p5.{date_str}/{file_name}"
    )

    dest = RAW_DIR / f"urma_{date_str}{hour_str}.grb2"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest

    try:
        with requests.get(url, stream=True, timeout=60, headers={"User-Agent": "Mozilla/5.0"}) as r:
            if r.status_code != 200:
                print(f"URMA missing for {date_str} {hour_str}Z (status={r.status_code})")
                return None

            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"Downloaded URMA for {date_str} {hour_str}Z")
            return dest

    except Exception as e:
        print(f"URMA request failed for {date_str} {hour_str}Z: {e}")

    return None
def get_latest_daily_temp_windows(now_utc: datetime) -> dict[str, dict[str, datetime]]:
    """
    Return the exact local/UTC windows used for latest daily high/low maps.
    """
    now_local = now_utc.astimezone(LOCAL_TZ)
    today = now_local.date()
    yesterday = today - timedelta(days=1)

    def full_local_day(day):
        start_local = datetime(day.year, day.month, day.day, 0, 0, tzinfo=LOCAL_TZ)
        end_local = start_local + timedelta(days=1)
        return {
            "start_local": start_local,
            "end_local": end_local,
            "start_utc": start_local.astimezone(timezone.utc),
            "end_utc": end_local.astimezone(timezone.utc),
        }

    def partial_today():
        start_local = datetime(today.year, today.month, today.day, 0, 0, tzinfo=LOCAL_TZ)
        end_local = now_utc.astimezone(LOCAL_TZ)
        return {
            "start_local": start_local,
            "end_local": end_local,
            "start_utc": start_local.astimezone(timezone.utc),
            "end_utc": end_local.astimezone(timezone.utc),
        }

    if now_local.hour < 16:
        high_window = full_local_day(yesterday)
    else:
        high_window = full_local_day(today)

    if now_local.hour >= 9:
        low_window = partial_today()
    else:
        low_window = full_local_day(yesterday)

    return {
        "air_temp_daily_max": high_window,
        "air_temp_daily_min": low_window,
    }

def process_urma() -> dict[str, Any]:
    """
    Build URMA background grids that match the station products.

    - Current temp: latest available URMA hour
    - Daily highs/lows: same time windows used by fetch_lix_obs.py
    """
    now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    latest_dt = None
    latest_path = None

    for i in range(18):
        dt = now_utc - timedelta(hours=i)
        p = download_urma_hour(dt)
        if p:
            latest_dt = dt
            latest_path = p
            print(f"Using latest available URMA hour: {latest_dt.strftime('%Y%m%d %HZ')}")
            break

    if not latest_dt or not latest_path:
        raise RuntimeError("Could not fetch any recent URMA data from NOMADS.")

    lon, lat = None, None

    try:
        ds = xr.open_dataset(latest_path, engine="cfgrib", backend_kwargs={"indexpath": ""})
        latest_k = ds["t2m"].values.astype(float)
        lon = ds.longitude.values
        lat = ds.latitude.values
        ds.close()
    except Exception as e:
        raise RuntimeError(f"Failed to decode latest URMA grid {latest_path.name}: {e}") from e

    latest_f = (latest_k - 273.15) * 1.8 + 32.0

    windows = get_latest_daily_temp_windows(now_utc)

    def build_urma_extreme(window: dict[str, datetime], mode: str) -> np.ndarray:
        grids = []

        start_utc = window["start_utc"].replace(minute=0, second=0, microsecond=0)
        end_utc = window["end_utc"].replace(minute=0, second=0, microsecond=0)

        print(
            f"URMA {mode} window: "
            f"{window['start_local'].strftime('%Y-%m-%d %H:%M %Z')} to "
            f"{window['end_local'].strftime('%Y-%m-%d %H:%M %Z')} "
            f"({start_utc.strftime('%Y%m%d %HZ')} to {end_utc.strftime('%Y%m%d %HZ')})"
        )

        dt = start_utc
        while dt <= end_utc:
            p = download_urma_hour(dt)

            if p:
                try:
                    ds = xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": ""})
                    grids.append(ds["t2m"].values.astype(float))
                    ds.close()
                except Exception as e:
                    print(f"Failed to decode daily URMA grid {p.name}: {e}")

            dt += timedelta(hours=1)

        print(f"Successfully retrieved {len(grids)} URMA hourly grids for {mode}.")

        if len(grids) < 6:
            raise RuntimeError(f"Too few valid URMA grids retrieved for {mode}: {len(grids)}")

        stack_k = np.array(grids)
        stack_f = (stack_k - 273.15) * 1.8 + 32.0

        if mode == "max":
            return np.nanmax(stack_f, axis=0)

        return np.nanmin(stack_f, axis=0)

    max_f = build_urma_extreme(windows["air_temp_daily_max"], "max")
    min_f = build_urma_extreme(windows["air_temp_daily_min"], "min")

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x_3857, y_3857 = transformer.transform(lon, lat)

    return {
        "x": x_3857,
        "y": y_3857,
        "air_temp_latest": latest_f,
        "air_temp_daily_max": max_f,
        "air_temp_daily_min": min_f,
    }
# -------------------------------------------------
# MRMS FETCHING & PROCESSING (Precipitation)
# -------------------------------------------------
def download_mrms(dt: datetime) -> Path | None:
    dt = dt.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    date_str = dt.strftime("%Y%m%d")
    hour_str = dt.strftime("%H")

    product = "MultiSensor_QPE_24H_Pass2_00.00"
    expected_suffix = f"_{date_str}-{hour_str}0000.grib2.gz"

    dest_gz = RAW_DIR / f"MRMS_{product}_{date_str}-{hour_str}0000.grib2.gz"
    dest_grib = RAW_DIR / f"MRMS_{product}_{date_str}-{hour_str}0000.grib2"

    if dest_grib.exists() and dest_grib.stat().st_size > 1_000_000:
        return dest_grib

    try:
        fs = s3fs.S3FileSystem(anon=True)
        prefix = f"noaa-mrms-pds/CONUS/{product}/{date_str}/"

        try:
            files = fs.ls(prefix, refresh=True)
        except Exception:
            print(f"MRMS AWS directory missing for {date_str}")
            return None

        match = next((f for f in files if f.endswith(expected_suffix)), None)

        if not match:
            print(f"MRMS missing for {date_str} {hour_str}Z in AWS bucket")
            return None

        url = "https://noaa-mrms-pds.s3.amazonaws.com/" + match[len("noaa-mrms-pds/"):]

        with requests.get(url, stream=True, timeout=120, headers={"User-Agent": "Mozilla/5.0"}) as r:
            if r.status_code != 200:
                print(f"MRMS AWS download failed for {date_str} {hour_str}Z (status={r.status_code})")
                return None

            with open(dest_gz, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        with gzip.open(dest_gz, "rb") as f_in:
            with open(dest_grib, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        dest_gz.unlink()

        if dest_grib.exists() and dest_grib.stat().st_size > 1_000_000:
            print(f"Downloaded MRMS from AWS for {date_str} {hour_str}Z")
            return dest_grib

    except Exception as e:
        print(f"MRMS AWS request failed for {date_str} {hour_str}Z: {e}")

    return None

def process_mrms() -> dict[str, Any]:
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    valid_path = None

    for i in range(18):
        dt = now - timedelta(hours=i)
        p = download_mrms(dt)
        if p:
            valid_path = p
            print(f"Using latest available MRMS hour: {dt.strftime('%Y%m%d %HZ')}")
            break

    if not valid_path:
        raise RuntimeError("Could not fetch any recent MRMS data.")

    try:
        ds = xr.open_dataset(valid_path, engine="cfgrib", backend_kwargs={"indexpath": ""})
        print(ds)

        var_name = list(ds.data_vars)[0]
        precip_raw = ds[var_name].values
        lon = ds.longitude.values
        lat = ds.latitude.values

        print(f"MRMS variable name: {var_name}")
        print(f"Raw precip shape: {precip_raw.shape}")
        print(f"Raw lon shape: {lon.shape}")
        print(f"Raw lat shape: {lat.shape}")

        ds.close()

    except Exception as e:
        raise RuntimeError(f"Failed to decode MRMS: {e}")

    # Flatten lon/lat to 1-D if needed
    if lon.ndim == 2:
        lon = lon[0, :]
    if lat.ndim == 2:
        lat = lat[:, 0]

    # Convert 0..360 longitude to -180..180 if needed
    if np.nanmax(lon) > 180:
        lon = lon - 360.0

    # Sort longitude ascending and reorder precip to match
    lon_order = np.argsort(lon)
    lon = lon[lon_order]
    precip_raw = precip_raw[:, lon_order]

    # Sort latitude ascending and reorder precip to match
    lat_order = np.argsort(lat)
    lat = lat[lat_order]
    precip_raw = precip_raw[lat_order, :]

    # Convert mm to inches
    precip_in = precip_raw / 25.4
    precip_in = np.where(np.isfinite(precip_in), precip_in, np.nan)
    precip_in = np.where(precip_in < 0, 0, precip_in)

    print(f"Fixed lon range: {np.nanmin(lon):.3f} to {np.nanmax(lon):.3f}")
    print(f"Fixed lat range: {np.nanmin(lat):.3f} to {np.nanmax(lat):.3f}")
    print(f"Precip inches min/max: {np.nanmin(precip_in):.3f} / {np.nanmax(precip_in):.3f}")

    lon2d, lat2d = np.meshgrid(lon, lat)

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x_3857, y_3857 = transformer.transform(lon2d, lat2d)

    return {
        "x": x_3857,
        "y": y_3857,
        "precip_24h": precip_in,
    }

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
@dataclass
class GeoContext:
    lix: gpd.GeoDataFrame
    counties: gpd.GeoDataFrame
    states: gpd.GeoDataFrame
    plot_domain: gpd.GeoDataFrame
    index_domain: gpd.GeoDataFrame
    cities: gpd.GeoDataFrame

def load_manual_excludes() -> dict[str, list[str]]:
    if not MANUAL_EXCLUDES_JSON.exists():
        return {"all": []}
    try:
        data = json.loads(MANUAL_EXCLUDES_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {"all": []}
    if not isinstance(data, dict):
        return {"all": []}

    cleaned: dict[str, list[str]] = {}
    for key, value in data.items():
        if isinstance(value, list):
            cleaned[key] = [str(v).strip().upper() for v in value if str(v).strip()]
    if "all" not in cleaned:
        cleaned["all"] = []
    return cleaned

def station_is_manually_excluded(dataset_key: str, stid: str, manual: dict[str, list[str]]) -> bool:
    stid = str(stid).strip().upper()
    return stid in manual.get("all", []) or stid in manual.get(dataset_key, [])

def modified_zscore_flags(values: np.ndarray, threshold: float = 4.5) -> np.ndarray:
    if len(values) < 5:
        return np.zeros(len(values), dtype=bool)
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad == 0 or not np.isfinite(mad):
        return np.zeros(len(values), dtype=bool)
    z = 0.6745 * (values - med) / mad
    return np.abs(z) > threshold

def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    r = 6371.0
    dlat = np.radians(lat1 - lat2)
    dlon = np.radians(lon1 - lon2)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(np.radians(lat2)) * np.cos(np.radians(lat1)) * np.sin(dlon / 2.0) ** 2
    return 2 * r * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def buddy_check_flags(
    lats: np.ndarray,
    lons: np.ndarray,
    values: np.ndarray,
    radius_km: float,
    min_neighbors: int,
    threshold: float,
) -> np.ndarray:
    n = len(values)
    flags = np.zeros(n, dtype=bool)
    for i in range(n):
        d = haversine_km(lats, lons, lats[i], lons[i])
        mask = (d > 0) & (d <= radius_km)
        neighbors = values[mask]
        if len(neighbors) < min_neighbors:
            continue
        med = np.median(neighbors)
        if abs(values[i] - med) > threshold:
            flags[i] = True
    return flags

def build_dataset_rows(dataset_key: str, config: dict[str, Any], manual: dict[str, list[str]]) -> pd.DataFrame:
    csv_path = DOCS_DIR / config["csv"]
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    value_col = config["value_col"]

    df = df.copy()
    df["stid"] = df["stid"].astype(str).str.upper()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["lat", "lon", value_col])

    df["manual_exclude"] = df["stid"].apply(lambda s: station_is_manually_excluded(dataset_key, s, manual))

    values = df[value_col].to_numpy(dtype=float)
    lats = df["lat"].to_numpy(dtype=float)
    lons = df["lon"].to_numpy(dtype=float)

    global_flags = modified_zscore_flags(values, threshold=4.5)
    buddy_flags = buddy_check_flags(
        lats=lats,
        lons=lons,
        values=values,
        radius_km=config["neighbor_radius_km"],
        min_neighbors=config["neighbor_min"],
        threshold=config["buddy_threshold"],
    )

    df["auto_outlier"] = global_flags | buddy_flags
    df["exclude_from_contours"] = df["manual_exclude"] | df["auto_outlier"]

    return df

def load_geography(plot_bbox: tuple[float, float, float, float] = PLOT_BBOX) -> GeoContext:
    cwa = gpd.read_file(CWA_SHP).to_crs(4326)
    counties = gpd.read_file(COUNTY_SHP).to_crs(4326)
    states = gpd.read_file(STATE_SHP).to_crs(4326)

    lix = cwa[cwa["CWA"] == "LIX"].copy()
    plot_bounds = box(*plot_bbox)
    index_bounds = box(*INDEX_BBOX)

    counties = counties[counties.intersects(plot_bounds)].copy()
    states = states[states.intersects(index_bounds)].copy()

    plot_domain = gpd.GeoDataFrame(geometry=[plot_bounds], crs=4326)
    index_domain = gpd.GeoDataFrame(geometry=[index_bounds], crs=4326)

    cities = gpd.GeoDataFrame(
        CITIES,
        geometry=gpd.points_from_xy([c["lon"] for c in CITIES], [c["lat"] for c in CITIES]),
        crs=4326,
    )

    cities = cities[cities.intersects(plot_bounds)].copy()

    return GeoContext(
        lix=lix.to_crs(TARGET_CRS),
        counties=counties.to_crs(TARGET_CRS),
        states=states.to_crs(TARGET_CRS),
        plot_domain=plot_domain.to_crs(TARGET_CRS),
        index_domain=index_domain.to_crs(TARGET_CRS),
        cities=cities.to_crs(TARGET_CRS),
    )

def triangle_edge_lengths_km(x: np.ndarray, y: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    x1, y1 = x[triangles[:, 0]], y[triangles[:, 0]]
    x2, y2 = x[triangles[:, 1]], y[triangles[:, 1]]
    x3, y3 = x[triangles[:, 2]], y[triangles[:, 2]]

    a = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) / 1000.0
    b = np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2) / 1000.0
    c = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2) / 1000.0
    return np.maximum.reduce([a, b, c])
def thin_precip_label_points(
    df_g: gpd.GeoDataFrame,
    value_col: str,
    max_labels: int = 110,
    min_distance_km: float = 8.0,
) -> gpd.GeoDataFrame:
    """
    Thin 24-hour rainfall station labels so the map remains readable.

    Strategy:
    - Higher rainfall totals get first priority.
    - Light/zero amounts are allowed only sparsely.
    - Labels must be separated by a minimum map distance.
    """

    if df_g.empty:
        return df_g

    work = df_g.copy()
    work["_label_value"] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=["_label_value", "geometry"])

    if work.empty:
        return work

    # Priority: biggest rainfall totals first, then representative smaller totals.
    work = work.sort_values("_label_value", ascending=False)

    selected_rows = []
    selected_xy: list[tuple[float, float]] = []

    base_dist_m = min_distance_km * 1000.0

    for _, row in work.iterrows():
        value = float(row["_label_value"])
        x = float(row.geometry.x)
        y = float(row.geometry.y)

        # Don’t let 0.00/trace values stampede the map like cattle.
        if value < 0.10 and len(selected_rows) >= 30:
            continue

        # Use larger spacing for lower-value labels.
        if value >= 1.00:
            required_dist_m = base_dist_m * 0.85
        elif value >= 0.50:
            required_dist_m = base_dist_m
        elif value >= 0.10:
            required_dist_m = base_dist_m * 1.35
        else:
            required_dist_m = base_dist_m * 2.00

        too_close = False
        for sx, sy in selected_xy:
            dist_m = ((x - sx) ** 2 + (y - sy) ** 2) ** 0.5
            if dist_m < required_dist_m:
                too_close = True
                break

        if too_close:
            continue

        selected_rows.append(row)
        selected_xy.append((x, y))

        if len(selected_rows) >= max_labels:
            break

    if not selected_rows:
        return work.head(0)

    return gpd.GeoDataFrame(selected_rows, crs=df_g.crs).drop(columns=["_label_value"], errors="ignore")

def thin_label_points_by_distance(
    df_g: gpd.GeoDataFrame,
    value_col: str,
    max_labels: int = 150,
    min_distance_km: float = 12.0,
) -> gpd.GeoDataFrame:
    """
    Thin station labels by map distance while keeping spatial coverage.

    Strategy:
    - Sort north-to-south/west-to-east-ish so the result is not biased only toward max/min values.
    - Keep a label only if it is far enough from already-selected labels.
    - Works in projected meters, so distance behaves consistently on the map.
    """

    if df_g.empty:
        return df_g

    work = df_g.copy()
    work["_label_value"] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=["_label_value", "geometry"])

    if work.empty:
        return work

    # Spatially stable ordering. This avoids only keeping the highest/lowest values.
    work["_sort_x"] = work.geometry.x
    work["_sort_y"] = work.geometry.y
    work = work.sort_values(["_sort_y", "_sort_x"], ascending=[False, True])

    selected_rows = []
    selected_xy: list[tuple[float, float]] = []

    min_dist_m = min_distance_km * 1000.0

    for _, row in work.iterrows():
        x = float(row.geometry.x)
        y = float(row.geometry.y)

        too_close = False
        for sx, sy in selected_xy:
            dist_m = ((x - sx) ** 2 + (y - sy) ** 2) ** 0.5
            if dist_m < min_dist_m:
                too_close = True
                break

        if too_close:
            continue

        selected_rows.append(row)
        selected_xy.append((x, y))

        if len(selected_rows) >= max_labels:
            break

    if not selected_rows:
        return work.head(0).drop(columns=["_label_value", "_sort_x", "_sort_y"], errors="ignore")

    return gpd.GeoDataFrame(selected_rows, crs=df_g.crs).drop(
        columns=["_label_value", "_sort_x", "_sort_y"],
        errors="ignore",
    )

def build_triangulation(df_proj: gpd.GeoDataFrame, max_edge_km: float) -> Triangulation | None:
    if len(df_proj) < 3:
        return None
    x = df_proj.geometry.x.to_numpy()
    y = df_proj.geometry.y.to_numpy()

    tri = Triangulation(x, y)
    max_edges = triangle_edge_lengths_km(x, y, tri.triangles)
    tri.set_mask(max_edges > max_edge_km)
    return tri

def draw_legend(fig, config: dict[str, Any], levels: list[float], colors: list[str]) -> None:
    ax_leg = fig.add_axes([0.03, 0.05, 0.16, 0.52])
    ax_leg.set_facecolor("white")
    for s in ax_leg.spines.values():
        s.set_linewidth(1.5)
        s.set_color("black")
    ax_leg.set_xticks([])
    ax_leg.set_yticks([])

    label_text = config["label"].replace(" Temperature", "\nTemperature")
    ax_leg.text(0.5, 0.96, label_text, ha="center", va="top", fontsize=14, fontweight="bold", linespacing=1.3)
    ax_leg.text(0.5, 0.85, f"({config['units']})", ha="center", va="top", fontsize=13, fontweight="bold")

    labels = []
    for i in range(len(levels) - 1):
        if i == len(levels) - 2:
            labels.append(f"> {levels[i]:g}")
        else:
            labels.append(f"{levels[i]:g} - {levels[i+1]:g}")
    labels = labels[::-1]
    colors_rev = colors[::-1]

    y0 = 0.77
    dy = 0.048
    for i, (label, color) in enumerate(zip(labels, colors_rev)):
        y = y0 - i * dy
        ax_leg.add_patch(Rectangle((0.15, y - 0.016), 0.25, 0.03, color=color, transform=ax_leg.transAxes))
        ax_leg.text(0.48, y - 0.001, label, fontsize=12, va="center", ha="left", fontweight="bold")

def draw_inset(fig, geo: GeoContext) -> None:
    ax_in = fig.add_axes([0.03, 0.60, 0.16, 0.22])
    ax_in.set_facecolor("#ffffff")
    for s in ax_in.spines.values():
        s.set_linewidth(1.5)
        s.set_color("black")

    geo.states.plot(ax=ax_in, facecolor="#f0f0f0", edgecolor="#555555", linewidth=0.8, zorder=1)
    geo.lix.plot(ax=ax_in, facecolor="#ffb000", edgecolor="black", linewidth=1.2, zorder=2)

    lbls = gpd.GeoDataFrame(
        {"name": ["LA", "MS", "AL", "TX", "AR"]},
        geometry=gpd.points_from_xy([-92.2, -89.6, -86.8, -94.2, -92.5], [31.2, 32.8, 32.8, 31.5, 34.5]),
        crs=4326,
    ).to_crs(TARGET_CRS)

    for _, row in lbls.iterrows():
        ax_in.text(
            row.geometry.x,
            row.geometry.y,
            row["name"],
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#444444",
        )

    minx, miny, maxx, maxy = geo.index_domain.total_bounds
    ax_in.set_xlim(minx, maxx)
    ax_in.set_ylim(miny, maxy)
    ax_in.set_xticks([])
    ax_in.set_yticks([])

def filter_points_to_domain(points: gpd.GeoDataFrame, geo: GeoContext) -> gpd.GeoDataFrame:
    if points.empty:
        return points

    domain_geom = geo.plot_domain.geometry.iloc[0]
    return points[points.geometry.within(domain_geom)].copy()

def format_ampm(dt: datetime) -> str:
    hour = dt.strftime("%I").lstrip("0")
    return f"{hour} {dt.strftime('%p')}"


def format_month_day_year(dt: datetime) -> str:
    return f"{dt.month}/{dt.day}/{dt.year}"


def get_latest_valid_time_local() -> datetime:
    """
    Approximate current observation valid time for map labeling.
    """
    return datetime.now(timezone.utc).astimezone(LOCAL_TZ).replace(
        minute=0,
        second=0,
        microsecond=0,
    )


def get_observed_24h_window_local() -> tuple[datetime, datetime]:
    """
    Build a 24-hour observed window ending at 6 AM local time.
    """
    now_local = datetime.now(timezone.utc).astimezone(LOCAL_TZ)

    end_local = now_local.replace(
        hour=OBS_WINDOW_END_LOCAL_HOUR,
        minute=0,
        second=0,
        microsecond=0,
    )

    if now_local < end_local:
        end_local -= timedelta(days=1)

    start_local = end_local - timedelta(days=1)
    return start_local, end_local


def build_dynamic_title(dataset_key: str, config: dict[str, Any]) -> str:
    if dataset_key == "air_temp_latest":
        valid_local = get_latest_valid_time_local()
        return f"Current Temperatures {format_ampm(valid_local)}"

    if dataset_key in ("air_temp_daily_min", "air_temp_daily_max"):
        now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        windows = get_latest_daily_temp_windows(now_utc)
        window = windows[dataset_key]

        start_local = window["start_local"]

        if dataset_key == "air_temp_daily_max":
            label = f"High Temperatures - {start_local.strftime('%A, %B')} {start_local.day}, {start_local.year}"
        else:
            today_local = now_utc.astimezone(LOCAL_TZ).date()
            if start_local.date() == today_local:
                label = f"Low Temperatures This Morning - {start_local.strftime('%A, %B')} {start_local.day}, {start_local.year}"
            else:
                label = f"Low Temperatures - {start_local.strftime('%A, %B')} {start_local.day}, {start_local.year}"

        return label 

    if dataset_key == "precip_24h":
        start_local, end_local = get_observed_24h_window_local()
        return (
            f"24 Hour Rainfall "
            f"({format_month_day_year(start_local)} @ {format_ampm(start_local)} - "
            f"{format_month_day_year(end_local)} @ {format_ampm(end_local)}) - Observed"
        )

    return config["title_suffix"]

def plot_dataset(
    dataset_key: str,
    config: dict[str, Any],
    geo: GeoContext,
    manual: dict[str, list[str]],
    grid_data: dict[str, Any],
    region_key: str = "full",
    region_config: dict[str, Any] | None = None,
) -> dict[str, Any]:

    df = build_dataset_rows(dataset_key, config, manual)
    value_col = config["value_col"]

    df_g = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs=4326,
    ).to_crs(TARGET_CRS)

    used = df_g[~df_g["exclude_from_contours"]].copy()

    levels = config["levels"]
    colors = config["colors"]
    cmap = ListedColormap(colors)
    cmap.set_under((1, 1, 1, 0))
    norm = BoundaryNorm(levels, cmap.N, clip=False)

    minx, miny, maxx, maxy = geo.plot_domain.total_bounds

    is_regional = region_key != "full"

    if is_regional:
        fig = plt.figure(figsize=(14, 10), facecolor="#ffffff")
    else:
        fig = plt.figure(figsize=(16, 11.5), facecolor="#ffffff")

    if is_regional:
        ax_head = fig.add_axes([0.04, 0.90, 0.92, 0.08])
    else:
        ax_head = fig.add_axes([0.05, 0.88, 0.9, 0.12])
    ax_head.set_facecolor("#ffffff")
    for s in ax_head.spines.values():
        s.set_visible(False)
    ax_head.set_xticks([])
    ax_head.set_yticks([])

    dynamic_title = build_dynamic_title(dataset_key, config)

    region_title = ""
    if is_regional and region_config:
        region_title = f" - {region_config.get('label', region_key)}"
    
    if is_regional:
        ax_head.text(
            0.5,
            0.72,
            TITLE_OFFICE,
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
        )
        ax_head.text(
            0.5,
            0.28,
            f"{dynamic_title}{region_title}",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
        )
    else:
        ax_head.text(
            0.5,
            0.8,
            "National Weather Service",
            ha="center",
            va="center",
            fontsize=28,
            fontweight="bold",
            style="italic",
        )
        ax_head.text(
            0.5,
            0.45,
            TITLE_OFFICE,
            ha="center",
            va="center",
            fontsize=26,
            fontweight="bold",
            style="italic",
        )
        ax_head.text(
            0.5,
            0.1,
            dynamic_title,
            ha="center",
            va="center",
            fontsize=18 if dataset_key == "precip_24h" else 22,
            fontweight="bold",
            style="italic",
        )

    if is_regional:
        ax = fig.add_axes([0.06, 0.08, 0.88, 0.80])
    else:
        ax = fig.add_axes([0.22, 0.05, 0.75, 0.82])
    ax.set_facecolor("#efefef")
    for s in ax.spines.values():
        s.set_linewidth(1.8)
        s.set_color("black")

    if dataset_key in grid_data:
        x_grid = grid_data.get(f"{dataset_key}_x", grid_data.get("x"))
        y_grid = grid_data.get(f"{dataset_key}_y", grid_data.get("y"))
        ax.contourf(
            x_grid,
            y_grid,
            grid_data[dataset_key],
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend="both",
            zorder=0,
            antialiased=True,
        )
    else:
        tri = build_triangulation(used, max_edge_km=config["tri_edge_km"])
        if tri is not None and len(used) >= 3:
            z = used[value_col].to_numpy(dtype=float).copy()
            ax.tricontourf(tri, z, levels=levels, cmap=cmap, norm=norm, extend="both", zorder=0)

    plot_box_geom = geo.plot_domain.geometry.iloc[0]
    lix_union = geo.lix.geometry.union_all()
    outside_geom = plot_box_geom.difference(lix_union)
    outside_gdf = gpd.GeoDataFrame(geometry=[outside_geom], crs=geo.plot_domain.crs)
    outside_gdf.plot(ax=ax, facecolor="white", edgecolor="none", alpha=0.30, zorder=2)

    geo.counties.plot(ax=ax, facecolor="none", edgecolor="#9a9a9a", linewidth=0.55, zorder=3)
    geo.states.plot(ax=ax, facecolor="none", edgecolor="#555555", linewidth=1.4, zorder=4)
    geo.lix.boundary.plot(ax=ax, color="black", linewidth=2.2, zorder=5)

    if not is_regional:
        draw_inset(fig, geo)
        draw_legend(fig, config, levels, colors)

    if dataset_key == "precip_24h":
        if region_key == "full":
            # Full LIX rainfall map already has MRMS shading.
            # Hide station labels here to keep the map readable.
            label_points = df_g.head(0)
        else:
            label_points = thin_precip_label_points(
                df_g,
                value_col=value_col,
                max_labels=85,
                min_distance_km=4.0,
            )
    
    else:
        if region_key == "full":
            label_points = thin_label_points_by_distance(
                df_g,
                value_col=value_col,
                max_labels=145,
                min_distance_km=16.0,
            )
        else:
            label_points = df_g
    
    label_points = filter_points_to_domain(label_points, geo)

    for _, row in geo.cities.iterrows():
        ax.plot(row.geometry.x, row.geometry.y, "o", color="white", markeredgecolor="black", markersize=4.5, zorder=7)
        ax.text(
            row.geometry.x,
            row.geometry.y + 9000,
            row["name"],
            color="black",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="bottom",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
            zorder=8,
        )

    for _, row in label_points.iterrows():
        txt_color = "#ff2d2d" if row["exclude_from_contours"] else "white"
        ax.text(
            row.geometry.x,
            row.geometry.y,
            config["value_fmt"].format(row[value_col]),
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
            color=txt_color,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="black")],
            zorder=9,
        )

    ax.text(
        0.02,
        -0.015,
        config["desc"],
        transform=ax.transAxes,
        fontsize=10,
        ha="left",
        va="top",
    )

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_xticks([])
    ax.set_yticks([])

    region_config = region_config or REGIONS["full"]
    suffix = region_config.get("suffix", "")
    
    base_png = config["png"]
    if suffix:
        png_name = base_png.replace("_latest.png", f"{suffix}_latest.png")
    else:
        png_name = base_png
    
    png_path = DOCS_DIR / png_name
    if is_regional:
        fig.savefig(png_path, dpi=170)
    else:
        fig.savefig(png_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    
    return {
        "image": png_name,
        "csv": config["csv"],
        "station_count": int(len(df_g)),
        "used_in_contours": int(len(used)),
        "excluded": int(df_g["exclude_from_contours"].sum()),
        "region": region_key,
        "region_label": region_config.get("label", region_key),
        "status": "ok",
    }

def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    manual = load_manual_excludes()

    print("Initiating URMA Grid Processing...")
    try:
        urma_data = process_urma()
    except Exception as e:
        print(f"URMA FETCH FAILED: {e}")
        urma_data = {}

    print("Initiating MRMS Grid Processing...")
    mrms_ok = True
    
    try:
        mrms_data = process_mrms()
    except Exception as e:
        print(f"MRMS FETCH FAILED: {e}")
        mrms_data = {}
        mrms_ok = False

    grid_data = {**urma_data}
    if mrms_data:
        grid_data["precip_24h"] = mrms_data["precip_24h"]
        grid_data["precip_24h_x"] = mrms_data["x"]
        grid_data["precip_24h_y"] = mrms_data["y"]

    manifest: dict[str, Any] = {
        "regions": {
            key: {
                "label": value["label"],
                "bbox": value["bbox"],
            }
            for key, value in REGIONS.items()
        },
        "maps": {},
    }

    for dataset_key, config in DATASETS.items():
        if dataset_key == "precip_24h" and not mrms_ok:
            for region_key, region_config in REGIONS.items():
                suffix = region_config.get("suffix", "")
                if suffix:
                    stale_name = config["png"].replace("_latest.png", f"{suffix}_latest.png")
                else:
                    stale_name = config["png"]

                stale_png = DOCS_DIR / stale_name
                if stale_png.exists():
                    stale_png.unlink()
                    print(f"Deleted stale precip map: {stale_png.name}")

            manifest["maps"][dataset_key] = {
                "image": None,
                "regions": {},
                "csv": config["csv"],
                "station_count": 0,
                "used_in_contours": 0,
                "excluded": 0,
                "status": "MRMS unavailable",
            }

            print("Skipping precip_24h maps because MRMS data was unavailable.")
            continue

        print(f"Building {dataset_key} regional maps...")

        dataset_result: dict[str, Any] | None = None
        region_images: dict[str, str] = {}

        for region_key, region_config in REGIONS.items():
            print(f"  Building {dataset_key} / {region_config['label']}...")
            geo = load_geography(region_config["bbox"])
            result = plot_dataset(
                dataset_key,
                config,
                geo,
                manual,
                grid_data,
                region_key=region_key,
                region_config=region_config,
            )

            region_images[region_key] = result["image"]

            if region_key == "full":
                dataset_result = result

            print(f"  Saved {result['image']}")

        if dataset_result is None:
            dataset_result = {
                "image": None,
                "csv": config["csv"],
                "station_count": 0,
                "used_in_contours": 0,
                "excluded": 0,
                "status": "No maps generated",
            }

        dataset_result["regions"] = region_images
        manifest["maps"][dataset_key] = dataset_result

    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {OUT_MANIFEST.name}")
    print("Finished building station maps.")

if __name__ == "__main__":
    main()

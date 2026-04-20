#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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
from matplotlib.tri import Triangulation, UniformTriRefiner
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
INDEX_BBOX = (-95.0, 28.2, -84.0, 35.0)
TARGET_CRS = 3857

TITLE_OFFICE = "National Weather Service New Orleans/Baton Rouge Louisiana"

CITIES = [
    {"name": "Baton Rouge", "lat": 30.4515, "lon": -91.1871},
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
    if dest.exists() and dest.stat().st_size > 1000000:
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

        if dest.exists() and dest.stat().st_size > 1000000:
            print(f"Downloaded URMA for {date_str} {hour_str}Z")
            return dest

    except Exception as e:
        print(f"URMA request failed for {date_str} {hour_str}Z: {e}")

    return None

def process_urma() -> dict[str, Any]:
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    latest_dt = None

    for i in range(18):
        dt = now - timedelta(hours=i)
        if download_urma_hour(dt):
            latest_dt = dt
            print(f"Using latest available URMA hour: {latest_dt.strftime('%Y%m%d %HZ')}")
            break

    if not latest_dt:
        raise RuntimeError("Could not fetch any recent URMA data from NOMADS.")

    valid_paths = []
    for i in range(24):
        dt = latest_dt - timedelta(hours=i)
        p = download_urma_hour(dt)
        if p:
            valid_paths.append(p)

    print(f"Successfully retrieved {len(valid_paths)}/24 URMA hourly grids.")

    if len(valid_paths) < 6:
        raise RuntimeError(f"Too few valid URMA grids retrieved: {len(valid_paths)}")

    grids = []
    lon, lat = None, None

    for p in valid_paths:
        try:
            ds = xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": ""})
            grids.append(ds["t2m"].values)

            if lon is None:
                lon = ds.longitude.values
                lat = ds.latitude.values

            ds.close()

        except Exception as e:
            print(f"Failed to decode {p.name}: {e}")

    if not grids:
        raise RuntimeError("No valid URMA grids parsed.")

    stack = np.array(grids)

    # Kelvin to Fahrenheit
    temp_f = (stack - 273.15) * 1.8 + 32.0

    latest_f = temp_f[0]
    max_f = np.max(temp_f, axis=0)
    min_f = np.min(temp_f, axis=0)

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
    file_name = f"MultiSensor_QPE_24H_Pass2_00.00_{date_str}-{hour_str}0000.grib2.gz"

    url = (
        f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/mrms/prod/"
        f"mrms.{date_str}/CONUS/{file_name}"
    )

    dest_gz = RAW_DIR / file_name
    dest_grib = RAW_DIR / file_name.replace(".gz", "")

    if dest_grib.exists() and dest_grib.stat().st_size > 1000000:
        return dest_grib

    try:
        with requests.get(url, stream=True, timeout=60, headers={"User-Agent": "Mozilla/5.0"}) as r:
            if r.status_code != 200:
                print(f"MRMS missing for {date_str} {hour_str}Z (status={r.status_code})")
                return None

            with open(dest_gz, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        # MRMS files are compressed with gzip. Decompress them for xarray
        with gzip.open(dest_gz, 'rb') as f_in:
            with open(dest_grib, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        dest_gz.unlink() # Clean up the gz file

        if dest_grib.exists() and dest_grib.stat().st_size > 1000000:
            print(f"Downloaded MRMS for {date_str} {hour_str}Z")
            return dest_grib

    except Exception as e:
        print(f"MRMS request failed for {date_str} {hour_str}Z: {e}")

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
        raise RuntimeError("Could not fetch any recent MRMS data from NOMADS.")

    try:
        ds = xr.open_dataset(valid_path, engine="cfgrib", backend_kwargs={"indexpath": ""})
        print(ds)

        var_name = list(ds.data_vars)[0]
        precip_mm = ds[var_name].values

        lon = ds.longitude.values
        lat = ds.latitude.values
        ds.close()

    except Exception as e:
        raise RuntimeError(f"Failed to decode MRMS: {e}")

    # Force lon/lat to 1-D if cfgrib gives 2-D arrays
    if lon.ndim == 2:
        lon = lon[0, :]
    if lat.ndim == 2:
        lat = lat[:, 0]

    # Convert mm to inches
    precip_in = precip_mm / 25.4
    precip_in = np.where(precip_in < 0, 0, precip_in)

    # Make sure coordinate order matches data orientation
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        precip_in = precip_in[::-1, :]

    if lon[0] > lon[-1]:
        lon = lon[::-1]
        precip_in = precip_in[:, ::-1]

    # Build 2-D coordinate grids after orientation is fixed
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

def buddy_check_flags(lats: np.ndarray, lons: np.ndarray, values: np.ndarray, radius_km: float, min_neighbors: int, threshold: float) -> np.ndarray:
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
        lats=lats, lons=lons, values=values,
        radius_km=config["neighbor_radius_km"],
        min_neighbors=config["neighbor_min"],
        threshold=config["buddy_threshold"],
    )

    df["auto_outlier"] = global_flags | buddy_flags
    df["exclude_from_contours"] = df["manual_exclude"] | df["auto_outlier"]

    return df

def load_geography() -> GeoContext:
    cwa = gpd.read_file(CWA_SHP).to_crs(4326)
    counties = gpd.read_file(COUNTY_SHP).to_crs(4326)
    states = gpd.read_file(STATE_SHP).to_crs(4326)

    lix = cwa[cwa["CWA"] == "LIX"].copy()
    plot_bounds = box(*PLOT_BBOX)
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
        ax_in.text(row.geometry.x, row.geometry.y, row["name"], ha="center", va="center", fontsize=9, fontweight="bold", color="#444444")

    minx, miny, maxx, maxy = geo.index_domain.total_bounds
    ax_in.set_xlim(minx, maxx)
    ax_in.set_ylim(miny, maxy)
    ax_in.set_xticks([])
    ax_in.set_yticks([])

def plot_dataset(dataset_key: str, config: dict[str, Any], geo: GeoContext, manual: dict[str, list[str]], grid_data: dict[str, Any]) -> dict[str, Any]:
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

    fig = plt.figure(figsize=(16, 11.5), facecolor="#ffffff")

    ax_head = fig.add_axes([0.05, 0.88, 0.9, 0.12])
    ax_head.set_facecolor("#ffffff") 
    for s in ax_head.spines.values():
        s.set_visible(False) 
    ax_head.set_xticks([])
    ax_head.set_yticks([])

    ax_head.text(0.5, 0.8, "National Weather Service", ha="center", va="center", fontsize=28, fontweight="bold", style="italic")
    ax_head.text(0.5, 0.45, TITLE_OFFICE, ha="center", va="center", fontsize=26, fontweight="bold", style="italic")
    ax_head.text(0.5, 0.1, config["title_suffix"], ha="center", va="center", fontsize=22, fontweight="bold", style="italic")

    ax = fig.add_axes([0.22, 0.05, 0.75, 0.82])
    ax.set_facecolor("#efefef")
    for s in ax.spines.values():
        s.set_linewidth(1.8)
        s.set_color("black")

    # -----------------------------------------------------------------
    # PLOTTING THE BACKGROUND
    # If the dataset is in grid_data, we paint the high-res grid (URMA or MRMS).
    # Otherwise, we use the triangulated stations as a fallback.
    # -----------------------------------------------------------------
    if dataset_key in grid_data:
        x_grid = grid_data.get(f"{dataset_key}_x", grid_data.get("x"))
        y_grid = grid_data.get(f"{dataset_key}_y", grid_data.get("y"))
        
        ax.contourf(x_grid, y_grid, grid_data[dataset_key], levels=levels, cmap=cmap, norm=norm, extend="both", zorder=0, antialiased=True)
    else:
        # If URMA/MRMS fails, we fall back to a simple, accurate station-only map
        tri = build_triangulation(used, max_edge_km=config["tri_edge_km"])
        if tri is not None and len(used) >= 3:
            x = used.geometry.x.to_numpy()
            y = used.geometry.y.to_numpy()
            z = used[value_col].to_numpy(dtype=float).copy()

            ax.tricontourf(tri, z, levels=levels, cmap=cmap, norm=norm, extend="both", zorder=0)

    # -----------------------------------------------------------------

    plot_box_geom = geo.plot_domain.geometry.iloc[0]
    lix_union = geo.lix.geometry.union_all()
    outside_geom = plot_box_geom.difference(lix_union)
    outside_gdf = gpd.GeoDataFrame(geometry=[outside_geom], crs=geo.plot_domain.crs)
    outside_gdf.plot(ax=ax, facecolor="white", edgecolor="none", alpha=0.30, zorder=2)

    geo.counties.plot(ax=ax, facecolor="none", edgecolor="#9a9a9a", linewidth=0.55, zorder=3)
    geo.states.plot(ax=ax, facecolor="none", edgecolor="#555555", linewidth=1.4, zorder=4)
    geo.lix.boundary.plot(ax=ax, color="black", linewidth=2.2, zorder=5)

    draw_inset(fig, geo)
    draw_legend(fig, config, levels, colors)

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

    for _, row in df_g.iterrows():
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

    png_path = DOCS_DIR / config["png"]
    fig.savefig(png_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    return {
        "image": config["png"],
        "csv": config["csv"],
        "station_count": int(len(df_g)),
        "used_in_contours": int(len(used)),
        "excluded": int(df_g["exclude_from_contours"].sum()),
    }

def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    geo = load_geography()
    manual = load_manual_excludes()

    # Hit NOMADS and process the 24-hour URMA grids (Temperatures)
    print("Initiating URMA Grid Processing...")
    try:
        urma_data = process_urma()
    except Exception as e:
        print(f"URMA FETCH FAILED: {e}")
        urma_data = {}

    # Hit NOMADS and process the MRMS grids (Precipitation)
    print("Initiating MRMS Grid Processing...")
    try:
        mrms_data = process_mrms()
    except Exception as e:
        print(f"MRMS FETCH FAILED: {e}")
        mrms_data = {}

    # Merge into a single dictionary for the plotting function
    grid_data = {**urma_data}
    if mrms_data:
        grid_data["precip_24h"] = mrms_data["precip_24h"]
        grid_data["precip_24h_x"] = mrms_data["x"]
        grid_data["precip_24h_y"] = mrms_data["y"]

    manifest: dict[str, Any] = {"maps": {}}

    for dataset_key, config in DATASETS.items():
        print(f"Building {dataset_key}...")
        result = plot_dataset(dataset_key, config, geo, manual, grid_data)
        manifest["maps"][dataset_key] = result
        print(f"Saved {result['image']}")

    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Finished building station maps.")

if __name__ == "__main__":
    main()

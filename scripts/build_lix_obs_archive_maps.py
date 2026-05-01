#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import requests
import s3fs
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
DOCS_DIR.mkdir(parents=True, exist_ok=True)

COUNTY_SHP = SHAPE_DIR / "county" / "c_16ap26.shp"
CWA_SHP = SHAPE_DIR / "cwa" / "w_16ap26.shp"
STATE_SHP = SHAPE_DIR / "state" / "s_16ap26.shp"

MANUAL_EXCLUDES_JSON = DATA_DIR / "manual_station_excludes.json"
LOCAL_TZ = ZoneInfo("America/Chicago")

# -------------------------------------------------
# MAP CONFIG
# -------------------------------------------------
PLOT_BBOX = (-92.5, 28.5, -87.0, 31.8)
INDEX_BBOX = (-95.0, 28.2, -84.0, 35.0)
TARGET_CRS = 3857
TITLE_OFFICE = "National Weather Service New Orleans/Baton Rouge Louisiana"

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
        "neighbor_radius_km": 90.0,
        "neighbor_min": 3,
        "buddy_threshold": 4.0,
        "desc": "MRMS 24-Hour Pass 2 QPE Gridded Background + Station Observations",
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
        "neighbor_radius_km": 90.0,
        "neighbor_min": 3,
        "buddy_threshold": 12.0,
        "desc": "URMA 24-Hour Gridded Maximum + Station Observations",
    },
}

# -------------------------------------------------
# DATA CLASSES
# -------------------------------------------------
@dataclass
class GeoContext:
    lix: gpd.GeoDataFrame
    counties: gpd.GeoDataFrame
    states: gpd.GeoDataFrame
    plot_domain: gpd.GeoDataFrame
    index_domain: gpd.GeoDataFrame
    cities: gpd.GeoDataFrame


# -------------------------------------------------
# ARGUMENTS / TIME HELPERS
# -------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build historical WFO LIX station archive maps with date-aware MRMS/URMA grids."
    )
    parser.add_argument(
        "--end-time",
        required=True,
        help="Archive end time UTC in YYYYMMDDHHMM format. Usually 6 AM local converted to UTC.",
    )
    return parser.parse_args()


def parse_end_time(value: str) -> datetime:
    return datetime.strptime(value.strip(), "%Y%m%d%H%M").replace(tzinfo=timezone.utc)


def format_ampm(dt: datetime) -> str:
    hour = dt.strftime("%I").lstrip("0")
    return f"{hour} {dt.strftime('%p')}"


def format_month_day_year(dt: datetime) -> str:
    return f"{dt.month}/{dt.day}/{dt.year}"


def build_dynamic_title(dataset_key: str, config: dict[str, Any], end_utc: datetime) -> str:
    end_local = end_utc.astimezone(LOCAL_TZ)
    start_local = end_local - timedelta(hours=24)

    if dataset_key == "precip_24h":
        return (
            f"24 Hour Rainfall "
            f"({format_month_day_year(start_local)} @ {format_ampm(start_local)} - "
            f"{format_month_day_year(end_local)} @ {format_ampm(end_local)}) - Observed"
        )

    if dataset_key in ("air_temp_daily_min", "air_temp_daily_max"):
        day_label = f"{end_local.strftime('%A, %B')} {end_local.day}, {end_local.year}"
        return f"{config['title_suffix']} - {day_label}"

    return config["title_suffix"]


# -------------------------------------------------
# GEOGRAPHY / QC HELPERS
# -------------------------------------------------
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
    cleaned.setdefault("all", [])
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
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(np.radians(lat2)) * np.cos(np.radians(lat1)) * np.sin(dlon / 2.0) ** 2
    )
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
    if lix.empty:
        raise RuntimeError("Could not find CWA=LIX in CWA shapefile.")

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


# -------------------------------------------------
# GRID DOWNLOAD / PROCESSING
# -------------------------------------------------
def download_urma_hour(dt: datetime) -> Path | None:
    dt = dt.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    date_str = dt.strftime("%Y%m%d")
    hour_str = dt.strftime("%H")
    file_name = f"urma2p5.t{hour_str}z.2dvaranl_ndfd.grb2_wexp"
    url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/urma/prod/urma2p5.{date_str}/{file_name}"
    dest = RAW_DIR / f"urma_{date_str}{hour_str}.grb2"

    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest

    try:
        with requests.get(url, stream=True, timeout=90, headers={"User-Agent": "Mozilla/5.0"}) as r:
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


def decode_urma_t2m(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        ds = xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={
                "indexpath": "",
                "filter_by_keys": {
                    "typeOfLevel": "heightAboveGround",
                    "level": 2,
                },
            },
        )

        if "t2m" in ds.data_vars:
            da = ds["t2m"]
        else:
            # Fallback in case cfgrib names the 2m temp field differently.
            var_name = list(ds.data_vars)[0]
            da = ds[var_name]

        temp_k = da.values.astype(float)
        lon = ds.longitude.values
        lat = ds.latitude.values
        ds.close()

        temp_f = (temp_k - 273.15) * 1.8 + 32.0
        return temp_f, lon, lat
    except Exception as e:
        print(f"Failed to decode URMA 2m temp from {path.name}: {e}")
        return None


def process_urma_window(end_utc: datetime) -> dict[str, Any]:
    end_hour = end_utc.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    grids: list[np.ndarray] = []
    lon = None
    lat = None

    for i in range(24):
        dt = end_hour - timedelta(hours=i)
        path = download_urma_hour(dt)
        if not path:
            continue

        decoded = decode_urma_t2m(path)
        if decoded is None:
            continue

        temp_f, this_lon, this_lat = decoded
        grids.append(temp_f)
        if lon is None:
            lon = this_lon
            lat = this_lat

    if len(grids) < 6 or lon is None or lat is None:
        raise RuntimeError(f"Too few valid URMA grids retrieved for archive window ending {end_hour:%Y%m%d %HZ}: {len(grids)}")

    stack = np.array(grids)
    min_f = np.nanmin(stack, axis=0)
    max_f = np.nanmax(stack, axis=0)

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x_3857, y_3857 = transformer.transform(lon, lat)

    return {
        "x": x_3857,
        "y": y_3857,
        "air_temp_daily_min": min_f,
        "air_temp_daily_max": max_f,
    }


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

        with gzip.open(dest_gz, "rb") as f_in, open(dest_grib, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        dest_gz.unlink(missing_ok=True)

        if dest_grib.exists() and dest_grib.stat().st_size > 1_000_000:
            print(f"Downloaded MRMS from AWS for {date_str} {hour_str}Z")
            return dest_grib
    except Exception as e:
        print(f"MRMS AWS request failed for {date_str} {hour_str}Z: {e}")

    return None


def process_mrms_24h(end_utc: datetime) -> dict[str, Any]:
    target_dt = end_utc.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    valid_path = None
    valid_dt = None

    # Pass2 occasionally lags or has missing hours. Stay date-aware, but allow limited fallback.
    for i in range(18):
        dt = target_dt - timedelta(hours=i)
        path = download_mrms(dt)
        if path:
            valid_path = path
            valid_dt = dt
            print(f"Using MRMS 24h ending {dt:%Y%m%d %HZ}")
            break

    if not valid_path:
        raise RuntimeError(f"Could not fetch MRMS 24h Pass2 near {target_dt:%Y%m%d %HZ}")

    try:
        ds = xr.open_dataset(valid_path, engine="cfgrib", backend_kwargs={"indexpath": ""})
        var_name = list(ds.data_vars)[0]
        precip_raw = ds[var_name].values
        lon = ds.longitude.values
        lat = ds.latitude.values
        ds.close()
    except Exception as e:
        raise RuntimeError(f"Failed to decode MRMS {valid_path.name}: {e}") from e

    if lon.ndim == 2:
        lon = lon[0, :]
    if lat.ndim == 2:
        lat = lat[:, 0]
    if np.nanmax(lon) > 180:
        lon = lon - 360.0

    lon_order = np.argsort(lon)
    lon = lon[lon_order]
    precip_raw = precip_raw[:, lon_order]

    lat_order = np.argsort(lat)
    lat = lat[lat_order]
    precip_raw = precip_raw[lat_order, :]

    precip_in = precip_raw / 25.4
    precip_in = np.where(np.isfinite(precip_in), precip_in, np.nan)
    precip_in = np.where(precip_in < 0, 0, precip_in)

    lon2d, lat2d = np.meshgrid(lon, lat)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x_3857, y_3857 = transformer.transform(lon2d, lat2d)

    return {
        "precip_24h": precip_in,
        "precip_24h_x": x_3857,
        "precip_24h_y": y_3857,
        "mrms_valid_utc": valid_dt.isoformat() if valid_dt else None,
    }


# -------------------------------------------------
# LABEL THINNING / PLOTTING HELPERS
# -------------------------------------------------
def thin_label_points_by_distance(
    df_g: gpd.GeoDataFrame,
    value_col: str,
    max_labels: int = 150,
    min_distance_km: float = 12.0,
) -> gpd.GeoDataFrame:
    if df_g.empty:
        return df_g

    work = df_g.copy()
    work["_label_value"] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=["_label_value", "geometry"])
    if work.empty:
        return work

    work["_sort_x"] = work.geometry.x
    work["_sort_y"] = work.geometry.y
    work = work.sort_values(["_sort_y", "_sort_x"], ascending=[False, True])

    selected_rows = []
    selected_xy: list[tuple[float, float]] = []
    min_dist_m = min_distance_km * 1000.0

    for _, row in work.iterrows():
        x = float(row.geometry.x)
        y = float(row.geometry.y)
        too_close = any(((x - sx) ** 2 + (y - sy) ** 2) ** 0.5 < min_dist_m for sx, sy in selected_xy)
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


def filter_points_to_domain(points: gpd.GeoDataFrame, geo: GeoContext) -> gpd.GeoDataFrame:
    if points.empty:
        return points
    domain_geom = geo.plot_domain.geometry.iloc[0]
    return points[points.geometry.within(domain_geom)].copy()


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
            labels.append(f"{levels[i]:g} - {levels[i + 1]:g}")
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


def png_name_for_region(base_png: str, region_config: dict[str, Any]) -> str:
    suffix = region_config.get("suffix", "")
    if suffix:
        return base_png.replace("_latest.png", f"{suffix}_latest.png")
    return base_png


def plot_dataset(
    dataset_key: str,
    config: dict[str, Any],
    geo: GeoContext,
    manual: dict[str, list[str]],
    grid_data: dict[str, Any],
    end_utc: datetime,
    region_key: str,
    region_config: dict[str, Any],
) -> str:
    df = build_dataset_rows(dataset_key, config, manual)
    value_col = config["value_col"]

    df_g = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=4326).to_crs(TARGET_CRS)

    levels = config["levels"]
    colors = config["colors"]
    cmap = ListedColormap(colors)
    cmap.set_under((1, 1, 1, 0))
    norm = BoundaryNorm(levels, cmap.N, clip=False)

    minx, miny, maxx, maxy = geo.plot_domain.total_bounds
    is_regional = region_key != "full"

    fig = plt.figure(figsize=(14, 10) if is_regional else (16, 11.5), facecolor="#ffffff")

    ax_head = fig.add_axes([0.04, 0.90, 0.92, 0.08] if is_regional else [0.05, 0.88, 0.9, 0.12])
    ax_head.set_facecolor("#ffffff")
    for s in ax_head.spines.values():
        s.set_visible(False)
    ax_head.set_xticks([])
    ax_head.set_yticks([])

    dynamic_title = build_dynamic_title(dataset_key, config, end_utc)
    region_title = f" - {region_config.get('label', region_key)}" if is_regional else ""

    if is_regional:
        ax_head.text(0.5, 0.72, TITLE_OFFICE, ha="center", va="center", fontsize=20, fontweight="bold")
        ax_head.text(0.5, 0.28, f"{dynamic_title}{region_title}", ha="center", va="center", fontsize=16, fontweight="bold")
    else:
        ax_head.text(0.5, 0.8, "National Weather Service", ha="center", va="center", fontsize=28, fontweight="bold", style="italic")
        ax_head.text(0.5, 0.45, TITLE_OFFICE, ha="center", va="center", fontsize=26, fontweight="bold", style="italic")
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

    ax = fig.add_axes([0.06, 0.08, 0.88, 0.80] if is_regional else [0.22, 0.05, 0.75, 0.82])
    ax.set_facecolor("#efefef")
    for s in ax.spines.values():
        s.set_linewidth(1.8)
        s.set_color("black")

    x_grid = grid_data.get(f"{dataset_key}_x", grid_data.get("x"))
    y_grid = grid_data.get(f"{dataset_key}_y", grid_data.get("y"))
    z_grid = grid_data.get(dataset_key)

    if x_grid is not None and y_grid is not None and z_grid is not None:
        ax.contourf(
            x_grid,
            y_grid,
            z_grid,
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend="both",
            zorder=0,
            antialiased=True,
        )
    else:
        raise RuntimeError(f"Missing gridded background for {dataset_key}")

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

    # Label rules:
    # - Full LIX 24h rainfall: hide station labels; MRMS shading carries the full-map signal.
    # - Zoomed rainfall maps: show local station labels.
    # - Full temperature maps: thin station labels.
    # - Zoomed temperature maps: show all labels in the region.
    if dataset_key == "precip_24h" and region_key == "full":
        label_points = df_g.head(0)
    elif region_key == "full":
        label_points = thin_label_points_by_distance(df_g, value_col=value_col, max_labels=145, min_distance_km=16.0)
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

    ax.text(0.02, -0.015, config["desc"], transform=ax.transAxes, fontsize=10, ha="left", va="top")

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_xticks([])
    ax.set_yticks([])

    png_name = png_name_for_region(config["png"], region_config)
    png_path = DOCS_DIR / png_name
    if is_regional:
        fig.savefig(png_path, dpi=170)
    else:
        fig.savefig(png_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    return png_name


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main() -> None:
    args = parse_args()
    end_utc = parse_end_time(args.end_time)
    manual = load_manual_excludes()

    print(f"Building archive station maps for end time {end_utc:%Y-%m-%d %H:%MZ}")

    print("Processing date-aware URMA 24-hour temperature grids...")
    urma_data = process_urma_window(end_utc)

    print("Processing date-aware MRMS 24-hour rainfall grid...")
    mrms_data = process_mrms_24h(end_utc)

    grid_data: dict[str, Any] = {
        **urma_data,
        "precip_24h": mrms_data["precip_24h"],
        "precip_24h_x": mrms_data["precip_24h_x"],
        "precip_24h_y": mrms_data["precip_24h_y"],
    }

    for dataset_key, config in DATASETS.items():
        print(f"Building archive maps for {dataset_key}...")
        for region_key, region_config in REGIONS.items():
            geo = load_geography(region_config["bbox"])
            png_name = plot_dataset(
                dataset_key=dataset_key,
                config=config,
                geo=geo,
                manual=manual,
                grid_data=grid_data,
                end_utc=end_utc,
                region_key=region_key,
                region_config=region_config,
            )
            print(f"  Saved {png_name}")

    print("Finished building archive station maps.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import math
import os
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.ndimage import gaussian_filter
from shapely import contains_xy
from shapely.geometry import Point, box

# -----------------------------
# CONFIG
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SHAPE_DIR = DATA_DIR / "shapes"
OUT_DIR = ROOT / "docs"

RAW_DIR.mkdir(parents=True, exist_ok=True)
SHAPE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYNOPTIC_TOKEN = os.getenv("SYNOPTIC_TOKEN", "").strip()
if not SYNOPTIC_TOKEN:
    raise SystemExit("Missing SYNOPTIC_TOKEN environment variable.")

# Broader bbox than LIX CWA so interpolation behaves better near the edges.
QUERY_BBOX = (-92.7, 28.5, -87.0, 32.7)  # lonmin, latmin, lonmax, latmax

# Rough plotting extent around LIX CWA.
PLOT_BBOX = (-91.9, 28.8, -87.6, 31.8)

CWA_URL = "https://www.weather.gov/source/gis/Shapefiles/WSOM/w_16ap26.zip"
COUNTY_URL = "https://www.weather.gov/source/gis/Shapefiles/County/c_16ap26.zip"
SYNOPTIC_URL = "https://api.synopticdata.com/v2/stations/precip"

TITLE_OFFICE = "National Weather Service New Orleans/Baton Rouge Louisiana"
SUBTITLE = "Estimated 24 Hour Rainfall"
PREPARED_BY = "Graphic prepared by: WFO New Orleans/Baton Rouge"
DESCRIPTION = (
    "Liquid precipitation observed over the specified 24 hour period. The field is a station-based "
    "interpolation using Synoptic-derived precipitation totals. It is not radar-QPE."
)

LEVELS = [
    0.00, 0.01, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 2.50,
    3.00, 4.00, 5.00, 6.00, 8.00, 10.00, 12.00, 14.00, 16.00, 18.00, 30.00
]
COLORS = [
    "#f7f7f7",  # <0.01
    "#bfefff",  # 0.01-0.10
    "#25a7e1",  # 0.10-0.25
    "#1158b7",  # 0.25-0.50
    "#57ff00",  # 0.50-0.75
    "#38b000",  # 0.75-1.0
    "#1f7a00",  # 1.0-1.5
    "#f2ef00",  # 1.5-2.0
    "#eec56f",  # 2.0-2.5
    "#ff9a00",  # 2.5-3.0
    "#ff3900",  # 3.0-4.0
    "#d40000",  # 4.0-5.0
    "#b10000",  # 5.0-6.0
    "#ff00c8",  # 6.0-8.0
    "#a700ff",  # 8.0-10.0
    "#dddddd",  # 10.0-12.0
    "#bdbdbd",  # 12.0-14.0
    "#8fb9eb",  # 14.0-16.0
    "#5e97cf",  # 16.0-18.0
    "#356ba8",  # 18.0+
]

CMAP = ListedColormap(COLORS)
NORM = BoundaryNorm(LEVELS, CMAP.N, clip=True)


@dataclass
class TimeWindow:
    start: datetime
    end: datetime


def infer_default_end(now_utc: datetime | None = None) -> datetime:
    now_utc = now_utc or datetime.now(timezone.utc)
    today_12z = now_utc.replace(hour=12, minute=0, second=0, microsecond=0)
    # Wait until 14:30 UTC so late morning daily gauges have a chance to land.
    if now_utc >= today_12z + timedelta(hours=2, minutes=30):
        return today_12z
    return today_12z - timedelta(days=1)


def parse_end_arg() -> datetime:
    if len(sys.argv) > 1:
        return datetime.strptime(sys.argv[1], "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    return infer_default_end()


def build_time_window(end_dt: datetime) -> TimeWindow:
    return TimeWindow(start=end_dt - timedelta(hours=24), end=end_dt)


def ymdhm(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%d%H%M")


def download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def unzip(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    shp_files = list(out_dir.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No .shp found in {zip_path}")
    return shp_files[0]


def load_shapes() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    cwa_zip = download(CWA_URL, RAW_DIR / "w_16ap26.zip")
    county_zip = download(COUNTY_URL, RAW_DIR / "c_16ap26.zip")

    cwa_shp = unzip(cwa_zip, SHAPE_DIR / "cwa")
    county_shp = unzip(county_zip, SHAPE_DIR / "county")

    cwa = gpd.read_file(cwa_shp)
    counties = gpd.read_file(county_shp)
    return cwa, counties


def fetch_synoptic_precip(window: TimeWindow) -> pd.DataFrame:
    params = {
        "token": SYNOPTIC_TOKEN,
        "bbox": ",".join(str(x) for x in QUERY_BBOX),
        "start": ymdhm(window.start),
        "end": ymdhm(window.end),
        "pmode": "totals",
        "search": "nearest",
        "window": 60,
        "units": "english",
        "complete": 1,
        "status": "active",
        "showemptystations": 0,
    }
    r = requests.get(SYNOPTIC_URL, params=params, timeout=180)
    r.raise_for_status()
    payload = r.json()

    summary = payload.get("SUMMARY", {})
    code = summary.get("RESPONSE_CODE")
    if code not in (1, "1"):
        raise RuntimeError(f"Synoptic API returned code {code}: {summary.get('RESPONSE_MESSAGE')}")

    rows: list[dict] = []
    for station in payload.get("STATION", []):
        precip_list = station.get("OBSERVATIONS", {}).get("precipitation", [])
        if not precip_list:
            continue
        total = precip_list[0].get("total")
        if total is None:
            continue
        try:
            lat = float(station["LATITUDE"])
            lon = float(station["LONGITUDE"])
            total = float(total)
        except Exception:
            continue
        rows.append(
            {
                "stid": station.get("STID"),
                "name": station.get("NAME"),
                "state": station.get("STATE"),
                "lat": lat,
                "lon": lon,
                "mnet_id": station.get("MNET_ID"),
                "timezone": station.get("TIMEZONE"),
                "precip_in": max(total, 0.0),
                "first_report": precip_list[0].get("first_report"),
                "last_report": precip_list[0].get("last_report"),
                "report_type": precip_list[0].get("report_type"),
                "count": precip_list[0].get("count"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No station precipitation rows returned from Synoptic.")

    df = df.drop_duplicates(subset=["stid"]).copy()
    df = df[(df["lon"] >= QUERY_BBOX[0]) & (df["lon"] <= QUERY_BBOX[2]) & (df["lat"] >= QUERY_BBOX[1]) & (df["lat"] <= QUERY_BBOX[3])]
    return df.sort_values(["state", "stid"]).reset_index(drop=True)


def prepare_geodata(df: pd.DataFrame, cwa: gpd.GeoDataFrame, counties: gpd.GeoDataFrame):
    lix = cwa[cwa["CWA"] == "LIX"].copy()
    if lix.empty:
        raise RuntimeError("Could not find CWA=LIX in CWA shapefile.")

    counties = counties.to_crs(4326)
    lix = lix.to_crs(4326)
    plot_bounds = box(*PLOT_BBOX)

    counties = counties[counties.intersects(plot_bounds)].copy()
    counties["in_lix"] = counties["CWA"].astype(str).str[:3].eq("LIX")

    stations_gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs=4326,
    )

    # Project for nicer interpolation/plotting math.
    target_crs = 3857
    return (
        lix.to_crs(target_crs),
        counties.to_crs(target_crs),
        stations_gdf.to_crs(target_crs),
    )


def idw_grid(x: np.ndarray, y: np.ndarray, z: np.ndarray, gx: np.ndarray, gy: np.ndarray, power: float = 2.0) -> np.ndarray:
    # gx, gy are 2D meshgrids
    xi = gx[..., None]
    yi = gy[..., None]
    dist2 = (xi - x) ** 2 + (yi - y) ** 2
    # Avoid divide by zero.
    exact = dist2 == 0
    dist2[exact] = 1e-12
    w = 1.0 / np.power(dist2, power / 2.0)
    vals = np.sum(w * z, axis=2) / np.sum(w, axis=2)
    if exact.any():
        row_idx, col_idx, pt_idx = np.where(exact)
        vals[row_idx, col_idx] = z[pt_idx]
    return vals


def format_station_value(v: float) -> str:
    if v < 0.01:
        return "0"
    if v < 0.1:
        return f"{v:.2f}".lstrip("0")
    if v < 1:
        return f"{v:.2f}".rstrip("0").rstrip("0")
    return f"{v:.1f}".rstrip("0").rstrip(".")


def plot_map(window: TimeWindow, lix: gpd.GeoDataFrame, counties: gpd.GeoDataFrame, stations: gpd.GeoDataFrame) -> None:
    county_union = lix.geometry.union_all()
    minx, miny, maxx, maxy = counties.total_bounds

    # Interpolate from all stations in the query domain, then clip to LIX CWA.
    x = stations.geometry.x.to_numpy()
    y = stations.geometry.y.to_numpy()
    z = stations["precip_in"].to_numpy()

    nx, ny = 550, 420
    gx, gy = np.meshgrid(np.linspace(minx, maxx, nx), np.linspace(miny, maxy, ny))
    grid = idw_grid(x, y, z, gx, gy, power=2.0)
    grid = gaussian_filter(grid, sigma=1.0)
    grid = np.clip(grid, 0, None)

    mask = contains_xy(county_union, gx, gy)
    grid = np.where(mask, grid, np.nan)

    fig = plt.figure(figsize=(15.5, 11.5), facecolor="#f2f2f2")

    # Header block
    ax_head = fig.add_axes([0.03, 0.83, 0.94, 0.14])
    ax_head.set_facecolor("white")
    for s in ax_head.spines.values():
        s.set_linewidth(1.8)
        s.set_color("black")
    ax_head.set_xticks([])
    ax_head.set_yticks([])
    ax_head.text(0.5, 0.82, SUBTITLE, ha="center", va="center", fontsize=30, fontweight="bold")
    ax_head.text(0.5, 0.60, PREPARED_BY, ha="center", va="center", fontsize=17, fontweight="bold")
    ax_head.text(0.5, 0.30, DESCRIPTION, ha="center", va="center", fontsize=13)

    # Main map
    ax = fig.add_axes([0.03, 0.07, 0.82, 0.74])
    ax.set_facecolor("white")
    for s in ax.spines.values():
        s.set_linewidth(1.8)
        s.set_color("black")

    counties.plot(ax=ax, facecolor="#ffffff", edgecolor="#8a8a8a", linewidth=0.6, zorder=1)
    counties[counties["in_lix"]].plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.8, zorder=2)

    ax.contourf(gx, gy, grid, levels=LEVELS, cmap=CMAP, norm=NORM, extend="max", zorder=0)

    lix.boundary.plot(ax=ax, color="black", linewidth=2.2, zorder=3)

    # Station labels inside the LIX CWA only.
    in_lix = stations[stations.within(county_union.buffer(1))].copy()
    for row in in_lix.itertuples():
        val = float(row.precip_in)
        if val < 0.01:
            color = "#666666"
            size = 7
        elif val < 1.0:
            color = "#0b57d0"
            size = 8
        elif val < 3.0:
            color = "#1f7a00"
            size = 8
        else:
            color = "#c1121f"
            size = 8
        txt = ax.text(
            row.geometry.x,
            row.geometry.y,
            format_station_value(val),
            ha="center",
            va="center",
            fontsize=size,
            color=color,
            zorder=4,
        )
        txt.set_path_effects([pe.withStroke(linewidth=1.5, foreground="white")])

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"{TITLE_OFFICE}\n24 Hour Precipitation Ending {window.end.strftime('%HZ %m-%d-%Y')}",
        fontsize=22,
        fontweight="bold",
        pad=16,
    )

    # Legend / info panel
    ax_leg = fig.add_axes([0.86, 0.07, 0.11, 0.74])
    ax_leg.set_facecolor("white")
    for s in ax_leg.spines.values():
        s.set_linewidth(1.8)
        s.set_color("black")
    ax_leg.set_xticks([])
    ax_leg.set_yticks([])

    ax_leg.text(0.5, 0.96, "1 day (24hr)\nrainfall ending:", ha="center", va="top", fontsize=15, fontweight="bold")
    ax_leg.text(0.5, 0.88, window.end.strftime("%Y/%m/%d\n%H00 UTC"), ha="center", va="top", fontsize=16)
    ax_leg.text(0.5, 0.77, "Rainfall\n(Inches)", ha="center", va="top", fontsize=16, fontweight="bold")

    labels = [
        "Less than 0.01", "0.01 to 0.1", "0.1 to 0.25", "0.25 to 0.5", "0.5 to 0.75",
        "0.75 to 1", "1 to 1.5", "1.5 to 2", "2 to 2.5", "2.5 to 3", "3 to 4", "4 to 5",
        "5 to 6", "6 to 8", "8 to 10", "10 to 12", "12 to 14", "14 to 16", "16 to 18", "18 or higher"
    ]
    y0 = 0.70
    dy = 0.028
    for i, (label, color) in enumerate(zip(labels[::-1], COLORS[::-1])):
        y = y0 - i * dy
        ax_leg.add_patch(plt.Rectangle((0.12, y - 0.012), 0.18, 0.018, color=color, transform=ax_leg.transAxes, clip_on=False))
        ax_leg.text(0.35, y - 0.002, label, fontsize=9.5, va="center", ha="left")

    ax_leg.text(
        0.5,
        0.05,
        "Sources: Synoptic derived precip + NWS GIS basemaps\nInterpolation: inverse-distance weighted",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#444444",
    )

    png_path = OUT_DIR / "lix_24h_precip_latest.png"
    fig.savefig(png_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_outputs(window: TimeWindow, df: pd.DataFrame) -> None:
    csv_path = OUT_DIR / "station_precip_latest.csv"
    json_path = OUT_DIR / "latest.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "start_utc": window.start.isoformat(),
                "end_utc": window.end.isoformat(),
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "station_count": int(len(df)),
                "image": "lix_24h_precip_latest.png",
                "csv": "station_precip_latest.csv",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    end_dt = parse_end_arg()
    window = build_time_window(end_dt)
    cwa, counties = load_shapes()
    df = fetch_synoptic_precip(window)
    lix, counties_p, stations_p = prepare_geodata(df, cwa, counties)
    plot_map(window, lix, counties_p, stations_p)
    write_outputs(window, df)
    print(f"Saved map to {OUT_DIR / 'lix_24h_precip_latest.png'}")
    print(f"Stations used: {len(df)}")


if __name__ == "__main__":
    main()

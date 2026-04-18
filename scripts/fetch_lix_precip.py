#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
from matplotlib.colors import BoundaryNorm, ListedColormap
from rasterio.mask import mask
from shapely.geometry import box

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

# Rough plotting extent around LIX CWA.
PLOT_BBOX = (-91.9, 28.8, -87.6, 31.8)

CWA_URL = "https://www.weather.gov/source/gis/Shapefiles/WSOM/w_16ap26.zip"
COUNTY_URL = "https://www.weather.gov/source/gis/Shapefiles/County/c_16ap26.zip"

TITLE_OFFICE = "National Weather Service New Orleans/Baton Rouge Louisiana"
SUBTITLE = "Estimated 24 Hour Rainfall"
PREPARED_BY = "Graphic prepared by: WFO New Orleans/Baton Rouge"
DESCRIPTION = (
    "Liquid precipitation observed over the specified 24 hour period. "
    "Rainfall shading is from official NWPS / RFC Stage IV multi-sensor QPE."
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
    if now_utc >= today_12z + timedelta(hours=2, minutes=30):
        return today_12z
    return today_12z - timedelta(days=1)


def parse_end_arg() -> datetime:
    if len(sys.argv) > 1:
        return datetime.strptime(sys.argv[1], "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    return infer_default_end()


def build_time_window(end_dt: datetime) -> TimeWindow:
    return TimeWindow(start=end_dt - timedelta(hours=24), end=end_dt)


def download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    r = requests.get(url, timeout=180)
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


def stageiv_tif_url(end_dt: datetime) -> str:
    y = end_dt.strftime("%Y")
    m = end_dt.strftime("%m")
    d = end_dt.strftime("%d")
    ymd = end_dt.strftime("%Y%m%d")
    return f"https://water.noaa.gov/resources/downloads/precip/stageIV/{y}/{m}/{d}/nws_precip_1day_{ymd}_conus.tif"


def fetch_stageiv_qpe(window: TimeWindow) -> Path:
    tif_url = stageiv_tif_url(window.end)
    tif_path = RAW_DIR / f"nws_precip_1day_{window.end.strftime('%Y%m%d')}_conus.tif"
    return download(tif_url, tif_path)


def prepare_geodata(cwa: gpd.GeoDataFrame, counties: gpd.GeoDataFrame):
    lix = cwa[cwa["CWA"] == "LIX"].copy()
    if lix.empty:
        raise RuntimeError("Could not find CWA=LIX in CWA shapefile.")

    counties = counties.to_crs(4326)
    lix = lix.to_crs(4326)

    plot_bounds = box(*PLOT_BBOX)
    counties = counties[counties.intersects(plot_bounds)].copy()
    counties["in_lix"] = counties["CWA"].astype(str).str[:3].eq("LIX")

    target_crs = 3857
    return (
        lix.to_crs(target_crs),
        counties.to_crs(target_crs),
    )


def read_and_clip_raster(tif_path: Path, lix_3857: gpd.GeoDataFrame):
    lix_4326 = lix_3857.to_crs(4326)

    with rasterio.open(tif_path) as src:
        clipped_data, clipped_transform = mask(
            src,
            lix_4326.geometry,
            crop=True,
            filled=True,
            nodata=np.nan,
        )
        arr = clipped_data[0].astype("float32")

        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)

        bounds = rasterio.transform.array_bounds(arr.shape[0], arr.shape[1], clipped_transform)

        left, bottom, right, top = bounds

        # Build corner box in raster CRS, then project to EPSG:3857 for plotting with counties/CWA.
        raster_bounds_gdf = gpd.GeoDataFrame(
            geometry=[box(left, bottom, right, top)],
            crs=src.crs,
        ).to_crs(3857)

        rb = raster_bounds_gdf.total_bounds
        extent_3857 = [rb[0], rb[2], rb[1], rb[3]]

    # NWPS Stage IV CONUS TIFFs are already in inches.
    arr = np.where(arr < 0, np.nan, arr)
    return arr, extent_3857


def plot_map(window: TimeWindow, lix: gpd.GeoDataFrame, counties: gpd.GeoDataFrame, raster_arr: np.ndarray, raster_extent_3857: list[float]) -> None:
    minx, miny, maxx, maxy = counties.total_bounds

    fig = plt.figure(figsize=(15.5, 11.5), facecolor="#f2f2f2")

    # -----------------------------
    # Header block
    # -----------------------------
    ax_head = fig.add_axes([0.03, 0.84, 0.94, 0.13])
    ax_head.set_facecolor("white")
    for s in ax_head.spines.values():
        s.set_linewidth(1.8)
        s.set_color("black")
    ax_head.set_xticks([])
    ax_head.set_yticks([])

    ax_head.text(0.5, 0.78, SUBTITLE, ha="center", va="center", fontsize=24, fontweight="bold")
    ax_head.text(0.5, 0.56, PREPARED_BY, ha="center", va="center", fontsize=15, fontweight="bold")
    ax_head.text(0.5, 0.34, TITLE_OFFICE, ha="center", va="center", fontsize=17, fontweight="bold")
    ax_head.text(
        0.5,
        0.14,
        f"24 Hour Precipitation Ending {window.end.strftime('%HZ %m-%d-%Y')}",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )

    # -----------------------------
    # Main map
    # -----------------------------
    ax = fig.add_axes([0.03, 0.07, 0.82, 0.75])
    ax.set_facecolor("white")
    for s in ax.spines.values():
        s.set_linewidth(1.8)
        s.set_color("black")

    counties.plot(ax=ax, facecolor="#ffffff", edgecolor="#b7b7b7", linewidth=0.6, zorder=1)

    ax.imshow(
        raster_arr,
        extent=raster_extent_3857,
        origin="upper",
        cmap=CMAP,
        norm=NORM,
        interpolation="nearest",
        zorder=0,
    )

    counties[counties["in_lix"]].plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.6, zorder=2)
    lix.boundary.plot(ax=ax, color="black", linewidth=2.2, zorder=3)

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_xticks([])
    ax.set_yticks([])

    # -----------------------------
    # Legend / info panel
    # -----------------------------
    ax_leg = fig.add_axes([0.86, 0.07, 0.11, 0.75])
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
        ax_leg.add_patch(
            plt.Rectangle((0.12, y - 0.012), 0.18, 0.018, color=color, transform=ax_leg.transAxes, clip_on=False)
        )
        ax_leg.text(0.35, y - 0.002, label, fontsize=9.5, va="center", ha="left")

    ax_leg.text(
        0.5,
        0.05,
        "Sources: NWPS/RFC Stage IV QPE + NWS GIS basemaps",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#444444",
    )

    png_path = OUT_DIR / "lix_24h_precip_latest.png"
    fig.savefig(png_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_outputs(window: TimeWindow, tif_path: Path) -> None:
    json_path = OUT_DIR / "latest.json"

    json_path.write_text(
        json.dumps(
            {
                "start_utc": window.start.isoformat(),
                "end_utc": window.end.isoformat(),
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "image": "lix_24h_precip_latest.png",
                "source_tif": tif_path.name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    end_dt = parse_end_arg()
    window = build_time_window(end_dt)

    cwa, counties = load_shapes()
    lix, counties_p = prepare_geodata(cwa, counties)

    tif_path = fetch_stageiv_qpe(window)
    raster_arr, raster_extent_3857 = read_and_clip_raster(tif_path, lix)

    plot_map(window, lix, counties_p, raster_arr, raster_extent_3857)
    write_outputs(window, tif_path)

    print(f"Saved map to {OUT_DIR / 'lix_24h_precip_latest.png'}")
    print(f"Used source raster: {tif_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import requests
from matplotlib.colors import BoundaryNorm, ListedColormap
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, bounds as window_bounds
from rasterio.windows import from_bounds
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

# Zoomed-in plot extent centered on LIX (LA/MS)
PLOT_BBOX = (-92.5, 28.5, -88.0, 32.0)

CWA_URL = "https://www.weather.gov/source/gis/Shapefiles/WSOM/w_16ap26.zip"
COUNTY_URL = "https://www.weather.gov/source/gis/Shapefiles/County/c_16ap26.zip"
STATE_URL = "https://www.weather.gov/source/gis/Shapefiles/County/s_16ap26.zip"

TITLE_OFFICE = "National Weather Service New Orleans/Baton Rouge Louisiana"
SUBTITLE = "Estimated 24 Hour Rainfall"
PREPARED_BY = "Graphic prepared by: WFO New Orleans/Baton Rouge"
DESCRIPTION = (
    "Liquid precipitation observed over the specified 24 hour period. "
    "Rainfall shading is from official NWPS / RFC Stage IV multi-sensor QPE."
)

# NWPS-like bins, but extended to 15+ at the top.
# Values below 0.01 are drawn transparent.
LEVELS = [
    0.01, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 2.50,
    3.00, 4.00, 5.00, 6.00, 8.00, 10.00, 15.00, 30.00
]

COLORS = [
    "#67c6e5",  # 0.01 to 0.1
    "#6e9ad0",  # 0.1 to 0.25
    "#4b4aa7",  # 0.25 to 0.5
    "#57ea58",  # 0.5 to 0.75
    "#52b852",  # 0.75 to 1
    "#4d8c50",  # 1 to 1.5
    "#eceb59",  # 1.5 to 2
    "#efd27a",  # 2 to 2.5
    "#eda24f",  # 2.5 to 3
    "#ff4b4b",  # 3 to 4
    "#c7484d",  # 4 to 5
    "#a44a50",  # 5 to 6
    "#e43ee0",  # 6 to 8
    "#9362d6",  # 8 to 10
    "#d9d9d9",  # 10 to 15
    "#bcbcbc",  # >= 15
]

CMAP = ListedColormap(COLORS)
CMAP.set_under((1, 1, 1, 0))   # <0.01 transparent
CMAP.set_bad("#8f8f8f")         # missing data gray
NORM = BoundaryNorm(LEVELS, CMAP.N, clip=False)


@dataclass
class TimeWindow:
    start: datetime
    end: datetime


def infer_default_end(now_utc: datetime | None = None) -> datetime:
    now_utc = now_utc or datetime.now(timezone.utc)
    today_12z = now_utc.replace(hour=12, minute=0, second=0, microsecond=0)

    # Wait until 14:30 UTC so latest daily product has time to show up.
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
    print(f"Downloading: {url}")
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def download_if_missing(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    print(f"Downloading: {url}")
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


def load_shapes() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    cwa_zip = download_if_missing(CWA_URL, RAW_DIR / "w_16ap26.zip")
    county_zip = download_if_missing(COUNTY_URL, RAW_DIR / "c_16ap26.zip")
    state_zip = download_if_missing(STATE_URL, RAW_DIR / "s_16ap26.zip")

    cwa_shp = unzip(cwa_zip, SHAPE_DIR / "cwa")
    county_shp = unzip(county_zip, SHAPE_DIR / "county")
    state_shp = unzip(state_zip, SHAPE_DIR / "state")

    cwa = gpd.read_file(cwa_shp)
    counties = gpd.read_file(county_shp)
    states = gpd.read_file(state_shp)
    
    return cwa, counties, states


def stageiv_current_url() -> str:
    return "https://water.noaa.gov/resources/downloads/precip/stageIV/current/nws_precip_last24hours_conus.tif"


def stageiv_archive_url(end_dt: datetime) -> str:
    y = end_dt.strftime("%Y")
    m = end_dt.strftime("%m")
    d = end_dt.strftime("%d")
    ymd = end_dt.strftime("%Y%m%d")
    return f"https://water.noaa.gov/resources/downloads/precip/stageIV/{y}/{m}/{d}/nws_precip_1day_{ymd}_conus.tif"


def fetch_stageiv_qpe(window: TimeWindow) -> Path:
    default_end = infer_default_end()
    is_default_cycle = window.end == default_end

    if is_default_cycle:
        tif_url = stageiv_current_url()
        tif_path = RAW_DIR / "nws_precip_last24hours_conus.tif"
    else:
        tif_url = stageiv_archive_url(window.end)
        tif_path = RAW_DIR / f"nws_precip_1day_{window.end.strftime('%Y%m%d')}_conus.tif"

    path = download(tif_url, tif_path)
    print(f"Used source raster: {path}")
    return path


def prepare_geodata(cwa: gpd.GeoDataFrame, counties: gpd.GeoDataFrame, states: gpd.GeoDataFrame):
    lix = cwa[cwa["CWA"] == "LIX"].copy()
    if lix.empty:
        raise RuntimeError("Could not find CWA=LIX in CWA shapefile.")

    counties = counties.to_crs(4326)
    states = states.to_crs(4326)
    lix = lix.to_crs(4326)

    plot_bounds = box(*PLOT_BBOX)
    counties = counties[counties.intersects(plot_bounds)].copy()
    states = states[states.intersects(plot_bounds)].copy()
    counties["in_lix"] = counties["CWA"].astype(str).str[:3].eq("LIX")

    plot_domain = gpd.GeoDataFrame(geometry=[plot_bounds], crs=4326)

    target_crs = 3857
    return (
        lix.to_crs(target_crs),
        counties.to_crs(target_crs),
        states.to_crs(target_crs),
        plot_domain.to_crs(target_crs),
    )


def read_raster_for_plotting(tif_path: Path):
    target_crs = "EPSG:3857"

    plot_box_4326 = gpd.GeoDataFrame(geometry=[box(*PLOT_BBOX)], crs=4326)
    plot_box_3857 = plot_box_4326.to_crs(target_crs)
    xmin, ymin, xmax, ymax = plot_box_3857.total_bounds

    with rasterio.open(tif_path) as src:
        print(f"Raster CRS: {src.crs}")
        print(f"Raster bounds: {src.bounds}")

        with WarpedVRT(src, crs=target_crs, resampling=Resampling.nearest) as vrt:
            requested_window = from_bounds(xmin, ymin, xmax, ymax, transform=vrt.transform)
            requested_window = requested_window.round_offsets().round_lengths()

            full_window = Window(0, 0, vrt.width, vrt.height)
            window = requested_window.intersection(full_window)

            arr = vrt.read(1, window=window).astype(np.float32)

            nodata_value = vrt.nodata
            if nodata_value is not None and not np.isnan(nodata_value):
                arr = np.where(arr == nodata_value, np.nan, arr)

            arr = np.where(arr < 0, np.nan, arr)

            left, bottom, right, top = window_bounds(window, vrt.transform)

    valid = arr[np.isfinite(arr)]
    if valid.size:
        print(f"Cropped shape: {arr.shape}")
        print(f"Crop extent 3857: left={left}, right={right}, bottom={bottom}, top={top}")
        print(f"Crop min/max: min={valid.min()}, max={valid.max()}")
    else:
        print(f"Cropped shape: {arr.shape}")
        print("Crop contains no finite precip values.")

    extent_3857 = [left, right, bottom, top]
    print(f"Final extent_3857: {extent_3857}")

    return arr, extent_3857


def plot_map(
    window: TimeWindow,
    lix: gpd.GeoDataFrame,
    counties: gpd.GeoDataFrame,
    states: gpd.GeoDataFrame,
    plot_domain: gpd.GeoDataFrame,
    raster_arr: np.ndarray,
    raster_extent_3857: list[float],
) -> None:
    minx, miny, maxx, maxy = plot_domain.total_bounds

    fig = plt.figure(figsize=(16, 11.5), facecolor="#f2f2f2")

    # Header block
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

    # Main map
    ax = fig.add_axes([0.03, 0.07, 0.82, 0.75])
    ax.set_facecolor("white")
    for s in ax.spines.values():
        s.set_linewidth(1.8)
        s.set_color("black")

    ax.imshow(
        raster_arr,
        extent=raster_extent_3857,
        origin="upper",
        cmap=CMAP,
        norm=NORM,
        interpolation="nearest",
        zorder=0,
    )

    # Fade everything outside LIX, but keep raster visible underneath.
    plot_box_geom = plot_domain.geometry.iloc[0]
    lix_union = lix.geometry.union_all()
    outside_geom = plot_box_geom.difference(lix_union)
    outside_gdf = gpd.GeoDataFrame(geometry=[outside_geom], crs=plot_domain.crs)
    outside_gdf.plot(ax=ax, facecolor="white", edgecolor="none", alpha=0.50, zorder=1)

    # County outlines over the top
    counties.plot(ax=ax, facecolor="none", edgecolor="#b7b7b7", linewidth=0.55, zorder=2)
    
    # State outlines
    states.plot(ax=ax, facecolor="none", edgecolor="#555555", linewidth=1.5, zorder=3)

    # LIX boundary bold
    lix.boundary.plot(ax=ax, color="black", linewidth=2.5, zorder=4)

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_xticks([])
    ax.set_yticks([])

    # Legend panel
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
        "Greater than or equal to 15",
        "10 to 15",
        "8 to 10",
        "6 to 8",
        "5 to 6",
        "4 to 5",
        "3 to 4",
        "2.5 to 3",
        "2 to 2.5",
        "1.5 to 2",
        "1 to 1.5",
        "0.75 to 1",
        "0.5 to 0.75",
        "0.25 to 0.5",
        "0.1 to 0.25",
        "0.01 to 0.1",
    ]

    y0 = 0.70
    dy = 0.043
    for i, (label, color) in enumerate(zip(labels, COLORS[::-1])):
        y = y0 - i * dy
        ax_leg.add_patch(
            plt.Rectangle((0.10, y - 0.016), 0.20, 0.026, color=color, transform=ax_leg.transAxes, clip_on=False)
        )
        ax_leg.text(0.36, y - 0.003, label, fontsize=10, va="center", ha="left")

    # Missing data patch
    y = y0 - len(labels) * dy
    ax_leg.add_patch(
        plt.Rectangle((0.10, y - 0.016), 0.20, 0.026, color="#8f8f8f", transform=ax_leg.transAxes, clip_on=False)
    )
    ax_leg.text(0.36, y - 0.003, "Missing data", fontsize=10, va="center", ha="left")

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

    cwa, counties, states = load_shapes()
    lix, counties_p, states_p, plot_domain = prepare_geodata(cwa, counties, states)

    tif_path = fetch_stageiv_qpe(window)
    raster_arr, raster_extent_3857 = read_raster_for_plotting(tif_path)

    plot_map(window, lix, counties_p, states_p, plot_domain, raster_arr, raster_extent_3857)
    write_outputs(window, tif_path)

    print(f"Saved map to {OUT_DIR / 'lix_24h_precip_latest.png'}")


if __name__ == "__main__":
    main()

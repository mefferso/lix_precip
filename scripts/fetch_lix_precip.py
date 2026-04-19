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
import matplotlib.patheffects as pe
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
PLOT_BBOX = (-92.5, 28.5, -88.0, 32.0) [cite: 19]

# Wider extent for the index (locator) map
INDEX_BBOX = (-95.0, 28.5, -84.0, 35.0) [cite: 19, 20]

CWA_URL = "https://www.weather.gov/source/gis/Shapefiles/WSOM/w_16ap26.zip" [cite: 20]
COUNTY_URL = "https://www.weather.gov/source/gis/Shapefiles/County/c_16ap26.zip" [cite: 20]
STATE_URL = "https://www.weather.gov/source/gis/Shapefiles/County/s_16ap26.zip" [cite: 20]
TITLE_OFFICE = "National Weather Service New Orleans/Baton Rouge Louisiana" [cite: 20]
SUBTITLE = "Estimated Rainfall"

DESCRIPTION = (
    "Liquid precipitation observed over the specified period.\n"
    "Rainfall shading is from official NWPS / RFC Stage IV multi-sensor QPE."
)

CITIES = [
    {"name": "Baton Rouge", "lat": 30.4515, "lon": -91.1871},
    {"name": "New Orleans", "lat": 29.9511, "lon": -90.0715},
    {"name": "Gulfport",    "lat": 30.3674, "lon": -89.0928},
    {"name": "McComb",      "lat": 31.2438, "lon": -90.4532},
    {"name": "Houma",       "lat": 29.5958, "lon": -90.7195},
] [cite: 21]

# Mapping keys to NOAA's exact filename strings and readable titles
PERIODS = {
    "24h": {"noaa_str": "1day", "title": "24 Hour"},
    "2d": {"noaa_str": "2day", "title": "2-Day"},
    "3d": {"noaa_str": "3day", "title": "3-Day"},
    "4d": {"noaa_str": "4day", "title": "4-Day"},
    "5d": {"noaa_str": "5day", "title": "5-Day"},
    "6d": {"noaa_str": "6day", "title": "6-Day"},
    "7d": {"noaa_str": "7day", "title": "7-Day"},
    "10d": {"noaa_str": "10day", "title": "10-Day"},
    "14d": {"noaa_str": "14day", "title": "14-Day"},
    "30d": {"noaa_str": "30day", "title": "30-Day"},
    "mtd": {"noaa_str": "mtd", "title": "Month-to-Date"},
    "ytd": {"noaa_str": "ytd", "title": "Year-to-Date"},
}

# NWPS-like bins, but extended to 15+ at the top.
# Values below 0.01 are drawn transparent.
LEVELS = [
    0.01, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 2.50,
    3.00, 4.00, 5.00, 6.00, 8.00, 10.00, 15.00, 30.00
] [cite: 21, 22]

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
] [cite: 22, 23]

CMAP = ListedColormap(COLORS) [cite: 23]
CMAP.set_under((1, 1, 1, 0))   # <0.01 transparent [cite: 23]
CMAP.set_bad("#8f8f8f")         # missing data gray [cite: 23]
NORM = BoundaryNorm(LEVELS, CMAP.N, clip=False) [cite: 23]


@dataclass
class TimeWindow:
    start: datetime [cite: 23, 24]
    end: datetime [cite: 24]


def infer_default_end(now_utc: datetime | None = None) -> datetime:
    now_utc = now_utc or datetime.now(timezone.utc) [cite: 24]
    today_12z = now_utc.replace(hour=12, minute=0, second=0, microsecond=0) [cite: 24]

    # Wait until 14:30 UTC so latest daily product has time to show up.
    if now_utc >= today_12z + timedelta(hours=2, minutes=30): [cite: 24, 25]
        return today_12z [cite: 25]
    return today_12z - timedelta(days=1) [cite: 25]


def parse_end_arg() -> datetime:
    if len(sys.argv) > 1: [cite: 25]
        return datetime.strptime(sys.argv[1], "%Y%m%d%H%M").replace(tzinfo=timezone.utc) [cite: 25]
    return infer_default_end() [cite: 25]


def build_time_window(end_dt: datetime) -> TimeWindow:
    return TimeWindow(start=end_dt - timedelta(hours=24), end=end_dt) [cite: 25]


def download_if_missing(url: str, dest: Path) -> Path:
    if dest.exists(): [cite: 25, 26]
        return dest [cite: 26]
    print(f"Downloading: {url}") [cite: 26]
    r = requests.get(url, timeout=180) [cite: 26]
    r.raise_for_status() [cite: 26]
    dest.write_bytes(r.content) [cite: 26]
    return dest [cite: 26]


def unzip(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True) [cite: 26]
    with zipfile.ZipFile(zip_path) as zf: [cite: 26]
        zf.extractall(out_dir) [cite: 26]

    shp_files = list(out_dir.glob("*.shp")) [cite: 26]
    if not shp_files: [cite: 26]
        raise FileNotFoundError(f"No .shp found in {zip_path}") [cite: 26]
    return shp_files[0] [cite: 26]


def load_shapes() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    cwa_zip = download_if_missing(CWA_URL, RAW_DIR / "w_16ap26.zip") [cite: 26, 27]
    county_zip = download_if_missing(COUNTY_URL, RAW_DIR / "c_16ap26.zip") [cite: 27]
    state_zip = download_if_missing(STATE_URL, RAW_DIR / "s_16ap26.zip") [cite: 27]

    cwa_shp = unzip(cwa_zip, SHAPE_DIR / "cwa") [cite: 27]
    county_shp = unzip(county_zip, SHAPE_DIR / "county") [cite: 27]
    state_shp = unzip(state_zip, SHAPE_DIR / "state") [cite: 27]

    cwa = gpd.read_file(cwa_shp) [cite: 27]
    counties = gpd.read_file(county_shp) [cite: 27]
    states = gpd.read_file(state_shp) [cite: 27]
    
    return cwa, counties, states [cite: 27]


def stageiv_archive_url(end_dt: datetime, noaa_str: str) -> str:
    y = end_dt.strftime("%Y") [cite: 27]
    m = end_dt.strftime("%m") [cite: 28]
    d = end_dt.strftime("%d") [cite: 28]
    ymd = end_dt.strftime("%Y%m%d") [cite: 28]
    return f"https://water.noaa.gov/resources/downloads/precip/stageIV/{y}/{m}/{d}/nws_precip_{noaa_str}_{ymd}_conus.tif"


def fetch_stageiv_qpe(window: TimeWindow, noaa_str: str) -> Path:
    # Always pull from the archive structure, even for the current day, 
    # to maintain consistent multi-day file naming conventions.
    tif_url = stageiv_archive_url(window.end, noaa_str)
    tif_path = RAW_DIR / f"nws_precip_{noaa_str}_{window.end.strftime('%Y%m%d')}_conus.tif"

    path = download_if_missing(tif_url, tif_path)
    print(f"Used source raster: {path}") [cite: 28, 29]
    return path


def prepare_geodata(cwa: gpd.GeoDataFrame, counties: gpd.GeoDataFrame, states: gpd.GeoDataFrame):
    lix = cwa[cwa["CWA"] == "LIX"].copy() [cite: 29]
    if lix.empty: [cite: 29]
        raise RuntimeError("Could not find CWA=LIX in CWA shapefile.") [cite: 29]

    counties = counties.to_crs(4326) [cite: 29]
    states = states.to_crs(4326) [cite: 29]
    lix = lix.to_crs(4326) [cite: 29]

    plot_bounds = box(*PLOT_BBOX) [cite: 29]
    index_bounds = box(*INDEX_BBOX) [cite: 29]
    
    counties = counties[counties.intersects(plot_bounds)].copy() [cite: 29]
    states = states[states.intersects(index_bounds)].copy() [cite: 29]
    counties["in_lix"] = counties["CWA"].astype(str).str[:3].eq("LIX") [cite: 29]

    plot_domain = gpd.GeoDataFrame(geometry=[plot_bounds], crs=4326) [cite: 29, 30]

    cities_gdf = gpd.GeoDataFrame(
        CITIES, [cite: 30]
        geometry=gpd.points_from_xy([c["lon"] for c in CITIES], [c["lat"] for c in CITIES]), [cite: 30]
        crs=4326 [cite: 30]
    )

    target_crs = 3857 [cite: 30]
    return (
        lix.to_crs(target_crs), [cite: 30]
        counties.to_crs(target_crs), [cite: 30]
        states.to_crs(target_crs), [cite: 30]
        plot_domain.to_crs(target_crs), [cite: 30]
        cities_gdf.to_crs(target_crs), [cite: 30]
    ) [cite: 30, 31]


def read_raster_for_plotting(tif_path: Path):
    target_crs = "EPSG:3857" [cite: 31]

    plot_box_4326 = gpd.GeoDataFrame(geometry=[box(*PLOT_BBOX)], crs=4326) [cite: 31]
    plot_box_3857 = plot_box_4326.to_crs(target_crs) [cite: 31]
    xmin, ymin, xmax, ymax = plot_box_3857.total_bounds [cite: 31]

    with rasterio.open(tif_path) as src: [cite: 31]
        with WarpedVRT(src, crs=target_crs, resampling=Resampling.nearest) as vrt: [cite: 31]
            requested_window = from_bounds(xmin, ymin, xmax, ymax, transform=vrt.transform) [cite: 31]
            requested_window = requested_window.round_offsets().round_lengths() [cite: 31, 32]

            full_window = Window(0, 0, vrt.width, vrt.height) [cite: 32]
            window = requested_window.intersection(full_window) [cite: 32]

            arr = vrt.read(1, window=window).astype(np.float32) [cite: 32]

            nodata_value = vrt.nodata [cite: 32]
            if nodata_value is not None and not np.isnan(nodata_value): [cite: 32]
                arr = np.where(arr == nodata_value, np.nan, arr) [cite: 32, 33]

            arr = np.where(arr < 0, np.nan, arr) [cite: 33]

            left, bottom, right, top = window_bounds(window, vrt.transform) [cite: 33]

    extent_3857 = [left, right, bottom, top] [cite: 34]
    return arr, extent_3857 [cite: 34]


def plot_map(
    window: TimeWindow,
    lix: gpd.GeoDataFrame,
    counties: gpd.GeoDataFrame,
    states: gpd.GeoDataFrame,
    plot_domain: gpd.GeoDataFrame,
    cities: gpd.GeoDataFrame,
    raster_arr: np.ndarray,
    raster_extent_3857: list[float],
    period_key: str,
    period_title: str
) -> None:
    minx, miny, maxx, maxy = plot_domain.total_bounds [cite: 34]

    fig = plt.figure(figsize=(16, 11.5), facecolor="#f2f2f2") [cite: 34]

    # Header block
    ax_head = fig.add_axes([0.03, 0.84, 0.94, 0.13]) [cite: 34, 35]
    ax_head.set_facecolor("white") [cite: 35]
    for s in ax_head.spines.values(): [cite: 35]
        s.set_linewidth(1.8) [cite: 35]
        s.set_color("black") [cite: 35]
    ax_head.set_xticks([]) [cite: 35]
    ax_head.set_yticks([]) [cite: 35]

    ax_head.text(0.5, 0.78, SUBTITLE, ha="center", va="center", fontsize=24, fontweight="bold") [cite: 35]
    ax_head.text(0.5, 0.60, "Data Source: water.noaa.gov", ha="center", va="center", fontsize=17, fontweight="bold") [cite: 35]
    ax_head.text(0.5, 0.34, TITLE_OFFICE, ha="center", va="center", fontsize=17, fontweight="bold") [cite: 35]
    ax_head.text(
        0.5, [cite: 35]
        0.14, [cite: 35]
        f"{period_title} Precipitation Ending {window.end.strftime('%HZ %m-%d-%Y')}",
        ha="center", [cite: 36]
        va="center", [cite: 36]
        fontsize=18, [cite: 36]
        fontweight="bold", [cite: 36]
    )

    # Main map
    ax = fig.add_axes([0.03, 0.07, 0.78, 0.75]) [cite: 36]
    ax.set_facecolor("white") [cite: 36]
    for s in ax.spines.values(): [cite: 36]
        s.set_linewidth(1.8) [cite: 36]
        s.set_color("black") [cite: 36]

    ax.imshow(
        raster_arr, [cite: 36]
        extent=raster_extent_3857, [cite: 36, 37]
        origin="upper", [cite: 37]
        cmap=CMAP, [cite: 37]
        norm=NORM, [cite: 37]
        interpolation="nearest", [cite: 37]
        zorder=0, [cite: 37]
    )

    # Fade everything outside LIX, but keep raster visible underneath.
    plot_box_geom = plot_domain.geometry.iloc[0] [cite: 37, 38]
    lix_union = lix.geometry.union_all() [cite: 38]
    outside_geom = plot_box_geom.difference(lix_union) [cite: 38]
    outside_gdf = gpd.GeoDataFrame(geometry=[outside_geom], crs=plot_domain.crs) [cite: 38]
    outside_gdf.plot(ax=ax, facecolor="white", edgecolor="none", alpha=0.50, zorder=1) [cite: 38]

    # County outlines over the top
    counties.plot(ax=ax, facecolor="none", edgecolor="#b7b7b7", linewidth=0.55, zorder=2) [cite: 38]
    
    # State outlines
    states.plot(ax=ax, facecolor="none", edgecolor="#555555", linewidth=1.5, zorder=3) [cite: 38]

    # LIX boundary bold
    lix.boundary.plot(ax=ax, color="black", linewidth=2.5, zorder=4) [cite: 38]

    # North Arrow 
    ax.text(
        0.04, 0.14, 'N', [cite: 38, 39]
        ha='center', va='center', fontsize=26, fontweight='bold', [cite: 39]
        transform=ax.transAxes, zorder=10,  [cite: 39]
        path_effects=[pe.withStroke(linewidth=4, foreground="white")] [cite: 39]
    )
    ax.annotate(
        '', xy=(0.04, 0.11), xytext=(0.04, 0.03), [cite: 39]
        arrowprops=dict(facecolor='black', edgecolor='white', width=8, headwidth=20, headlength=18), [cite: 39]
        xycoords='axes fraction', textcoords='axes fraction', zorder=10 [cite: 39]
    )

    # Inset Map (Locator Map)
    ax_in = ax.inset_axes([0.00, 0.788, 0.24, 0.24]) [cite: 39, 40]
    ax_in.set_facecolor("#d4e6f1") [cite: 40]
    for s in ax_in.spines.values(): [cite: 40]
        s.set_linewidth(1.5) [cite: 40]
        s.set_color("black") [cite: 40]
    
    states.plot(ax=ax_in, facecolor="#f0f0f0", edgecolor="#555555", linewidth=0.8, zorder=1) [cite: 40]
    lix.plot(ax=ax_in, facecolor="#ff9900", edgecolor="black", linewidth=1.2, zorder=2) [cite: 40]
    
    lbls = gpd.GeoDataFrame(
        {"name": ["LA", "MS", "AL", "TX", "AR"]}, [cite: 40, 41]
        geometry=gpd.points_from_xy([-92.2, -89.6, -86.8, -94.2, -92.5], [31.2, 32.8, 32.8, 31.5, 34.5]), [cite: 41]
        crs=4326 [cite: 41]
    ).to_crs(plot_domain.crs) [cite: 41]
    
    for _, row in lbls.iterrows(): [cite: 41]
        ax_in.text(
            row.geometry.x, row.geometry.y, row['name'],  [cite: 41]
            ha='center', va='center', fontsize=9, fontweight='bold', color='#444444', zorder=3 [cite: 41]
        ) [cite: 41, 42]
        
    index_domain = gpd.GeoDataFrame(geometry=[box(*INDEX_BBOX)], crs=4326).to_crs(plot_domain.crs) [cite: 42]
    inx_minx, inx_miny, inx_maxx, inx_maxy = index_domain.total_bounds [cite: 42]
    
    ax_in.set_xlim(inx_minx, inx_maxx) [cite: 42]
    ax_in.set_ylim(inx_miny, inx_maxy) [cite: 42]
    ax_in.set_xticks([]) [cite: 42]
    ax_in.set_yticks([]) [cite: 42]

    # City dots, sampling, and labels
    rx_min, rx_max, ry_min, ry_max = raster_extent_3857 [cite: 42]
    height, width = raster_arr.shape [cite: 42]

    for idx, row in cities.iterrows(): [cite: 42]
        x, y = row.geometry.x, row.geometry.y [cite: 42]
        val_str = "0.00" [cite: 42, 43]
        
        if rx_min <= x <= rx_max and ry_min <= y <= ry_max: [cite: 43]
            col_idx = int((x - rx_min) / (rx_max - rx_min) * width) [cite: 43, 44]
            row_idx = int((ry_max - y) / (ry_max - ry_min) * height) [cite: 44]
            
            col_idx = max(0, min(col_idx, width - 1)) [cite: 44]
            row_idx = max(0, min(row_idx, height - 1)) [cite: 44, 45]
            
            val = raster_arr[row_idx, col_idx] [cite: 45]
            
            if np.isfinite(val) and val > 0: [cite: 45]
                val_str = f"{val:.2f}" [cite: 45, 46]
                
        label_text = f"{row['name']}\n{val_str}\"" [cite: 46]
        
        ax.plot(x, y, 'o', color='white', markeredgecolor='black', markersize=5, zorder=5) [cite: 46]
        ax.text(
            x, y + 8000, label_text, [cite: 46]
            color='black', fontsize=11, fontweight='bold', ha='center', va='bottom', [cite: 46, 47]
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")], [cite: 47]
            zorder=6 [cite: 47]
        )

    ax.set_xlim(minx, maxx) [cite: 47]
    ax.set_ylim(miny, maxy) [cite: 47]
    ax.set_xticks([]) [cite: 47]
    ax.set_yticks([]) [cite: 47]

    # Legend panel
    ax_leg = fig.add_axes([0.82, 0.07, 0.15, 0.75]) [cite: 47]
    ax_leg.set_facecolor("white") [cite: 47]
    for s in ax_leg.spines.values(): [cite: 47]
        s.set_linewidth(1.8) [cite: 47, 48]
        s.set_color("black") [cite: 48]
    ax_leg.set_xticks([]) [cite: 48]
    ax_leg.set_yticks([]) [cite: 48]

    ax_leg.text(0.5, 0.96, f"{period_title}\nrainfall ending:", ha="center", va="top", fontsize=15, fontweight="bold")
    ax_leg.text(0.5, 0.88, window.end.strftime("%Y/%m/%d\n%H00 UTC"), ha="center", va="top", fontsize=16) [cite: 48]
    ax_leg.text(0.5, 0.77, "Rainfall\n(Inches)", ha="center", va="top", fontsize=16, fontweight="bold") [cite: 48]

    labels = [
        "Greater than 15", [cite: 48]
        "10 to 15", [cite: 48]
        "8 to 10", [cite: 48]
        "6 to 8", [cite: 48]
        "5 to 6", [cite: 48, 49]
        "4 to 5", [cite: 49]
        "3 to 4", [cite: 49]
        "2.5 to 3", [cite: 49]
        "2 to 2.5", [cite: 49]
        "1.5 to 2", [cite: 49]
        "1 to 1.5", [cite: 49]
        "0.75 to 1", [cite: 49]
        "0.5 to 0.75", [cite: 49]
        "0.25 to 0.5", [cite: 49]
        "0.1 to 0.25", [cite: 49, 50]
        "0.01 to 0.1", [cite: 50]
    ]

    y0 = 0.70 [cite: 50]
    dy = 0.043 [cite: 50]
    for i, (label, color) in enumerate(zip(labels, COLORS[::-1])): [cite: 50]
        y = y0 - i * dy [cite: 50]
        ax_leg.add_patch(
            plt.Rectangle((0.10, y - 0.016), 0.20, 0.026, color=color, transform=ax_leg.transAxes, clip_on=False) [cite: 50]
        ) [cite: 50]
        ax_leg.text(0.36, y - 0.003, label, fontsize=10, va="center", ha="left") [cite: 50, 51]

    # Missing data patch
    y = y0 - len(labels) * dy [cite: 51]
    ax_leg.add_patch(
        plt.Rectangle((0.10, y - 0.016), 0.20, 0.026, color="#8f8f8f", transform=ax_leg.transAxes, clip_on=False) [cite: 51]
    )
    ax_leg.text(0.36, y - 0.003, "Missing data", fontsize=10, va="center", ha="left") [cite: 51]

    png_path = OUT_DIR / f"lix_{period_key}_precip_latest.png"
    fig.savefig(png_path, dpi=170, bbox_inches="tight") [cite: 51]
    plt.close(fig) [cite: 51]


def write_outputs(window: TimeWindow, generated_maps: dict[str, dict[str, str]]) -> None:
    json_path = OUT_DIR / "latest.json" [cite: 51]

    data = {
        "start_utc": window.start.isoformat(),
        "end_utc": window.end.isoformat(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "maps": generated_maps
    }
    
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8") [cite: 52]


def main() -> None:
    end_dt = parse_end_arg() [cite: 53]
    window = build_time_window(end_dt) [cite: 53]

    cwa, counties, states = load_shapes() [cite: 53]
    lix, counties_p, states_p, plot_domain, cities_p = prepare_geodata(cwa, counties, states) [cite: 53]

    generated_maps = {}

    for period_key, period_info in PERIODS.items():
        print(f"--- Processing {period_info['title']} ---")
        try:
            tif_path = fetch_stageiv_qpe(window, period_info["noaa_str"])
            raster_arr, raster_extent_3857 = read_raster_for_plotting(tif_path)

            plot_map(
                window, lix, counties_p, states_p, plot_domain, cities_p, 
                raster_arr, raster_extent_3857, period_key, period_info["title"]
            )
            
            generated_maps[period_key] = {
                "image": f"lix_{period_key}_precip_latest.png",
                "source_tif": tif_path.name
            }
            print(f"Saved map to {OUT_DIR / f'lix_{period_key}_precip_latest.png'}")

        except Exception as e:
            print(f"Skipping {period_key}: {e}")

    write_outputs(window, generated_maps)
    print("Finished generating all maps.")


if __name__ == "__main__":
    main() [cite: 54]

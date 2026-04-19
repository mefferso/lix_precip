#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

API_BASE = "https://api.synopticdata.com/v2/stations"
DEFAULT_BBOX = "-92.5,28.5,-87.0,31.8"  # LIX-ish; widen/narrow as needed
DEFAULT_TIMEZONE = "America/Chicago"
DEFAULT_RECENT_MINUTES = 90
REQUEST_TIMEOUT = 180

# This is the whole point: products are config-driven.
# Add more products later without rewriting the whole damn script.
PRODUCTS: dict[str, dict[str, Any]] = {
    "precip_24h": {
        "label": "24-Hour Rainfall",
        "service": "precip",
        "params": {
            "pmode": "totals",
            "search": "nearest",
            "window": 60,
            "units": "english",
            "obtimezone": "local",
        },
        "csv": "station_precip_24h_latest.csv",
        "json": "station_precip_24h_latest.json",
    },
    "air_temp_latest": {
        "label": "Current Temperature",
        "service": "latest",
        "params": {
            "vars": "air_temp",
            "units": "english",
            "within": DEFAULT_RECENT_MINUTES,
        },
        "csv": "station_air_temp_latest.csv",
        "json": "station_air_temp_latest.json",
    },
    "air_temp_daily_min": {
        "label": "Daily Minimum Temperature",
        "service": "statistics",
        "params": {
            "vars": "air_temp",
            "statistic": "min",
            "period": "day",
            "units": "english",
            "obtimezone": "local",
        },
        "csv": "station_air_temp_daily_min_latest.csv",
        "json": "station_air_temp_daily_min_latest.json",
    },
    "air_temp_daily_max": {
        "label": "Daily Maximum Temperature",
        "service": "statistics",
        "params": {
            "vars": "air_temp",
            "statistic": "max",
            "period": "day",
            "units": "english",
            "obtimezone": "local",
        },
        "csv": "station_air_temp_daily_max_latest.csv",
        "json": "station_air_temp_daily_max_latest.json",
    },
}


@dataclass
class TimeWindow:
    start_utc: datetime
    end_utc: datetime


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_end_arg() -> datetime:
    if len(sys.argv) > 1 and sys.argv[1].strip():
        return datetime.strptime(sys.argv[1].strip(), "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    return utc_now().replace(second=0, microsecond=0)


def build_time_window(end_utc: datetime) -> TimeWindow:
    return TimeWindow(start_utc=end_utc - timedelta(hours=24), end_utc=end_utc)


def require_token() -> str:
    token = os.environ.get("SYNOPTIC_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing SYNOPTIC_TOKEN environment variable.")
    return token


def iso_to_local_str(ts: str | None) -> str:
    if not ts:
        return ""
    return ts


def request_json(service: str, params: dict[str, Any]) -> dict[str, Any]:
    token = require_token()
    url = f"{API_BASE}/{service}"
    query = {
        "token": token,
        "bbox": DEFAULT_BBOX,
        "status": "active",
        "output": "json",
        **params,
    }

    response = requests.get(url, params=query, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    payload = response.json()

    summary = payload.get("SUMMARY", {})
    code = str(summary.get("RESPONSE_CODE", ""))
    if code not in {"1", "OK", "200"}:
        raise RuntimeError(f"Synoptic {service} request failed: {summary}")

    return payload


def get_station_meta(station: dict[str, Any]) -> dict[str, Any]:
    return {
        "stid": station.get("STID", ""),
        "name": station.get("NAME", ""),
        "state": station.get("STATE", ""),
        "country": station.get("COUNTRY", ""),
        "county": station.get("COUNTY", ""),
        "cwa": station.get("CWA", ""),
        "nwszone": station.get("NWSZONE", ""),
        "timezone": station.get("TIMEZONE", DEFAULT_TIMEZONE),
        "elevation_ft": station.get("ELEVATION", ""),
        "lat": station.get("LATITUDE", ""),
        "lon": station.get("LONGITUDE", ""),
        "mnet_id": station.get("MNET_ID", ""),
    }


def flatten_precip_station(station: dict[str, Any], window: TimeWindow) -> dict[str, Any] | None:
    base = get_station_meta(station)
    obs = station.get("OBSERVATIONS", {})
    precip_list = obs.get("precipitation", [])
    if not precip_list:
        return None

    best = max(
        precip_list,
        key=lambda item: (
            float(item.get("total") or 0.0),
            int(item.get("count") or 0),
        ),
    )

    return {
        **base,
        "window_start_utc": window.start_utc.isoformat(),
        "window_end_utc": window.end_utc.isoformat(),
        "precip_in": round(float(best.get("total") or 0.0), 2),
        "first_report": iso_to_local_str(best.get("first_report")),
        "last_report": iso_to_local_str(best.get("last_report")),
        "report_type": best.get("report_type", ""),
        "count": best.get("count", ""),
    }


def pick_latest_value(obs: dict[str, Any], variable: str) -> tuple[Any, str]:
    for key, value in obs.items():
        if not key.startswith(variable):
            continue
        if isinstance(value, dict) and "value" in value:
            return value.get("value"), value.get("date_time", "")
    return "", ""


def flatten_latest_station(station: dict[str, Any], variable: str, value_col: str) -> dict[str, Any] | None:
    base = get_station_meta(station)
    obs = station.get("OBSERVATIONS", {})
    value, valid_time = pick_latest_value(obs, variable)
    if value in (None, ""):
        return None

    return {
        **base,
        value_col: round(float(value), 1),
        "valid_time": valid_time,
    }


def pick_stat_block(statistics: dict[str, Any], variable: str) -> dict[str, Any] | None:
    for key, value in statistics.items():
        if key.startswith(variable) and isinstance(value, dict):
            return value
    return None


def flatten_statistics_station(
    station: dict[str, Any],
    variable: str,
    stat_name: str,
    value_col: str,
) -> dict[str, Any] | None:
    base = get_station_meta(station)
    stats = station.get("STATISTICS", {})
    block = pick_stat_block(stats, variable)
    if not block or not isinstance(block, list):
        return None

    newest = block[-1]
    if not isinstance(newest, dict):
        return None

    value = newest.get(stat_name)
    if value in (None, ""):
        return None

    time_period = newest.get("time_period", {}) or {}
    valid_time = newest.get(f"{stat_name}_time", "")

    return {
        **base,
        value_col: round(float(value), 1),
        "valid_time": valid_time,
        "period_type": time_period.get("type", ""),
        "period_value": time_period.get("value", ""),
        "period_timezone": time_period.get("timezone", ""),
        "count": newest.get("count", ""),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sort_rows(rows: list[dict[str, Any]], value_col: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            row.get(value_col, float("-inf")) if isinstance(row.get(value_col), (int, float)) else -999999,
            row.get("stid", ""),
        ),
        reverse=True,
    )


def build_precip_product(window: TimeWindow) -> dict[str, Any]:
    config = PRODUCTS["precip_24h"]
    payload = request_json(
        "precip",
        {
            **config["params"],
            "start": window.start_utc.strftime("%Y%m%d%H%M"),
            "end": window.end_utc.strftime("%Y%m%d%H%M"),
        },
    )
    stations = payload.get("STATION", [])
    rows = [flatten_precip_station(stn, window) for stn in stations]
    rows = [row for row in rows if row is not None]
    rows = sort_rows(rows, "precip_in")

    csv_path = DOCS_DIR / config["csv"]
    json_path = DOCS_DIR / config["json"]

    write_csv(csv_path, rows)
    write_json(
        json_path,
        {
            "product": "precip_24h",
            "label": config["label"],
            "start_utc": window.start_utc.isoformat(),
            "end_utc": window.end_utc.isoformat(),
            "generated_utc": utc_now().isoformat(),
            "station_count": len(rows),
            "csv": csv_path.name,
            "rows": rows,
        },
    )

    return {
        "label": config["label"],
        "csv": csv_path.name,
        "json": json_path.name,
        "station_count": len(rows),
    }


def build_latest_product(product_key: str) -> dict[str, Any]:
    config = PRODUCTS[product_key]
    payload = request_json(config["service"], config["params"])
    stations = payload.get("STATION", [])
    rows = [flatten_latest_station(stn, "air_temp", "air_temp_f") for stn in stations]
    rows = [row for row in rows if row is not None]
    rows = sort_rows(rows, "air_temp_f")

    csv_path = DOCS_DIR / config["csv"]
    json_path = DOCS_DIR / config["json"]

    write_csv(csv_path, rows)
    write_json(
        json_path,
        {
            "product": product_key,
            "label": config["label"],
            "generated_utc": utc_now().isoformat(),
            "station_count": len(rows),
            "csv": csv_path.name,
            "rows": rows,
        },
    )

    return {
        "label": config["label"],
        "csv": csv_path.name,
        "json": json_path.name,
        "station_count": len(rows),
    }


def build_statistics_product(product_key: str, end_utc: datetime) -> dict[str, Any]:
    config = PRODUCTS[product_key]

    # Synoptic statistics with period=day requires YYYYmmdd, not YYYYmmddHHMM.
    # Since obtimezone=local, use the station-local/current local calendar day.
    local_day_str = end_utc.astimezone().strftime("%Y%m%d")

    payload = request_json(
        config["service"],
        {
            **config["params"],
            "start": local_day_str,
            "end": local_day_str,
        },
    )
    stations = payload.get("STATION", [])

    stat_name = config["params"]["statistic"]
    value_col = "air_temp_min_f" if stat_name == "min" else "air_temp_max_f"
    rows = [flatten_statistics_station(stn, "air_temp", stat_name, value_col) for stn in stations]
    rows = [row for row in rows if row is not None]
    rows = sort_rows(rows, value_col)

    csv_path = DOCS_DIR / config["csv"]
    json_path = DOCS_DIR / config["json"]

    write_csv(csv_path, rows)
    write_json(
        json_path,
        {
            "product": product_key,
            "label": config["label"],
            "generated_utc": utc_now().isoformat(),
            "station_count": len(rows),
            "csv": csv_path.name,
            "rows": rows,
        },
    )

    return {
        "label": config["label"],
        "csv": csv_path.name,
        "json": json_path.name,
        "station_count": len(rows),
    }


def write_manifest(window: TimeWindow, outputs: dict[str, Any]) -> None:
    manifest = {
        "generated_utc": utc_now().isoformat(),
        "bbox": DEFAULT_BBOX,
        "start_utc": window.start_utc.isoformat(),
        "end_utc": window.end_utc.isoformat(),
        "products": outputs,
    }
    write_json(DOCS_DIR / "latest_obs_manifest.json", manifest)


def main() -> None:
    end_utc = parse_end_arg()
    window = build_time_window(end_utc)

    outputs: dict[str, Any] = {}
    outputs["precip_24h"] = build_precip_product(window)
    outputs["air_temp_latest"] = build_latest_product("air_temp_latest")
    outputs["air_temp_daily_min"] = build_statistics_product("air_temp_daily_min", end_utc)
    outputs["air_temp_daily_max"] = build_statistics_product("air_temp_daily_max", end_utc)
    write_manifest(window, outputs)

    print("Finished building LIX station observation products.")
    for key, info in outputs.items():
        print(f"- {key}: {info['station_count']} stations -> {info['csv']}")


if __name__ == "__main__":
    main()

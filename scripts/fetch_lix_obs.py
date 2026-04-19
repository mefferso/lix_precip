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
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

API_BASE = "https://api.synopticdata.com/v2/stations"
DEFAULT_BBOX = "-92.5,28.5,-87.0,31.8"  # LIX-ish; widen/narrow as needed
DEFAULT_TIMEZONE = "America/Chicago"
CENTRAL_TZ = ZoneInfo(DEFAULT_TIMEZONE)
DEFAULT_RECENT_MINUTES = 90
REQUEST_TIMEOUT = 180

# Config-driven products so more variables can be added later without rewriting everything.
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
        "service": "timeseries",
        "params": {
            "vars": "air_temp",
            "units": "english",
            "obtimezone": "local",
        },
        "csv": "station_air_temp_daily_min_latest.csv",
        "json": "station_air_temp_daily_min_latest.json",
    },
    "air_temp_daily_max": {
        "label": "Daily Maximum Temperature",
        "service": "timeseries",
        "params": {
            "vars": "air_temp",
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


@dataclass
class LocalDayWindow:
    start_local: datetime
    end_local: datetime
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


def build_local_day_window(end_utc: datetime) -> LocalDayWindow:
    end_local = end_utc.astimezone(CENTRAL_TZ)
    start_local = end_local.replace(hour=0, minute=0, second=0, microsecond=0)
    end_local_day = start_local + timedelta(days=1)
    return LocalDayWindow(
        start_local=start_local,
        end_local=end_local_day,
        start_utc=start_local.astimezone(timezone.utc),
        end_utc=end_local_day.astimezone(timezone.utc),
    )


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


def get_timeseries_series(station: dict[str, Any], variable: str) -> tuple[list[Any], list[str]]:
    obs = station.get("OBSERVATIONS", {})
    value_key = ""
    for key in obs:
        if key.startswith(variable):
            value_key = key
            break

    if not value_key:
        return [], []

    values = obs.get(value_key, []) or []
    date_times = obs.get("date_time", []) or []
    return values, date_times


def flatten_timeseries_stat_station(
    station: dict[str, Any],
    variable: str,
    mode: str,
    value_col: str,
    day_window: LocalDayWindow,
) -> dict[str, Any] | None:
    base = get_station_meta(station)
    values, date_times = get_timeseries_series(station, variable)
    if not values or not date_times:
        return None

    candidates: list[tuple[float, str]] = []
    for value, ts in zip(values, date_times):
        if value in (None, ""):
            continue
        try:
            num = float(value)
        except (TypeError, ValueError):
            continue
        candidates.append((num, ts))

    if not candidates:
        return None

    chosen = min(candidates, key=lambda item: item[0]) if mode == "min" else max(candidates, key=lambda item: item[0])
    chosen_value, chosen_time = chosen

    return {
        **base,
        value_col: round(chosen_value, 1),
        "valid_time": chosen_time,
        "period_start_local": day_window.start_local.isoformat(),
        "period_end_local": day_window.end_local.isoformat(),
        "period_start_utc": day_window.start_utc.isoformat(),
        "period_end_utc": day_window.end_utc.isoformat(),
        "count": len(candidates),
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


def sort_rows(rows: list[dict[str, Any]], value_col: str, descending: bool = True) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            row.get(value_col, float("-inf")) if isinstance(row.get(value_col), (int, float)) else -999999,
            row.get("stid", ""),
        ),
        reverse=descending,
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
    rows = sort_rows(rows, "precip_in", descending=True)

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
    rows = sort_rows(rows, "air_temp_f", descending=True)

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


def build_daily_temp_extreme_product(product_key: str, end_utc: datetime) -> dict[str, Any]:
    config = PRODUCTS[product_key]
    day_window = build_local_day_window(end_utc)

    payload = request_json(
        config["service"],
        {
            **config["params"],
            "start": day_window.start_utc.strftime("%Y%m%d%H%M"),
            "end": day_window.end_utc.strftime("%Y%m%d%H%M"),
        },
    )
    stations = payload.get("STATION", [])

    is_min = product_key.endswith("_min")
    value_col = "air_temp_min_f" if is_min else "air_temp_max_f"
    mode = "min" if is_min else "max"

    rows = [
        flatten_timeseries_stat_station(stn, "air_temp", mode, value_col, day_window)
        for stn in stations
    ]
    rows = [row for row in rows if row is not None]
    rows = sort_rows(rows, value_col, descending=not is_min)

    csv_path = DOCS_DIR / config["csv"]
    json_path = DOCS_DIR / config["json"]

    write_csv(csv_path, rows)
    write_json(
        json_path,
        {
            "product": product_key,
            "label": config["label"],
            "period_start_local": day_window.start_local.isoformat(),
            "period_end_local": day_window.end_local.isoformat(),
            "period_start_utc": day_window.start_utc.isoformat(),
            "period_end_utc": day_window.end_utc.isoformat(),
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
    outputs["air_temp_daily_min"] = build_daily_temp_extreme_product("air_temp_daily_min", end_utc)
    outputs["air_temp_daily_max"] = build_daily_temp_extreme_product("air_temp_daily_max", end_utc)
    write_manifest(window, outputs)

    print("Finished building LIX station observation products.")
    for key, info in outputs.items():
        print(f"- {key}: {info['station_count']} stations -> {info['csv']}")


if __name__ == "__main__":
    main()

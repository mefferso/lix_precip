#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
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
CURRENT_TEMP_MATCH_WINDOW_MINUTES = 90
REQUEST_CONNECT_TIMEOUT = 20
REQUEST_READ_TIMEOUT = 180
REQUEST_RETRIES = 4
REQUEST_BACKOFF_SECONDS = 12
RTMA_MATCH_FILE = DOCS_DIR / "rtma_match_time.json"

PRODUCTS: dict[str, dict[str, Any]] = {
    "precip_24h": {
        "label": "24-Hour Rainfall",
        "service": "precip",
        "params": {"pmode": "totals", "search": "nearest", "window": 60, "units": "english", "obtimezone": "local"},
        "csv": "station_precip_24h_latest.csv",
        "json": "station_precip_24h_latest.json",
    },
    "air_temp_latest": {
        "label": "Current Temperature",
        "service": "latest",
        "params": {"vars": "air_temp", "units": "english", "within": DEFAULT_RECENT_MINUTES, "obtimezone": "utc"},
        "csv": "station_air_temp_latest.csv",
        "json": "station_air_temp_latest.json",
    },
    "air_temp_daily_min": {
        "label": "Daily Minimum Temperature",
        "service": "timeseries",
        "params": {"vars": "air_temp", "units": "english", "obtimezone": "local"},
        "csv": "station_air_temp_daily_min_latest.csv",
        "json": "station_air_temp_daily_min_latest.json",
    },
    "air_temp_daily_max": {
        "label": "Daily Maximum Temperature",
        "service": "timeseries",
        "params": {"vars": "air_temp", "units": "english", "obtimezone": "local"},
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

def build_local_calendar_day_window(day, end_override_utc: datetime | None = None) -> LocalDayWindow:
    start_local = datetime(day.year, day.month, day.day, 0, 0, tzinfo=CENTRAL_TZ)
    end_local = start_local + timedelta(days=1) if end_override_utc is None else end_override_utc.astimezone(CENTRAL_TZ)
    return LocalDayWindow(start_local=start_local, end_local=end_local, start_utc=start_local.astimezone(timezone.utc), end_utc=end_local.astimezone(timezone.utc))

def build_daily_temp_window(product_key: str, end_utc: datetime) -> LocalDayWindow:
    now_local = end_utc.astimezone(CENTRAL_TZ)
    today = now_local.date()
    yesterday = today - timedelta(days=1)
    if product_key == "air_temp_daily_max":
        return build_local_calendar_day_window(yesterday if now_local.hour < 16 else today)
    if product_key == "air_temp_daily_min":
        return build_local_calendar_day_window(today, end_override_utc=end_utc) if now_local.hour >= 9 else build_local_calendar_day_window(yesterday)
    return build_local_calendar_day_window(today)

def require_token() -> str:
    token = os.environ.get("SYNOPTIC_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing SYNOPTIC_TOKEN environment variable.")
    return token

def iso_to_local_str(ts: str | None) -> str:
    return "" if not ts else ts

def parse_synoptic_time(ts: str) -> datetime | None:
    if not ts:
        return None
    ts = str(ts).strip()
    try:
        if ts.endswith("Z"):
            ts = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

def read_rtma_match_time() -> datetime | None:
    if not RTMA_MATCH_FILE.exists():
        return None
    try:
        data = json.loads(RTMA_MATCH_FILE.read_text(encoding="utf-8"))
        value = data.get("rtma_valid_utc") or data.get("valid_utc")
        return parse_synoptic_time(value) if value else None
    except Exception as e:
        print(f"Could not read RTMA match time file {RTMA_MATCH_FILE}: {e}")
        return None

def request_json(service: str, params: dict[str, Any]) -> dict[str, Any]:
    token = require_token()
    url = f"{API_BASE}/{service}"
    query = {"token": token, "bbox": DEFAULT_BBOX, "status": "active", "output": "json", **params}
    last_error: Exception | None = None
    timeout = (REQUEST_CONNECT_TIMEOUT, REQUEST_READ_TIMEOUT)
    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            print(f"Synoptic {service} request attempt {attempt}/{REQUEST_RETRIES}...")
            response = requests.get(url, params=query, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            summary = payload.get("SUMMARY", {})
            code = str(summary.get("RESPONSE_CODE", ""))
            if code not in {"1", "OK", "200"}:
                raise RuntimeError(f"Synoptic {service} request failed: {summary}")
            return payload
        except (requests.exceptions.RequestException, ValueError, RuntimeError) as e:
            last_error = e
            if attempt >= REQUEST_RETRIES:
                break
            sleep_seconds = REQUEST_BACKOFF_SECONDS * attempt
            print(f"Synoptic {service} request failed on attempt {attempt}/{REQUEST_RETRIES}: {e}. Retrying in {sleep_seconds} seconds...")
            time.sleep(sleep_seconds)
    raise RuntimeError(f"Synoptic {service} request failed after {REQUEST_RETRIES} attempts: {last_error}")

def get_station_meta(station: dict[str, Any]) -> dict[str, Any]:
    return {
        "stid": station.get("STID", ""), "name": station.get("NAME", ""), "state": station.get("STATE", ""),
        "country": station.get("COUNTRY", ""), "county": station.get("COUNTY", ""), "cwa": station.get("CWA", ""),
        "nwszone": station.get("NWSZONE", ""), "timezone": station.get("TIMEZONE", DEFAULT_TIMEZONE),
        "elevation_ft": station.get("ELEVATION", ""), "lat": station.get("LATITUDE", ""),
        "lon": station.get("LONGITUDE", ""), "mnet_id": station.get("MNET_ID", ""),
    }

def flatten_precip_station(station: dict[str, Any], window: TimeWindow) -> dict[str, Any] | None:
    base = get_station_meta(station)
    precip_list = station.get("OBSERVATIONS", {}).get("precipitation", [])
    if not precip_list:
        return None
    best = max(precip_list, key=lambda item: (float(item.get("total") or 0.0), int(item.get("count") or 0)))
    return {**base, "window_start_utc": window.start_utc.isoformat(), "window_end_utc": window.end_utc.isoformat(), "precip_in": round(float(best.get("total") or 0.0), 2), "first_report": iso_to_local_str(best.get("first_report")), "last_report": iso_to_local_str(best.get("last_report")), "report_type": best.get("report_type", ""), "count": best.get("count", "")}

def pick_latest_value(obs: dict[str, Any], variable: str) -> tuple[Any, str]:
    for key, value in obs.items():
        if key.startswith(variable) and isinstance(value, dict) and "value" in value:
            return value.get("value"), value.get("date_time", "")
    return "", ""

def flatten_latest_station(station: dict[str, Any], variable: str, value_col: str) -> dict[str, Any] | None:
    base = get_station_meta(station)
    value, valid_time = pick_latest_value(station.get("OBSERVATIONS", {}), variable)
    if value in (None, ""):
        return None
    return {**base, value_col: round(float(value), 1), "valid_time": valid_time}

def get_timeseries_series(station: dict[str, Any], variable: str) -> tuple[list[Any], list[str]]:
    obs = station.get("OBSERVATIONS", {})
    value_key = next((key for key in obs if key.startswith(variable)), "")
    if not value_key:
        return [], []
    return obs.get(value_key, []) or [], obs.get("date_time", []) or []

def flatten_nearest_time_station(station: dict[str, Any], variable: str, value_col: str, target_utc: datetime) -> dict[str, Any] | None:
    base = get_station_meta(station)
    values, date_times = get_timeseries_series(station, variable)
    if not values or not date_times:
        return None
    best: tuple[float, str, datetime, float] | None = None
    for value, ts in zip(values, date_times):
        if value in (None, ""):
            continue
        obs_time_utc = parse_synoptic_time(ts)
        if obs_time_utc is None:
            continue
        try:
            num = float(value)
        except (TypeError, ValueError):
            continue
        delta_minutes = abs((obs_time_utc - target_utc).total_seconds()) / 60.0
        if delta_minutes > CURRENT_TEMP_MATCH_WINDOW_MINUTES:
            continue
        candidate = (num, ts, obs_time_utc, delta_minutes)
        if best is None or candidate[3] < best[3]:
            best = candidate
    if best is None:
        return None
    value, valid_time, valid_time_utc, delta_minutes = best
    return {**base, value_col: round(value, 1), "valid_time": valid_time_utc.isoformat(), "target_time_utc": target_utc.isoformat(), "minutes_from_target": round(delta_minutes, 1)}

def flatten_timeseries_stat_station(station: dict[str, Any], variable: str, mode: str, value_col: str, day_window: LocalDayWindow) -> dict[str, Any] | None:
    base = get_station_meta(station)
    values, date_times = get_timeseries_series(station, variable)
    if not values or not date_times:
        return None
    candidates: list[tuple[float, str, datetime]] = []
    for value, ts in zip(values, date_times):
        if value in (None, ""):
            continue
        try:
            num = float(value)
        except (TypeError, ValueError):
            continue
        obs_time_utc = parse_synoptic_time(ts)
        if obs_time_utc is None:
            continue
        if not (day_window.start_utc <= obs_time_utc < day_window.end_utc):
            continue
        candidates.append((num, ts, obs_time_utc))
    if not candidates:
        return None
    chosen = min(candidates, key=lambda item: item[0]) if mode == "min" else max(candidates, key=lambda item: item[0])
    chosen_value, chosen_time, chosen_time_utc = chosen
    return {**base, value_col: round(chosen_value, 1), "valid_time": chosen_time, "valid_time_utc": chosen_time_utc.isoformat(), "period_start_local": day_window.start_local.isoformat(), "period_end_local": day_window.end_local.isoformat(), "period_start_utc": day_window.start_utc.isoformat(), "period_end_utc": day_window.end_utc.isoformat(), "count": len(candidates)}

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def sort_rows(rows: list[dict[str, Any]], value_col: str, descending: bool = True) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (row.get(value_col, float("-inf")) if isinstance(row.get(value_col), (int, float)) else -999999, row.get("stid", "")), reverse=descending)

def build_precip_product(window: TimeWindow) -> dict[str, Any]:
    config = PRODUCTS["precip_24h"]
    payload = request_json("precip", {**config["params"], "start": window.start_utc.strftime("%Y%m%d%H%M"), "end": window.end_utc.strftime("%Y%m%d%H%M")})
    rows = [flatten_precip_station(stn, window) for stn in payload.get("STATION", [])]
    rows = sort_rows([row for row in rows if row is not None], "precip_in", descending=True)
    csv_path = DOCS_DIR / config["csv"]
    json_path = DOCS_DIR / config["json"]
    write_csv(csv_path, rows)
    write_json(json_path, {"product": "precip_24h", "label": config["label"], "start_utc": window.start_utc.isoformat(), "end_utc": window.end_utc.isoformat(), "generated_utc": utc_now().isoformat(), "station_count": len(rows), "csv": csv_path.name, "rows": rows})
    return {"label": config["label"], "csv": csv_path.name, "json": json_path.name, "station_count": len(rows)}

def build_latest_product(product_key: str) -> dict[str, Any]:
    config = PRODUCTS[product_key]
    rtma_target_utc = read_rtma_match_time() if product_key == "air_temp_latest" else None
    if product_key == "air_temp_latest" and rtma_target_utc is not None:
        start_utc = rtma_target_utc - timedelta(minutes=CURRENT_TEMP_MATCH_WINDOW_MINUTES)
        end_utc = rtma_target_utc + timedelta(minutes=CURRENT_TEMP_MATCH_WINDOW_MINUTES)
        print(f"Building current temperature stations matched to RTMA hour: {rtma_target_utc.strftime('%Y%m%d %HZ')} using obs window {start_utc.strftime('%Y%m%d %H:%MZ')} to {end_utc.strftime('%Y%m%d %H:%MZ')}")
        payload = request_json("timeseries", {"vars": "air_temp", "units": "english", "obtimezone": "utc", "start": start_utc.strftime("%Y%m%d%H%M"), "end": end_utc.strftime("%Y%m%d%H%M")})
        rows = [flatten_nearest_time_station(stn, "air_temp", "air_temp_f", rtma_target_utc) for stn in payload.get("STATION", [])]
        rows = sort_rows([row for row in rows if row is not None], "air_temp_f", descending=True)
        csv_path = DOCS_DIR / config["csv"]
        json_path = DOCS_DIR / config["json"]
        write_csv(csv_path, rows)
        write_json(json_path, {"product": product_key, "label": config["label"], "generated_utc": utc_now().isoformat(), "station_count": len(rows), "csv": csv_path.name, "matched_to_grid_model": "RTMA", "matched_to_grid_valid_utc": rtma_target_utc.isoformat(), "match_window_minutes": CURRENT_TEMP_MATCH_WINDOW_MINUTES, "rows": rows})
        return {"label": config["label"], "csv": csv_path.name, "json": json_path.name, "station_count": len(rows), "matched_to_grid_model": "RTMA", "matched_to_grid_valid_utc": rtma_target_utc.isoformat()}
    payload = request_json(config["service"], config["params"])
    rows = [flatten_latest_station(stn, "air_temp", "air_temp_f") for stn in payload.get("STATION", [])]
    rows = sort_rows([row for row in rows if row is not None], "air_temp_f", descending=True)
    csv_path = DOCS_DIR / config["csv"]
    json_path = DOCS_DIR / config["json"]
    write_csv(csv_path, rows)
    write_json(json_path, {"product": product_key, "label": config["label"], "generated_utc": utc_now().isoformat(), "station_count": len(rows), "csv": csv_path.name, "rows": rows})
    return {"label": config["label"], "csv": csv_path.name, "json": json_path.name, "station_count": len(rows)}

def build_daily_temp_extreme_product(product_key: str, end_utc: datetime) -> dict[str, Any]:
    config = PRODUCTS[product_key]
    day_window = build_daily_temp_window(product_key, end_utc)
    payload = request_json(config["service"], {**config["params"], "start": day_window.start_utc.strftime("%Y%m%d%H%M"), "end": day_window.end_utc.strftime("%Y%m%d%H%M")})
    is_min = product_key.endswith("_min")
    value_col = "air_temp_min_f" if is_min else "air_temp_max_f"
    mode = "min" if is_min else "max"
    rows = [flatten_timeseries_stat_station(stn, "air_temp", mode, value_col, day_window) for stn in payload.get("STATION", [])]
    rows = sort_rows([row for row in rows if row is not None], value_col, descending=not is_min)
    csv_path = DOCS_DIR / config["csv"]
    json_path = DOCS_DIR / config["json"]
    write_csv(csv_path, rows)
    write_json(json_path, {"product": product_key, "label": config["label"], "period_start_local": day_window.start_local.isoformat(), "period_end_local": day_window.end_local.isoformat(), "period_start_utc": day_window.start_utc.isoformat(), "period_end_utc": day_window.end_utc.isoformat(), "generated_utc": utc_now().isoformat(), "station_count": len(rows), "csv": csv_path.name, "rows": rows})
    return {"label": config["label"], "csv": csv_path.name, "json": json_path.name, "station_count": len(rows)}

def stale_product_summary(product_key: str, error: Exception) -> dict[str, Any] | None:
    config = PRODUCTS[product_key]
    csv_path = DOCS_DIR / config["csv"]
    json_path = DOCS_DIR / config["json"]
    if not csv_path.exists() or not json_path.exists():
        return None
    station_count = 0
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        station_count = int(payload.get("station_count") or len(payload.get("rows", [])) or 0)
    except Exception:
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                station_count = max(sum(1 for _ in f) - 1, 0)
        except Exception:
            station_count = 0
    print(f"WARNING: using existing {product_key} files because the fresh Synoptic request failed: {error}")
    return {"label": config["label"], "csv": csv_path.name, "json": json_path.name, "station_count": station_count, "stale": True, "warning": f"Fresh Synoptic request failed; reused previous files. Error: {error}"}

def build_or_reuse_existing(product_key: str, builder: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return builder()
    except Exception as e:
        stale = stale_product_summary(product_key, e)
        if stale is not None:
            return stale
        raise

def write_manifest(window: TimeWindow, outputs: dict[str, Any]) -> None:
    manifest = {"generated_utc": utc_now().isoformat(), "bbox": DEFAULT_BBOX, "start_utc": window.start_utc.isoformat(), "end_utc": window.end_utc.isoformat(), "products": outputs}
    write_json(DOCS_DIR / "latest_obs_manifest.json", manifest)

def main() -> None:
    end_utc = parse_end_arg()
    window = build_time_window(end_utc)
    outputs: dict[str, Any] = {}
    outputs["precip_24h"] = build_or_reuse_existing("precip_24h", lambda: build_precip_product(window))
    outputs["air_temp_latest"] = build_or_reuse_existing("air_temp_latest", lambda: build_latest_product("air_temp_latest"))
    outputs["air_temp_daily_min"] = build_or_reuse_existing("air_temp_daily_min", lambda: build_daily_temp_extreme_product("air_temp_daily_min", end_utc))
    outputs["air_temp_daily_max"] = build_or_reuse_existing("air_temp_daily_max", lambda: build_daily_temp_extreme_product("air_temp_daily_max", end_utc))
    write_manifest(window, outputs)
    print("Finished building LIX station observation products.")
    for key, info in outputs.items():
        stale_note = " (reused previous files)" if info.get("stale") else ""
        print(f"- {key}: {info['station_count']} stations -> {info['csv']}{stale_note}")

if __name__ == "__main__":
    main()

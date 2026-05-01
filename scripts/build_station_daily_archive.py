#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
ARCHIVE_ROOT = DOCS_DIR / "archive" / "station"
MANIFEST_PATH = ARCHIVE_ROOT / "station_manifest.json"

LOCAL_TZ = ZoneInfo("America/Chicago")
LOCAL_END_HOUR = 6

DATASETS_TO_ARCHIVE = {
    "precip_24h": [
        "lix_station_precip_24h_latest.png",
        "lix_station_precip_24h_baton_rouge_metro_latest.png",
        "lix_station_precip_24h_new_orleans_metro_latest.png",
        "lix_station_precip_24h_southwest_ms_latest.png",
        "lix_station_precip_24h_coastal_ms_latest.png",
        "lix_station_precip_24h_northshore_latest.png",
    ],
    "air_temp_daily_min": [
        "lix_station_air_temp_daily_min_latest.png",
        "lix_station_air_temp_daily_min_baton_rouge_metro_latest.png",
        "lix_station_air_temp_daily_min_new_orleans_metro_latest.png",
        "lix_station_air_temp_daily_min_southwest_ms_latest.png",
        "lix_station_air_temp_daily_min_coastal_ms_latest.png",
        "lix_station_air_temp_daily_min_northshore_latest.png",
    ],
    "air_temp_daily_max": [
        "lix_station_air_temp_daily_max_latest.png",
        "lix_station_air_temp_daily_max_baton_rouge_metro_latest.png",
        "lix_station_air_temp_daily_max_new_orleans_metro_latest.png",
        "lix_station_air_temp_daily_max_southwest_ms_latest.png",
        "lix_station_air_temp_daily_max_coastal_ms_latest.png",
        "lix_station_air_temp_daily_max_northshore_latest.png",
    ],
}

REGION_BY_FILENAME_PART = {
    "_baton_rouge_metro_": "baton_rouge_metro",
    "_new_orleans_metro_": "new_orleans_metro",
    "_southwest_ms_": "southwest_ms",
    "_coastal_ms_": "coastal_ms",
    "_northshore_": "northshore",
}


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def iter_dates(start_date: date, end_date: date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def archive_date_dir(day: date) -> Path:
    return ARCHIVE_ROOT / f"{day.year:04d}" / f"{day.month:02d}" / f"{day.day:02d}"


def local_day_to_end_utc_arg(day: date) -> str:
    end_local = datetime(
        day.year,
        day.month,
        day.day,
        LOCAL_END_HOUR,
        0,
        tzinfo=LOCAL_TZ,
    )
    end_utc = end_local.astimezone(timezone.utc)
    return end_utc.strftime("%Y%m%d%H%M")


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {
            "generated_utc": None,
            "archive_type": "station_daily",
            "products": [
                "precip_24h",
                "air_temp_daily_min",
                "air_temp_daily_max",
            ],
            "regions": [
                "full",
                "baton_rouge_metro",
                "new_orleans_metro",
                "southwest_ms",
                "coastal_ms",
                "northshore",
            ],
            "dates": {},
        }

    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {
            "generated_utc": None,
            "archive_type": "station_daily",
            "products": [
                "precip_24h",
                "air_temp_daily_min",
                "air_temp_daily_max",
            ],
            "regions": [
                "full",
                "baton_rouge_metro",
                "new_orleans_metro",
                "southwest_ms",
                "coastal_ms",
                "northshore",
            ],
            "dates": {},
        }


def write_manifest(manifest: dict) -> None:
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    manifest["generated_utc"] = datetime.now(timezone.utc).isoformat()
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def region_for_png(filename: str) -> str:
    for part, region in REGION_BY_FILENAME_PART.items():
        if part in filename:
            return region
    return "full"


def run_command(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def build_for_date(day: date, skip_existing: bool) -> dict:
    day_key = day.isoformat()
    out_dir = archive_date_dir(day)

    if skip_existing and out_dir.exists() and any(out_dir.glob("*.png")):
        print(f"Archive already exists for {day_key}; skipping.")
        return {}

    end_arg = local_day_to_end_utc_arg(day)

    run_command([sys.executable, "scripts/fetch_lix_obs.py", end_arg])
    run_command([sys.executable, "scripts/build_lix_obs_maps.py"])

    out_dir.mkdir(parents=True, exist_ok=True)

    day_manifest: dict[str, dict[str, str]] = {}

    for dataset_key, filenames in DATASETS_TO_ARCHIVE.items():
        day_manifest[dataset_key] = {}

        for filename in filenames:
            src = DOCS_DIR / filename
            if not src.exists():
                print(f"WARNING: missing expected PNG: {filename}")
                continue

            dst = out_dir / filename
            shutil.copy2(src, dst)

            region = region_for_png(filename)
            rel_path = dst.relative_to(DOCS_DIR).as_posix()
            day_manifest[dataset_key][region] = rel_path

    return day_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill or update daily station map archive."
    )
    parser.add_argument("--start-date", help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", help="End date YYYY-MM-DD")
    parser.add_argument(
        "--days-back",
        type=int,
        default=None,
        help="Build archive for N days ending today local.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Regenerate dates even when archive PNGs already exist.",
    )

    args = parser.parse_args()

    if args.days_back is not None:
        today_local = datetime.now(timezone.utc).astimezone(LOCAL_TZ).date()
        end_day = today_local
        start_day = today_local - timedelta(days=max(args.days_back - 1, 0))
    else:
        if not args.start_date or not args.end_date:
            raise SystemExit(
                "Provide --start-date and --end-date, or use --days-back."
            )
        start_day = parse_date(args.start_date)
        end_day = parse_date(args.end_date)

    if end_day < start_day:
        raise SystemExit("end date must be on or after start date")

    manifest = load_manifest()
    manifest.setdefault("dates", {})

    skip_existing = not args.no_skip_existing

    for day in iter_dates(start_day, end_day):
        print(f"\n=== Building archive for {day.isoformat()} ===")
        day_manifest = build_for_date(day, skip_existing=skip_existing)

        if day_manifest:
            manifest["dates"][day.isoformat()] = day_manifest
            write_manifest(manifest)

    write_manifest(manifest)
    print(f"Archive manifest written: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()

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


def base_manifest() -> dict:
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
        "skipped_dates": {},
    }


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return base_manifest()

    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            return base_manifest()
    except Exception:
        return base_manifest()

    manifest.setdefault("dates", {})
    manifest.setdefault("skipped_dates", {})
    manifest.setdefault("products", base_manifest()["products"])
    manifest.setdefault("regions", base_manifest()["regions"])
    manifest.setdefault("archive_type", "station_daily")
    return manifest


def write_manifest(manifest: dict) -> None:
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    manifest["generated_utc"] = datetime.now(timezone.utc).isoformat()
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def region_for_png(filename: str) -> str:
    for part, region in REGION_BY_FILENAME_PART.items():
        if part in filename:
            return region
    return "full"


def run_command(cmd: list[str]) -> subprocess.CompletedProcess:
    print("Running:", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=True)


def archive_existing_pngs() -> dict[str, dict[str, str]]:
    day_manifest: dict[str, dict[str, str]] = {}

    for dataset_key, filenames in DATASETS_TO_ARCHIVE.items():
        day_manifest[dataset_key] = {}

        for filename in filenames:
            src = DOCS_DIR / filename
            if not src.exists():
                print(f"WARNING: missing expected PNG: {filename}", flush=True)
                continue

            region = region_for_png(filename)
            day_manifest[dataset_key][region] = filename

    return day_manifest


def copy_day_pngs_to_archive(day: date) -> dict[str, dict[str, str]]:
    out_dir = archive_date_dir(day)
    out_dir.mkdir(parents=True, exist_ok=True)

    day_manifest: dict[str, dict[str, str]] = {}

    for dataset_key, filenames in DATASETS_TO_ARCHIVE.items():
        day_manifest[dataset_key] = {}

        for filename in filenames:
            src = DOCS_DIR / filename
            if not src.exists():
                print(f"WARNING: missing expected PNG: {filename}", flush=True)
                continue

            dst = out_dir / filename
            shutil.copy2(src, dst)

            region = region_for_png(filename)
            rel_path = dst.relative_to(DOCS_DIR).as_posix()
            day_manifest[dataset_key][region] = rel_path

    return day_manifest


def count_archived_images(day_manifest: dict[str, dict[str, str]]) -> int:
    return sum(len(region_map) for region_map in day_manifest.values())


def build_for_date(day: date, skip_existing: bool) -> dict:
    day_key = day.isoformat()
    out_dir = archive_date_dir(day)

    if skip_existing and out_dir.exists() and any(out_dir.glob("*.png")):
        print(f"Archive already exists for {day_key}; skipping.", flush=True)
        return {"status": "skipped_existing", "date": day_key}

    end_arg = local_day_to_end_utc_arg(day)

    try:
        # fetch station obs for this date
        run_command([sys.executable, "scripts/fetch_lix_obs.py", end_arg])

        # build archive maps using date-aware URMA/MRMS
        run_command([
            sys.executable,
            "scripts/build_lix_obs_archive_maps.py",
            "--end-time",
            end_arg,
        ])

    except subprocess.CalledProcessError as exc:
        reason = (
            f"Archive build failed for {day_key}. "
            f"Command exited with status {exc.returncode}: {' '.join(str(x) for x in exc.cmd)}"
        )
        print(f"WARNING: {reason}", flush=True)
        return {
            "status": "skipped_failed_build",
            "date": day_key,
            "end_time_utc": end_arg,
            "reason": reason,
        }

    day_manifest = copy_day_pngs_to_archive(day)
    image_count = count_archived_images(day_manifest)

    if image_count == 0:
        reason = f"Archive build for {day_key} produced no PNGs."
        print(f"WARNING: {reason}", flush=True)
        return {
            "status": "skipped_no_images",
            "date": day_key,
            "end_time_utc": end_arg,
            "reason": reason,
        }

    return {
        "status": "ok",
        "date": day_key,
        "end_time_utc": end_arg,
        "images": day_manifest,
        "image_count": image_count,
    }


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
    parser.add_argument(
        "--fail-on-skipped",
        action="store_true",
        help="Return a non-zero exit code if any dates fail or are skipped due to missing data.",
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
    manifest.setdefault("skipped_dates", {})

    skip_existing = not args.no_skip_existing
    failed_or_skipped_dates: list[str] = []

    for day in iter_dates(start_day, end_day):
        day_key = day.isoformat()
        print(f"\n=== Building archive for {day_key} ===", flush=True)
        result = build_for_date(day, skip_existing=skip_existing)
        status = result.get("status")

        if status == "ok":
            manifest["dates"][day_key] = result["images"]
            manifest["skipped_dates"].pop(day_key, None)
            print(f"Archived {result['image_count']} PNGs for {day_key}.", flush=True)
            write_manifest(manifest)
        elif status == "skipped_existing":
            print(f"Skipped {day_key}: archive already exists.", flush=True)
        else:
            manifest["skipped_dates"][day_key] = {
                "status": status,
                "reason": result.get("reason", "Unknown reason"),
                "end_time_utc": result.get("end_time_utc"),
                "recorded_utc": datetime.now(timezone.utc).isoformat(),
            }
            failed_or_skipped_dates.append(day_key)
            print(f"Skipped {day_key}: {result.get('reason', status)}", flush=True)
            write_manifest(manifest)

    write_manifest(manifest)
    print(f"Archive manifest written: {MANIFEST_PATH}", flush=True)

    if failed_or_skipped_dates:
        print(
            "Completed with skipped/failed dates: " + ", ".join(failed_or_skipped_dates),
            flush=True,
        )
        if args.fail_on_skipped:
            raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Read-only downloads for all Milon training sources."""

import datetime as dt
from pathlib import Path

import requests

from training_data import merge_history, migrate_cache, premium_records, read_json, write_json


HOST = "https://www.milonme.com"


def download_all(session, user_id, studio_id, data_folder):
    """Refresh monthly logs, rolling histories, and dated snapshots together.

    Each source can fail independently; successful caches survive later failures.
    The range selected for plotting never restricts the downloaded history.
    """
    data_folder = Path(data_folder)
    stats = data_folder / str(user_id) / "stats"
    migrate_cache(stats)
    now = dt.datetime.now(dt.timezone.utc)
    captured = now.isoformat()
    manifest = {"captured_at": captured, "endpoints": {}}

    def fetch(name, endpoint, *, snapshot=False, destination=None):
        try:
            response = session.get(HOST + endpoint, timeout=30)
            response.raise_for_status()
            value = response.json()
            if isinstance(value, dict) and "message" in value:
                raise ValueError("API returned an error object")
            if name.startswith("premium/") and (not isinstance(value, dict) or not isinstance(value.get("stats"), list)):
                raise ValueError("Expected premium statistics list")
            cache_path = destination if destination is not None else stats / (name + ".json")
            merged = merge_history(name, read_json(cache_path), value)
            write_json(cache_path, merged)
            if snapshot:
                write_json(stats / "snapshots" / captured[:10] / (name + ".json"),
                           {"captured_at": captured, "data": value})
            manifest["endpoints"][name] = {"ok": True, "captured_at": captured}
            print(f"  {name}: downloaded")
            return value
        except (requests.RequestException, ValueError) as error:
            manifest["endpoints"][name] = {"ok": False, "error": type(error).__name__}
            print(f"  {name}: unavailable ({type(error).__name__}); keeping any cached data")
            return None

    print("Downloading training data:")
    fetch("home", f"/api/user/stats/home/{studio_id}/{user_id}")
    fetch("devices", "/api/devices/en_US", destination=data_folder / "devices.json")

    # Refresh the newest cached month too: it may have been partial on the last run.
    files = sorted((stats / "premium").glob("[0-9][0-9][0-9][0-9].json"))
    last_yymm = files[-1].stem if files else None
    year, month = now.year, now.month
    for _ in range(13):
        yymm = f"{year % 100:02}{month:02}"
        fetch(f"premium/{yymm}", f"/api/user/stats/premium/{studio_id}/{user_id}/{yymm}")
        if yymm == last_yymm:
            break
        month -= 1
        if month == 0:
            month = 12
            year -= 1

    for name in ("bioage", "bioage_history", "muscles", "progress"):
        fetch(name, f"/api/user/stats/{name}/{user_id}", snapshot=name in ("bioage", "muscles"))
    fetch("progress-devices", f"/api/user/stats/progress/devices/{user_id}")

    plans = {str(r["training"]["plan_id"]) for r in premium_records(stats)
             if r["training"].get("plan_id")}
    current = (read_json(stats / "home.json", {}) or {}).get("profile", {}).get("currplan")
    if current:
        plans.add(str(current))
    for plan in sorted(plans):
        fetch(f"plans/{plan}", f"/api/user/plans/premium/{user_id}/{plan}", snapshot=True)
    write_json(stats / "manifest.json", manifest)

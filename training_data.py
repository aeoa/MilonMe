"""Cached training data, history retention, and shared plotting context."""

import json
from collections import defaultdict
from functools import cached_property
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

TZ = ZoneInfo("Europe/Berlin")

def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def premium_records(stats):
    """Deduplicate overlapping monthly caches without dropping same-day sessions."""
    records = {}
    for path in sorted((Path(stats) / "premium").glob("*.json")):
        for record in (read_json(path, {}) or {}).get("stats", []):
            training = record["training"]
            key = (training["t"], training.get("trainingitem_id"))
            records[key] = record
    return sorted(records.values(), key=lambda r: r["training"]["t"])


def merge_rows(previous, incoming, keys):
    """Refresh matching buckets while retaining rows outside the API's window."""
    rows = {}
    for row in [*(previous or []), *(incoming or [])]:
        if not isinstance(row, dict) or any(row.get(key) is None for key in keys):
            raise ValueError("Missing history key in API response")
        rows[tuple(row[key] for key in keys)] = row
    return sorted(rows.values(), key=lambda row: tuple(row[key] for key in keys))


def merge_history(name, previous, incoming):
    if name == "bioage_history":
        if not isinstance(incoming, list):
            raise ValueError("Expected biological-age history list")
        return merge_rows(previous, incoming, ("t",))
    fields = {
        "progress": ("nodegroups_weeks", "nodegroups_months"),
        "progress-devices": ("devices_weeks", "devices_months"),
    }
    if name not in fields:
        return incoming
    if not isinstance(incoming, dict):
        raise ValueError("Expected progress object")
    previous = previous or {}
    merged = {**previous, **incoming}
    keys = ("datepart", "nodegroup_type", "id") if name == "progress-devices" else ("datepart", "nodegroup_type")
    for field in fields[name]:
        if field not in incoming or not isinstance(incoming[field], list):
            raise ValueError("Missing progress history list")
        merged[field] = merge_rows(previous.get(field, []), incoming[field], keys)
    if name == "progress":
        merged["devices"] = sorted(set(previous.get("devices", [])) | set(incoming.get("devices", [])))
    return merged


def migrate_cache(stats):
    """Copy the old insights namespace once, preserving the original as a backup."""
    stats = Path(stats)
    legacy = stats / "insights"
    marker = stats / ".insights-migrated"
    if not legacy.exists() or marker.exists():
        return
    for source in sorted(legacy.rglob("*.json")):
        relative = source.relative_to(legacy)
        if relative.name == "manifest.json":
            continue
        destination = stats / relative
        value = read_json(source)
        if value is None:
            raise ValueError(f"Cannot migrate invalid cache: {source}")
        if destination.exists():
            current = read_json(destination)
            if current is None:
                raise ValueError(f"Cannot merge invalid cache: {destination}")
            if len(relative.parts) == 1 and relative.stem in ("bioage_history", "progress", "progress-devices"):
                value = merge_history(relative.stem, value, current)
            else:
                continue
        write_json(destination, value)
    marker.write_text("Copied insights cache; original retained as backup.\n")


def strength_data(records):
    """Prepare the same filtered exercise history for reporting and strength plots."""
    data = pd.DataFrame(columns=["training", "set", "device", "time", "duration", "moves", "concentric", "eccentric", "work"])
    for training in records:
        device_count = defaultdict(int)
        for device in training["devices"]:
            if "moves" not in device:
                continue
            device_count[device["id"]] += 1
            data.loc[len(data)] = [
                training["training"]["t"],
                device_count[device["id"]],
                device["id"],
                device["t"],
                device["d"],
                device["moves"],
                device["aw"],
                device["adw"],
                device["ws"],
            ]
    if data.empty:
        return data
    data.sort_values(by=["time"], inplace=True)
    data = data[data["time"] > data["time"].min() + 3 * 60 * 60]
    data = data[~((data["device"] == 15) & (data["concentric"] == 12) & (data["eccentric"] == 14)  & (data["duration"] == 10)  & (data["moves"] == 1))]
    
    # Konvertierung in DateTime
    local_tz = TZ
    data["time"] = pd.to_datetime(data["time"], unit='s', utc=True).dt.tz_convert(local_tz)
    data["training"] = pd.to_datetime(data["training"], unit='s', utc=True).dt.tz_convert(local_tz)

    return data


@dataclass
class PlotContext:
    stats: Path
    output: Path
    devices: dict
    records: list
    days: int
    now: pd.Timestamp
    cutoff: pd.Timestamp | None

    @cached_property
    def strength_data(self):
        return strength_data(self.records)

    @classmethod
    def load(cls, user_id, data_folder, graph_folder, days=365):
        data_folder = Path(data_folder)
        stats = data_folder / str(user_id) / "stats"
        migrate_cache(stats)
        output = Path(graph_folder) / str(user_id)
        output.mkdir(parents=True, exist_ok=True)
        now = pd.Timestamp.now(tz=TZ)
        cutoff = now - pd.Timedelta(days=days) if days > 0 else None
        return cls(stats, output, read_json(data_folder / "devices.json", {}) or {},
                   premium_records(stats), days, now, cutoff)


def timestamp(value):
    return pd.to_datetime(value, unit="s", utc=True).tz_convert(TZ)



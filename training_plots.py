"""Training visualizations, with one context and registry for all plot groups."""

import datetime
import json
from collections import defaultdict

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from training_data import TZ, read_json, timestamp

GROUPS = {"arm": "Arms", "back": "Back", "body": "Abdomen", "chest": "Chest", "leg": "Legs"}

def configure_date_axis(ax):
    locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
    # Allow every month in a one-year view, including the plot margins.
    locator.maxticks[mdates.MONTHLY] = 14
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.tick_params(axis="x", labelsize=8, labelbottom=True)


def plot_history_series(ax, times, values, label, cutoff=None):
    """Plot the selected range, retaining a reference to the full-history peak."""
    series = pd.Series(list(values), index=pd.DatetimeIndex(times), dtype=float)
    series = series.replace([float("inf"), -float("inf")], float("nan")).dropna()
    visible = series if cutoff is None or series.empty else series[series.index >= cutoff]
    line, = ax.plot(visible.index, visible.values, label=label)
    if series.empty:
        return

    peak = series.max()
    color = line.get_color()
    line.set_label(f"{label} (ATH {peak:g})")
    ax.axhline(peak, color=color, linestyle="--", linewidth=0.8, alpha=0.4)
    if not visible.empty:
        # Mark the most recent occurrence when the same maximum was reached repeatedly.
        range_peak = visible.max()
        peak_time = visible[visible == range_peak].index[-1]
        ax.plot([peak_time], [range_peak], linestyle="none",
                marker="o", markersize=5,
                markerfacecolor=color if range_peak == peak else "white",
                markeredgecolor=color, color=color, zorder=4)


def explain_peak_markers(fig):
    fig.text(0.5, 0.008,
             "Dashed line / filled circle: all-time high    •    Hollow circle: range maximum below all-time high",
             ha="center", fontsize=8, color="0.4")


def mark_year_boundaries(axes):
    for ax in axes.flat:
        if not ax.get_visible():
            continue
        left, right = ax.get_xlim()
        start = mdates.num2date(left)
        end = mdates.num2date(right)
        for year in range(start.year, end.year + 1):
            boundary = datetime.datetime(year, 1, 1, tzinfo=start.tzinfo)
            if left < mdates.date2num(boundary) < right:
                ax.axvline(boundary, color="0.5", linewidth=0.6, alpha=0.35, zorder=0)


def plot_weights(data, devices, output_folder, ids, cutoff=None):
    """Plottet Eccentric und Concentric als Gewichtsdarstellung und speichert weights.png"""
    fig, axes = plt.subplots(nrows=3, ncols=4, sharex=True)
    fig.subplots_adjust(left=0.04, bottom=0.085, right=0.98, top=0.965, wspace=0.15, hspace=0.28)
    fig.set_size_inches(16, 9)
    for ax in axes.flat:
        configure_date_axis(ax)
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    for idx, device_id in enumerate(ids):
        row = idx // axes.shape[1]
        col = idx % axes.shape[1]
        ax = axes[row, col]
        # Filter für Gerät
        data_id = data[data["device"] == device_id]
        plot_history_series(ax, data_id["time"], data_id["eccentric"], label="Eccentric", cutoff=cutoff)
        plot_history_series(ax, data_id["time"], data_id["concentric"], label="Concentric", cutoff=cutoff)
        ax.set_title(devices[str(device_id)]["name"])
        if col == 0:
            ax.set_ylabel("Weight / kg")
        ax.legend(loc='lower right')
    explain_peak_markers(fig)
    mark_year_boundaries(axes)
    save_figure(fig, output_folder, "weights.png", dpi=300)

def plot_delta_percentage(data, devices, output_folder, ids, cutoff=None):
    """Plottet den prozentualen Unterschied zwischen Eccentric und Concentric und speichert delta percentage.png"""
    fig, axes = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.04, bottom=0.085, right=0.98, top=0.965, wspace=0.15, hspace=0.28)
    fig.set_size_inches(16, 9)
    for ax in axes.flat:
        configure_date_axis(ax)
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    for idx, device_id in enumerate(ids):
        row = idx // axes.shape[1]
        col = idx % axes.shape[1]
        ax = axes[row, col]
        data_id = data[data["device"] == device_id]
        ax.axhline(y=20, color='r', linestyle='--', label="20%")
        percentage = (data_id["eccentric"] / data_id["concentric"] - 1) * 100
        plot_history_series(ax, data_id["time"], percentage, label="delta %", cutoff=cutoff)
        ax.set_title(devices[str(device_id)]["name"])
        if col == 0:
            ax.set_ylabel("Percentage")
        ax.legend(loc='lower right')
    explain_peak_markers(fig)
    mark_year_boundaries(axes)
    save_figure(fig, output_folder, "delta percentage.png", dpi=300)

def plot_reps(data, devices, output_folder, ids, cutoff=None):
    """Plottet Wiederholungen (reps) und speichert reps.png"""
    fig, axes = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.04, bottom=0.085, right=0.98, top=0.965, wspace=0.15, hspace=0.28)
    fig.set_size_inches(16, 9)
    for ax in axes.flat:
        configure_date_axis(ax)
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    for idx, device_id in enumerate(ids):
        row = idx // axes.shape[1]
        col = idx % axes.shape[1]
        ax = axes[row, col]
        data_id = data[data["device"] == device_id]
        ax.axhline(y=8, color='r', linestyle='--', label="8")
        plot_history_series(ax, data_id["time"], data_id["moves"], label="reps", cutoff=cutoff)
        ax.set_title(devices[str(device_id)]["name"])
        if col == 0:
            ax.set_ylabel("Repetitions")
        ax.legend(loc='lower right')
    explain_peak_markers(fig)
    mark_year_boundaries(axes)
    save_figure(fig, output_folder, "reps.png", dpi=300)

def plot_work_individual(data_training_accumulated, devices, output_folder, ids, cutoff=None):
    fig, axes = plt.subplots(nrows=3, ncols=4, sharex=True)
    fig.subplots_adjust(left=0.04, bottom=0.085, right=0.98, top=0.965, wspace=0.15, hspace=0.28)
    fig.set_size_inches(16, 9)
    for ax in axes.flat:
        configure_date_axis(ax)
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    for idx, device_id in enumerate(ids):
        row = idx // axes.shape[1]
        col = idx % axes.shape[1]
        ax = axes[row, col]
        data_id_accumulated_3 = data_training_accumulated[(data_training_accumulated["device"] == device_id) & (data_training_accumulated["sets"] == 3)]
        plot_history_series(ax, data_id_accumulated_3["training"], data_id_accumulated_3["work"], label="3 sets", cutoff=cutoff)
        data_id_accumulated_2 = data_training_accumulated[(data_training_accumulated["device"] == device_id) & (data_training_accumulated["sets"] == 2)]
        plot_history_series(ax, data_id_accumulated_2["training"], data_id_accumulated_2["work"], label="2 sets", cutoff=cutoff)
        ax.set_title(devices[str(device_id)]["name"])
        if col == 0:
            ax.set_ylabel("Work / kWs")
        ax.legend(loc='lower right')
    explain_peak_markers(fig)
    mark_year_boundaries(axes)
    save_figure(fig, output_folder, "work_individual.png", dpi=300)

def plot_work_muscle_group(data_training_accumulated, devices, output_folder, ids, cutoff=None):
        # Collect and map muscle groups to device names
        mg_to_names = {}
        for device_id in ids:
            mg = devices[str(device_id)]["mg"]
            name = devices[str(device_id)]["name"]
            mg_to_names.setdefault(mg, set()).add(name)
        # Convert sets to sorted lists
        for mg in mg_to_names:
            mg_to_names[mg] = sorted(mg_to_names[mg])
        
        # Collect the muscle groups for the given device ids
        muscle_groups = sorted({devices[str(device_id)]["mg"] for device_id in ids})
        total_plots = 1 + len(muscle_groups)  # one overall plot + one per muscle group

        # Use a 2d grid with 2 columns
        nrows = 2
        ncols = (total_plots + nrows - 1) // nrows

        # Create a figure with subplots arranged in a grid
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True)
        fig.subplots_adjust(left=0.04, bottom=0.085, right=0.98, top=0.965, wspace=0.15, hspace=0.28)
        fig.set_size_inches(16, 9)

        for ax in axes.flat:
            configure_date_axis(ax)
            ax.yaxis.get_major_locator().set_params(integer=True)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
            ax.set_visible(False)

        # ----- Plot 1: Total Work for All Devices -----
        valid_3 = data_training_accumulated.groupby("training")["sets"].apply(lambda x: x.eq(3).all())
        trainings_3 = valid_3[valid_3].index
        valid_2 = data_training_accumulated.groupby("training")["sets"].apply(lambda x: x.eq(2).all())
        trainings_2 = valid_2[valid_2].index

        total_3 = data_training_accumulated[
            data_training_accumulated["training"].isin(trainings_3)
        ].groupby("training")["work"].sum()
        total_2 = data_training_accumulated[
            data_training_accumulated["training"].isin(trainings_2)
        ].groupby("training")["work"].sum()

        idx = 0
        row = idx // axes.shape[1]
        col = idx % axes.shape[1]
        ax = axes[row, col]
        ax.set_visible(True)
        plot_history_series(ax, total_3.index, total_3.values, label="3 sets", cutoff=cutoff)
        plot_history_series(ax, total_2.index, total_2.values, label="2 sets", cutoff=cutoff)
        ax.set_title("Total Work (All Devices)")
        if col == 0:
            ax.set_ylabel("Work / kWs")
        ax.legend(loc='lower right')
        idx += 1

        # ----- Plot 2+: Total Work per Muscle Group -----
        for mg in muscle_groups:
            row = idx // axes.shape[1]
            col = idx % axes.shape[1]
            ax = axes[row, col]
            ax.set_visible(True)

            # Get device ids corresponding to the current muscle group
            mg_device_ids = [device_id for device_id in ids if devices[str(device_id)]["mg"] == mg]
            mg_data = data_training_accumulated[data_training_accumulated["device"].isin(mg_device_ids)]
            if mg_data.empty:
                ax.set_title(f"{mg.capitalize()} (no devices)")
                ax.set_ylabel("Work / kWs")
                idx += 1
                continue

            valid_3_mg = mg_data.groupby("training")["sets"].apply(lambda x: x.eq(3).all())
            trainings_3_mg = valid_3_mg[valid_3_mg].index
            valid_2_mg = mg_data.groupby("training")["sets"].apply(lambda x: x.eq(2).all())
            trainings_2_mg = valid_2_mg[valid_2_mg].index

            total_3_mg = mg_data[
            mg_data["training"].isin(trainings_3_mg)
            ].groupby("training")["work"].sum()
            total_2_mg = mg_data[
            mg_data["training"].isin(trainings_2_mg)
            ].groupby("training")["work"].sum()

            plot_history_series(ax, total_3_mg.index, total_3_mg.values, label="3 sets", cutoff=cutoff)
            plot_history_series(ax, total_2_mg.index, total_2_mg.values, label="2 sets", cutoff=cutoff)
            device_names = ", ".join(mg_to_names[mg])
            ax.set_title(f"{mg.capitalize()} ({device_names})")
            if col == 0:
                ax.set_ylabel("Work / kWs")
            ax.legend(loc='lower right')
            idx += 1

        explain_peak_markers(fig)
        mark_year_boundaries(axes)
        save_figure(fig, output_folder, "work_muscle_group.png", dpi=300)


def plot_strength(context):
    """Plot loads, repetitions, and work from the shared session history."""
    output_folder = context.output
    devices = context.devices
    # IDs für die Plots
    ids = [22, 17, 5, 6, 11, 19, 21, 10, 14, 13, 15, 12]

    # Datenaufbereitung (wie bisher)
    # Daten aus Premium-Stats zusammenfassen
    data = context.strength_data
    if data.empty:
        print("No premium exercise records available; skipping strength plots.")
        return

    cutoff = context.cutoff

    data_training_accumulated = pd.DataFrame(columns=["training", "sets", "device", "duration", "moves", "concentric", "eccentric", "work"])
    for training in data["training"].unique():
        data_training = data[data["training"] == training]
        for device in data_training["device"].unique():
            data_device = data_training[data_training["device"] == device]
            data_training_accumulated.loc[len(data_training_accumulated)] = [
                training,
                data_device.shape[0],
                device,
                data_device["duration"].sum(),
                data_device["moves"].sum(),
                data_device["concentric"].mean(),
                data_device["eccentric"].mean(),
                data_device["work"].sum(),
            ]

    history = data
    if cutoff is not None:
        data = data[data["training"] >= cutoff]

    if data.empty:
        print("No strength data in the selected range; skipping strength plots.")
        return

    # Aufruf der einzelnen Plot-Funktionen
    plot_weights(history, devices, output_folder, ids, cutoff=cutoff)
    plot_delta_percentage(history, devices, output_folder, ids, cutoff=cutoff)
    plot_reps(history, devices, output_folder, ids, cutoff=cutoff)
    plot_work_individual(data_training_accumulated, devices, output_folder, ids, cutoff=cutoff)
    plot_work_muscle_group(data_training_accumulated, devices, output_folder, ids, cutoff=cutoff)

def date_axis(ax):
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.grid(alpha=0.18)


def save_figure(fig, output, filename, dpi=160):
    """Save, close, and report every chart consistently."""
    fig.savefig(output / filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  {filename}")


def save(fig, output, filename, note=None):
    if note:
        fig.text(0.5, 0.012, note, ha="center", va="bottom", fontsize=8, color="0.4")
    fig.tight_layout(rect=(0, 0.045 if note else 0, 1, 0.96))
    save_figure(fig, output, filename)


def plot_bioage(folder, output, cutoff):
    rows = read_json(folder / "bioage_history.json", []) or []
    # Daily current-value snapshots continue the series beyond the API history.
    for path in sorted((folder / "snapshots").glob("*/bioage.json")):
        snapshot = read_json(path, {})
        value = snapshot.get("data")
        if value and value.get("t"):
            rows = [*rows, value]
    current = read_json(folder / "bioage.json")
    if current and current.get("t"):
        rows = [*rows, current]
    rows = {r["t"]: r for r in rows if r.get("t") and (cutoff is None or timestamp(r["t"]) >= cutoff)}
    if not rows:
        return
    rows = [rows[t] for t in sorted(rows)]
    times = [timestamp(r["t"]) for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(times, [r.get("age") for r in rows], "--", color="0.4", label="Chronological age")
    for key, label in [("bioAge", "Overall"), ("muscleAge", "Muscle"),
                       ("cardioAge", "Cardio"), ("mobilityAge", "Mobility")]:
        values = [r.get(key) if key == "bioAge" else (r.get(key) or {}).get("bioAge") for r in rows]
        if any(v is not None for v in values):
            axes[0].plot(times, values, marker=".", label=label)
    for group, label in GROUPS.items():
        values = [(r.get("muscleAge") or {}).get("muscleGroups", {}).get(group, {}).get("bioAge") for r in rows]
        gap = [v - r["age"] if v is not None and r.get("age") is not None else np.nan
               for r, v in zip(rows, values)]
        axes[1].plot(times, gap, marker=".", label=label)
    axes[0].set_ylabel("Age / years")
    axes[1].set_ylabel("Muscle-group age − actual age / years")
    axes[1].axhline(0, color="0.4", linestyle="--", linewidth=1)
    for ax in axes:
        date_axis(ax)
        ax.legend(ncol=3, fontsize=8)
    fig.suptitle("Milon biological-age estimates")
    save(fig, output, "bioage.png", "Milon estimates; missing cardio/mobility values are omitted. Lower age gaps mean younger estimates.")


def aggregate_frame(rows, cutoff):
    frame = pd.DataFrame(rows)
    if frame.empty or "datepart" not in frame:
        return pd.DataFrame()
    # These are UTC calendar-bucket labels, not local event timestamps. Converting
    # them to local time makes monthly reindexing lose winter/summer rows at DST.
    frame["date"] = pd.to_datetime(frame["datepart"], utc=True).dt.tz_localize(None)
    # Keep the complete bucket containing the cutoff; make this explicit in captions.
    if cutoff is not None:
        start = cutoff.tz_localize(None).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        frame = frame[frame["date"] >= start]
    return frame.sort_values("date")


def plot_progress(folder, output, cutoff, devices):
    response = read_json(folder / "progress.json", {}) or {}
    frame = aggregate_frame(response.get("nodegroups_months", []), cutoff)
    if not frame.empty:
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        for kind, data in frame.groupby("nodegroup_type"):
            for ax, field in zip(axes, ["nodegroup_count", "aw", "aws"]):
                ax.plot(data["date"], data[field], marker="o", markersize=3, label=kind)
        for ax, label in zip(axes, ["Circuit occurrences", "Average load / kg", "Average work / kWs"]):
            ax.set_ylabel(label)
            date_axis(ax)
            ax.legend(fontsize=8)
        fig.suptitle("Monthly circuit progress")
        save(fig, output, "progress.png", "API monthly aggregates; includes the cutoff month. Current month may be incomplete. Counts are circuits, not visits.")

    response = read_json(folder / "progress-devices.json", {}) or {}
    frame = aggregate_frame(response.get("devices_months", []), cutoff)
    if frame.empty:
        return
    # Preserve circuit type rather than averaging incompatible training modes.
    frame["label"] = frame.apply(lambda r: f"{devices.get(str(r['id']), {}).get('name', r['id'])} ({r['nodegroup_type']})", axis=1)
    table = frame.pivot_table(index="label", columns="date", values="aw", aggfunc="mean")
    months = pd.date_range(table.columns.min(), table.columns.max(), freq="MS")
    table = table.reindex(columns=months)
    baseline = table.apply(lambda row: row.dropna().iloc[0] if row.notna().any() else np.nan, axis=1).replace(0, np.nan)
    change = table.div(baseline, axis=0).sub(1).mul(100)
    finite = change.to_numpy()[np.isfinite(change.to_numpy())]
    if not finite.size:
        return
    limit = max(10, float(np.max(np.abs(finite))))
    fig, ax = plt.subplots(figsize=(max(11, len(months) * .38), max(5, len(table) * .32)))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#e8e8e8")
    im = ax.imshow(change, aspect="auto", cmap=cmap, vmin=-limit, vmax=limit)
    ax.set_yticks(range(len(table)), table.index, fontsize=8)
    ax.set_xticks(range(len(months)), [d.strftime("%Y-%m") for d in months], rotation=60, ha="right", fontsize=8)
    fig.colorbar(im, ax=ax, label="Average load change / %")
    fig.suptitle("Monthly device load relative to first available month in range")
    save(fig, output, "device_progress.png", "Gray: no data. Includes the cutoff month; each device uses its own baseline. Current month may be incomplete.")


def latest_snapshot(folder, name):
    paths = sorted((folder / "snapshots").glob(f"*/{name}.json"))
    if not paths:
        return read_json(folder / f"{name}.json", {}), "capture date unknown"
    value = read_json(paths[-1], {})
    return value.get("data", {}), value.get("captured_at", "unknown")[:10]


def plot_muscles(folder, output):
    response, captured = latest_snapshot(folder, "muscles")
    premium = (response or {}).get("premium", {})
    groups = premium.get("groups", {})
    balances = premium.get("balance", {})
    if not groups and not balances:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, (kind, values) in enumerate(groups.items()):
        total = sum(values.get("w" + g, 0) or 0 for g in GROUPS)
        if total:
            width = .8 / len(groups)
            axes[0].bar(np.arange(len(GROUPS)) + (i - (len(groups) - 1) / 2) * width,
                        [(values.get("w" + g, 0) or 0) / total * 100 for g in GROUPS], width=width, label=kind)
    axes[0].set_xticks(range(len(GROUPS)), GROUPS.values())
    axes[0].set_ylabel("Share of group values / %")
    axes[0].set_title("Muscle-group distribution")
    if axes[0].patches:
        axes[0].legend()
    labels = {"back": ("Back extension", "Abdominal crunch"),
              "arm": ("Chest press", "Seated row"), "leg": ("Leg extension", "Leg curl")}
    pairs = [(kind, key, pair) for kind, data in balances.items() for key, pair in data.items()
             if pair.get("a") is not None and pair.get("b") is not None]
    for i, (kind, key, pair) in enumerate(pairs):
        a, b = pair["a"], pair["b"]
        axes[1].barh(i, a, color="#3978a8")
        axes[1].barh(i, b, left=a, color="#e69f45")
        axes[1].text(a / 2, i, f"{a:g}%", ha="center", va="center", color="white")
        axes[1].text(a + b / 2, i, f"{b:g}%", ha="center", va="center")
    axes[1].set_yticks(range(len(pairs)), [f"{labels.get(key, (key, key))[0]} /\n{labels.get(key, (key, key))[1]} ({kind})" for kind, key, _ in pairs], fontsize=8)
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Reported balance / %")
    axes[1].set_title("Available opposing muscle pairs")
    fig.suptitle(f"Muscle snapshot · downloaded {captured}")
    save(fig, output, "muscle_balance.png", "Current snapshot, independent of --range. API averaging window is unspecified; values do not measure muscle mass.")


def plot_plans(folder, output, devices, records, cutoff):
    # Include plans actually used in the selected range; retain their snapshot dates.
    selected = {str(r["training"].get("plan_id")) for r in records
                if cutoff is None or timestamp(r["training"]["t"]) >= cutoff}
    plans = [read_json(path) for path in (folder / "plans").glob("*.json") if path.stem in selected]
    plans = sorted([p for p in plans if p and p.get("items")], key=lambda p: p.get("startdate", 0))
    rows = []
    for plan in plans:
        name = f"{plan.get('name', 'Plan')} · {plan['id']}"
        if plan.get("startdate"):
            name += "\n" + pd.to_datetime(plan["startdate"], unit="ms", utc=True).tz_convert(TZ).strftime("%Y-%m-%d")
        seen = set()
        for item in plan["items"]:
            for element in item.get("elements", []):
                for node in element.get("nodes", []):
                    for power in node.get("power", []):
                        settings = power.get("jsonPower", {})
                        device = node["devicetype_id"]
                        # Repeated circuit entries commonly share exactly the same settings.
                        key = (device, json.dumps(settings, sort_keys=True))
                        if key in seen:
                            continue
                        seen.add(key)
                        rows.append({"plan": name, "device": devices.get(str(device), {}).get("name", str(device)),
                                     "Concentric / kg": settings.get("WEIGHT", {}).get("value"),
                                     "Eccentric / kg": settings.get("DELTAWEIGHT", {}).get("value")})
    if not rows:
        return
    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(plans) * 2.2), 7))
    order = list(dict.fromkeys(frame["plan"]))
    for ax, field in zip(axes, ["Concentric / kg", "Eccentric / kg"]):
        table = frame.pivot_table(index="device", columns="plan", values=field, aggfunc="mean").reindex(columns=order)
        if table.empty:
            ax.set_visible(False)
            continue
        im = ax.imshow(table, aspect="auto", cmap="Blues")
        ax.set_yticks(range(len(table)), table.index, fontsize=8)
        ax.set_xticks(range(len(order)), order, rotation=35, ha="right", fontsize=8)
        ax.set_title(field)
        for i in range(len(table)):
            for j in range(len(order)):
                value = table.iloc[i, j]
                if pd.notna(value):
                    ax.text(j, i, f"{value:g}", ha="center", va="center", fontsize=8,
                            color="white" if value > table.max().max() * .65 else "black")
        fig.colorbar(im, ax=ax, shrink=.7)
    fig.suptitle("Stored plan settings · plans used in selected range")
    save(fig, output, "plan_settings.png", "Snapshots, not setting-change history. Means across distinct settings per device; dates are plan start dates.")


def plot_sessions(records, output, cutoff, devices):
    rows, changes, heart = [], [], []
    for record in records:
        training = record["training"]
        time = timestamp(training["t"])
        if cutoff is not None and time < cutoff:
            continue
        entries = sorted(record.get("devices", []), key=lambda d: d["t"])
        if not entries:
            continue
        active = sum(d.get("d", 0) or 0 for d in entries)
        span = max(d["t"] + (d.get("d", 0) or 0) for d in entries) - min(d["t"] for d in entries)
        rows.append({"time": time, "active": active / 60, "span": span / 60,
                     "plan": str(training.get("plan_id")), "name": training.get("plan_name", "")})
        if training.get("ahr", 0) > 0 or training.get("maxhr", 0) > 0:
            heart.append({"time": time, "Average": training.get("ahr") or np.nan,
                          "Maximum": training.get("maxhr") or np.nan})
        by_device = defaultdict(list)
        for entry in entries:
            if entry.get("moves") is not None:
                by_device[entry["id"]].append(entry)
        for device, sets in by_device.items():
            first, last = sets[0], sets[-1]
            if len(sets) < 2 or first.get("moves", 0) <= 0:
                continue
            # Compare only matching loads/durations, with duration tolerance for logging jitter.
            if any(s.get("aw") != first.get("aw") or s.get("adw") != first.get("adw")
                   or abs((s.get("d") or 0) - (first.get("d") or 0)) > 1 for s in sets):
                continue
            changes.append({"time": time, "device": device, "sets": len(sets),
                            "change": (last["moves"] / first["moves"] - 1) * 100})
    if not rows:
        return
    frame = pd.DataFrame(rows).set_index("time").sort_index()
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    counts = frame["active"].resample("W-MON", closed="left", label="left").count()
    end = pd.Timestamp.now(tz=TZ).normalize()
    start = cutoff if cutoff is not None else frame.index.min()
    week_start = start.normalize() - pd.DateOffset(days=start.weekday())
    counts = counts.reindex(pd.date_range(week_start, end, freq="W-MON"), fill_value=0)
    axes[0].bar(counts.index, counts, width=5, color="#3978a8")
    axes[0].set_ylabel("Recorded sessions / week")
    axes[1].plot(frame.index, frame["active"], ".-", label="Active device time")
    axes[1].plot(frame.index, frame["span"], ".-", alpha=.65, label="First start to last finish")
    axes[1].set_ylabel("Minutes")
    axes[1].legend(fontsize=8)
    # Compute gaps over full history so the first visible point has its preceding session.
    times = pd.Series([timestamp(r["training"]["t"]) for r in records])
    gaps = pd.Series(times.diff().dt.total_seconds().to_numpy() / 86400, index=pd.DatetimeIndex(times))
    if cutoff is not None:
        gaps = gaps[gaps.index >= cutoff]
    axes[2].plot(gaps.index, gaps, ".-", color="#a46a26")
    axes[2].set_ylabel("Days since prior session")
    transitions = frame[frame["plan"].ne(frame["plan"].shift())]
    for time, row in transitions.iloc[1:].iterrows():
        for ax in axes:
            ax.axvline(time, color="0.5", linestyle=":", linewidth=.8)
        axes[1].annotate(row["name"], (time, 1), xycoords=("data", "axes fraction"), fontsize=8, rotation=90, va="top")
    for ax in axes:
        date_axis(ax)
    fig.suptitle("Training consistency and session structure")
    save(fig, output, "training_consistency.png", "Dotted lines: observed plan transitions. Gaps include rest and transitions; first/last weeks may be partial.")

    if changes:
        change_frame = pd.DataFrame(changes)
        ids = sorted(change_frame["device"].unique())
        fig, axes = plt.subplots((len(ids) + 3) // 4, 4, figsize=(16, max(4, ((len(ids) + 3) // 4) * 2.8)), squeeze=False)
        for ax in axes.flat:
            ax.set_visible(False)
        for ax, device in zip(axes.flat, ids):
            ax.set_visible(True)
            for count, data in change_frame[change_frame["device"] == device].groupby("sets"):
                ax.plot(data["time"], data["change"], ".", label=f"{count} sets", alpha=.75, color=f"C{(int(count) - 2) % 10}")
            ax.axhline(0, color="0.4", linewidth=.7)
            ax.set_title(devices.get(str(device), {}).get("name", str(device)), fontsize=9)
            ax.set_ylabel("Rep change / %", fontsize=8)
            date_axis(ax)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7)
        fig.suptitle("Last versus first set · repetitions")
        save(fig, output, "set_changes.png", "Matching loads and durations (±1 second) only; grouped by number of device records. Change does not by itself measure fatigue.")

    if heart:
        fig, ax = plt.subplots(figsize=(12, 4))
        for field in ("Average", "Maximum"):
            ax.scatter([r["time"] for r in heart], [r[field] for r in heart], label=field, s=22)
        ax.set_ylabel("Recorded heart rate / bpm")
        ax.legend()
        date_axis(ax)
        ax.set_xlim(start, end + pd.Timedelta(days=1))
        fig.suptitle("Available session heart-rate measurements")
        save(fig, output, "heart_rate.png", "Sparse measurements only; no interpolation. Profile heartrate_max and zero/missing readings are excluded.")


# Plot groups are peers; grouping reflects subject matter, not API age/source.
PLOT_GROUPS = {
    "strength": (lambda c: plot_strength(c),
                 ("weights.png", "delta percentage.png", "reps.png", "work_individual.png", "work_muscle_group.png")),
    "bioage": (lambda c: plot_bioage(c.stats, c.output, c.cutoff), ("bioage.png",)),
    "progress": (lambda c: plot_progress(c.stats, c.output, c.cutoff, c.devices),
                 ("progress.png", "device_progress.png")),
    "muscles": (lambda c: plot_muscles(c.stats, c.output), ("muscle_balance.png",)),
    "plans": (lambda c: plot_plans(c.stats, c.output, c.devices, c.records, c.cutoff), ("plan_settings.png",)),
    "sessions": (lambda c: plot_sessions(c.records, c.output, c.cutoff, c.devices),
                 ("training_consistency.png", "set_changes.png", "heart_rate.png")),
}


def plot_all(context, groups=None):
    selected = list(dict.fromkeys(groups)) if groups is not None else list(PLOT_GROUPS)
    unknown = set(selected) - PLOT_GROUPS.keys()
    if unknown:
        raise ValueError(f"Unknown plot groups: {', '.join(sorted(unknown))}")
    with plt.rc_context({"axes.spines.top": False, "axes.spines.right": False}):
        for name in selected:
            render, filenames = PLOT_GROUPS[name]
            # Same no-data handling for every selected group; unselected files stay intact.
            for filename in filenames:
                (context.output / filename).unlink(missing_ok=True)
            print(f"Plotting {name}:")
            render(context)

import requests
import json
import os
from getpass import getpass
import datetime
import pathlib

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import defaultdict

HOST = 'https://www.milonme.com'

DATA_FOLDER = pathlib.Path("data")
SESSION_FILE = DATA_FOLDER / 'session.json'

GRAPH_FOLDER = pathlib.Path("graphs")

rs = None
ms = None

def perform_login():
    print("Login to MilonMe")
    email = input("E-Mail: "),
    password = getpass("Password: ")

    rs = requests.Session()
    
    try:
        response = rs.post(
            f"{HOST}/api/user/login",
            data={'email': email, 'password': password, 'long_session': 0},
            headers={'x-api-key': 'v1uCCMWOFj8mbTdtkO7ia76K3h76tuvb2lOrL8RF'}
            )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Login error: {e}")
        return
    
    del password
    
    session = {
        'session': response.json(),
        'cookies': rs.cookies.get_dict()
    }

    os.makedirs(SESSION_FILE.parent, exist_ok=True)
    with open(SESSION_FILE, 'w') as f:
        json.dump(session, f, indent=4)

def load_session():
    global rs, ms
    try:
        with open(SESSION_FILE, 'r') as f:
            session_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Session file not found or corrupted, please log in")
        rs = None
        ms = None
        return
        
    ms = session_data.get('session', None)
    
    rs = requests.Session()
    rs.cookies.update(session_data['cookies'] if 'cookies' in session_data else {})

    response = rs.get(f'{HOST}/api/user/session')
    if response.status_code != 200:
        print("Session expired or invalid, please log in again")
        rs = None
        ms = None

def delete_session():
    global rs, ms

    if rs is None:
        load_session()
    
    if rs is not None:
        rs.delete(f'{HOST}/api/user/session')
        rs = None

    if os.path.exists(SESSION_FILE):
        os.remove(SESSION_FILE)
    ms = None

def establish_session():
    while True:
        load_session()
        if rs is not None:
            break
        perform_login()

def load_stats_home():
    studio_id = ms['d']['studios']
    user_id = ms['id']
    response = rs.get(f'{HOST}/api/user/stats/home/{studio_id}/{user_id}')
    response.raise_for_status()

    output_folder = DATA_FOLDER / user_id / 'stats'
    os.makedirs(output_folder, exist_ok=True)
    with open(output_folder / 'home.json', 'w') as f:
        f.write(response.text)

    home = response.json()
    print(f"Welcome back {home['profile']['firstname']} {home['profile']['lastname']}!")
    print(f"Studio: {home['studio']['studioname']}")

def load_devices():
    r = rs.get(f'{HOST}/api/devices/en_US')
    r.raise_for_status()

    os.makedirs(DATA_FOLDER, exist_ok=True)
    with open(DATA_FOLDER / 'devices.json', 'w') as f:
        f.write(r.text)

def load_stats_premium():
    studio_id = ms['d']['studios']
    user_id = ms['id']

    output_folder = DATA_FOLDER / user_id / 'stats' / 'premium'
    os.makedirs(output_folder, exist_ok=True)

    now = datetime.datetime.now()
    y = now.year % 100
    m = now.month

    # Get the last year and month that was already downloaded
    files = sorted(output_folder.glob('*.json'))
    if files:
        last_file = files[-1]
        last_yymm = os.path.basename(last_file)[:4]
    else:
        last_yymm = None

    print('Downloading premium stats for the last 12 months:', end='', flush=True)
    for i in range(13):
        yymm = f'{y:02}{m:02}'
        print(f" {yymm}", end='', flush=True)

        response = rs.get(f'{HOST}/api/user/stats/premium/{studio_id}/{user_id}/{yymm}')
        response.raise_for_status()

        with open(output_folder / f'{yymm}.json', 'w') as f:
            f.write(response.text)

        if last_yymm and yymm == last_yymm:
            print('\nSkipping previous months since they were already fully downloaded')
            break
        
        # decrement month
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    print()

def plot_weights(data, devices, output_folder, ids):
    """Plottet Eccentric und Concentric als Gewichtsdarstellung und speichert weights.png"""
    fig, axes = plt.subplots(nrows=3, ncols=4, sharex=True)
    fig.subplots_adjust(left=0.04, bottom=0.05, right=0.98, top=0.965, wspace=0.15, hspace=0.28)
    fig.set_size_inches(16, 9)
    for ax in axes.flat:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    for idx, device_id in enumerate(ids):
        row = idx // axes.shape[1]
        col = idx % axes.shape[1]
        ax = axes[row, col]
        # Filter für Gerät
        data_id = data[data["device"] == device_id]
        ax.plot(data_id["time"], data_id["eccentric"], label="Eccentric")
        ax.plot(data_id["time"], data_id["concentric"], label="Concentric")
        ax.set_title(devices[str(device_id)]["name"])
        if row == axes.shape[0] - 1:
            ax.set_xlabel("Time")
        if col == 0:
            ax.set_ylabel("Weight / kg")
        ax.legend(loc='lower right')
    plt.savefig(output_folder / "weights.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_delta_percentage(data, devices, output_folder, ids):
    """Plottet den prozentualen Unterschied zwischen Eccentric und Concentric und speichert delta percentage.png"""
    fig, axes = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.04, bottom=0.05, right=0.98, top=0.965, wspace=0.15, hspace=0.28)
    fig.set_size_inches(16, 9)
    for ax in axes.flat:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    for idx, device_id in enumerate(ids):
        row = idx // axes.shape[1]
        col = idx % axes.shape[1]
        ax = axes[row, col]
        data_id = data[data["device"] == device_id]
        ax.axhline(y=20, color='r', linestyle='--', label="20%")
        percentage = (data_id["eccentric"] / data_id["concentric"] - 1) * 100
        ax.plot(data_id["time"], percentage, label="delta %")
        ax.set_title(devices[str(device_id)]["name"])
        if row == axes.shape[0] - 1:
            ax.set_xlabel("Time")
        if col == 0:
            ax.set_ylabel("Percentage")
        ax.legend(loc='lower right')
    plt.savefig(output_folder / "delta percentage.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_reps(data, devices, output_folder, ids):
    """Plottet Wiederholungen (reps) und speichert reps.png"""
    fig, axes = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.04, bottom=0.05, right=0.98, top=0.965, wspace=0.15, hspace=0.28)
    fig.set_size_inches(16, 9)
    for ax in axes.flat:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    for idx, device_id in enumerate(ids):
        row = idx // axes.shape[1]
        col = idx % axes.shape[1]
        ax = axes[row, col]
        data_id = data[data["device"] == device_id]
        ax.axhline(y=8, color='r', linestyle='--', label="8")
        ax.plot(data_id["time"], data_id["moves"], label="reps")
        ax.set_title(devices[str(device_id)]["name"])
        if row == axes.shape[0] - 1:
            ax.set_xlabel("Time")
        if col == 0:
            ax.set_ylabel("Repetitions")
        ax.legend(loc='lower right')
    plt.savefig(output_folder / "reps.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_work_individual(data_training_accumulated, devices, output_folder, ids):
    fig, axes = plt.subplots(nrows=3, ncols=4, sharex=True)
    fig.subplots_adjust(left=0.04, bottom=0.05, right=0.98, top=0.965, wspace=0.15, hspace=0.28)
    fig.set_size_inches(16, 9)
    for ax in axes.flat:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    for idx, device_id in enumerate(ids):
        row = idx // axes.shape[1]
        col = idx % axes.shape[1]
        ax = axes[row, col]
        data_id_accumulated_3 = data_training_accumulated[(data_training_accumulated["device"] == device_id) & (data_training_accumulated["sets"] == 3)]
        ax.plot(data_id_accumulated_3["training"], data_id_accumulated_3["work"], label="3 sets")
        data_id_accumulated_2 = data_training_accumulated[(data_training_accumulated["device"] == device_id) & (data_training_accumulated["sets"] == 2)]
        ax.plot(data_id_accumulated_2["training"], data_id_accumulated_2["work"], label="2 sets")
        ax.set_title(devices[str(device_id)]["name"])
        if row == axes.shape[0] - 1:
            ax.set_xlabel("Time")
        if col == 0:
            ax.set_ylabel("Work / kWs")
        ax.legend(loc='lower right')
    plt.savefig(output_folder / "work_individual.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_work_muscle_group(data_training_accumulated, devices, output_folder, ids):
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
        fig.subplots_adjust(left=0.04, bottom=0.05, right=0.98, top=0.965, wspace=0.15, hspace=0.28)
        fig.set_size_inches(16, 9)

        for ax in axes.flat:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
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
        ax.plot(total_3.index, total_3.values, label="3 sets")
        ax.plot(total_2.index, total_2.values, label="2 sets")
        ax.set_title("Total Work (All Devices)")
        if row == axes.shape[0] - 1:
            ax.set_xlabel("Time")
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
                ax.set_xlabel("Time")
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

            ax.plot(total_3_mg.index, total_3_mg.values, label="3 sets")
            ax.plot(total_2_mg.index, total_2_mg.values, label="2 sets")
            device_names = ", ".join(mg_to_names[mg])
            ax.set_title(f"{mg.capitalize()} ({device_names})")
            if row == axes.shape[0] - 1:
                ax.set_xlabel("Time")
            if col == 0:
                ax.set_ylabel("Work / kWs")
            ax.legend(loc='lower right')
            idx += 1

        plt.savefig(output_folder / "work_muscle_group.png", dpi=300, bbox_inches='tight')
        plt.close(fig)


def format_delta(delta, components=3):
    days = delta.days
    total_seconds = delta.seconds
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)

    parts = [
        ("days",    days),
        ("hours",   hours),
        ("minutes", minutes),
        ("seconds", seconds),
    ]
    while len(parts) > 1 and parts[0][1] == 0:
        parts.pop(0)
    parts = parts[:components]
    return ", ".join(f"{value} {name}" for name, value in parts)

def plot_all():
    """Führt alle Plot-Funktionen aus."""
    user_id = ms['id']
    output_folder = GRAPH_FOLDER / user_id
    os.makedirs(output_folder, exist_ok=True)
    # Geräte laden
    with open(DATA_FOLDER / "devices.json", 'r') as file:
        devices = json.load(file)
    # IDs für die Plots
    ids = [22, 17, 5, 6, 11, 19, 21, 10, 14, 13, 15, 12]

    # Datenaufbereitung (wie bisher)
    # Daten aus Premium-Stats zusammenfassen
    data = pd.DataFrame(columns=["training", "set", "device", "time", "duration", "moves", "concentric", "eccentric", "work"])
    premium_stats_folder = DATA_FOLDER / ms['id'] / 'stats' / 'premium'
    for file_path in sorted(premium_stats_folder.glob("*.json")): 
        # print(f"Processing {file_path}")
        with open(file_path, 'r') as file:
            jsondata = json.load(file)
        for training in jsondata["stats"]:
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
    data.sort_values(by=["time"], inplace=True)
    data = data[data["time"] > data["time"].min() + 3 * 60 * 60]
    data = data[~((data["device"] == 15) & (data["concentric"] == 12) & (data["eccentric"] == 14)  & (data["duration"] == 10)  & (data["moves"] == 1))]
    
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
    # Konvertierung in DateTime
    local_tz = datetime.datetime.now().astimezone().tzinfo
    data["time"] = pd.to_datetime(data["time"], unit='s', utc=True).dt.tz_convert(local_tz)
    data["training"] = pd.to_datetime(data["training"], unit='s', utc=True).dt.tz_convert(local_tz)
    data_training_accumulated["training"] = pd.to_datetime(data_training_accumulated["training"], unit='s', utc=True).dt.tz_convert(local_tz)

    num_trainings = len(data["training"].unique())
    print(f"Total number of trainings: {num_trainings}")
    print(f"Total reps: {data['moves'].sum()}")
    now = datetime.datetime.now(local_tz)
    first_training = data["training"].min()
    print(f"First training: {first_training} ({format_delta(now - first_training)} ago)")
    last_training = data["training"].max()
    print(f"Last training: {last_training} ({format_delta(now - last_training)} ago)")
    print(f"Average time between trainings: {format_delta((last_training - first_training) / num_trainings)}")
    print(f"Total work: {data['work'].sum() / 3600:.2f} kWh")
    total_duration = datetime.timedelta(seconds=int(data["duration"].sum()))
    print(f"Total active time: {format_delta(total_duration)}")
    print(f"Active time per training: {format_delta(total_duration / num_trainings)}")

    # Aufruf der einzelnen Plot-Funktionen
    plot_weights(data, devices, output_folder, ids)
    plot_delta_percentage(data, devices, output_folder, ids)
    plot_reps(data, devices, output_folder, ids)
    plot_work_individual(data_training_accumulated, devices, output_folder, ids)
    plot_work_muscle_group(data_training_accumulated, devices, output_folder, ids)

def main():
    # delete_session()
    establish_session()
    load_stats_home()
    load_devices()
    load_stats_premium()
    plot_all()

if __name__ == "__main__":
    main()

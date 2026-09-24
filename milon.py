import argparse
import re
import requests
import json
import os
from getpass import getpass
import pathlib

from milon_api import HOST, download_all
from training_plots import PLOT_GROUPS, plot_all
from training_data import PlotContext
from training_stats import print_training_summary


DATA_FOLDER = pathlib.Path("data")
SESSION_FILE = DATA_FOLDER / 'session.json'

GRAPH_FOLDER = pathlib.Path("graphs")

rs = None
ms = None

def perform_login():
    print("Login to MilonMe")
    email = input("E-Mail: ")
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

def parse_range(value):
    if value == "all":
        return 0
    match = re.fullmatch(r"([1-9][0-9]*)([dwy])", value)
    if match is None:
        raise argparse.ArgumentTypeError("use a positive duration such as 365d, 4w, or 1y, or 'all'")
    amount, unit = match.groups()
    return int(amount) * {"d": 1, "w": 7, "y": 365}[unit]


def main():
    global ms
    parser = argparse.ArgumentParser(description="Download and plot Milon Me training data.")
    parser.add_argument(
        "--range", type=parse_range, default="1y", metavar="RANGE",
        help="range for stats and plots: e.g. 365d, 4w, 1y, or all (default: 1y; a year is 365 days)",
    )
    parser.add_argument("--offline", action="store_true", help="plot cached data without logging in or downloading")
    parser.add_argument("--plots", nargs="+", choices=tuple(PLOT_GROUPS), metavar="GROUP",
                        help="plot selected groups: " + ", ".join(PLOT_GROUPS) + " (default: all)")
    args = parser.parse_args()
    if args.offline:
        try:
            with open(SESSION_FILE) as file:
                ms = json.load(file)["session"]
        except (OSError, ValueError, KeyError):
            parser.error("offline mode requires a saved session identifying the user")
    else:
        establish_session()
        download_all(rs, ms["id"], ms["d"]["studios"], DATA_FOLDER)
    context = PlotContext.load(ms["id"], DATA_FOLDER, GRAPH_FOLDER, days=args.range)
    print_training_summary(context)
    plot_all(context, groups=args.plots)

if __name__ == "__main__":
    main()

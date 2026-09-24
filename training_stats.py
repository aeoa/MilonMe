"""Console reporting, independent of chart rendering."""

import datetime


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

def print_training_summary(context):
    """Report the selected range using the strength charts' shared data filters."""
    data = context.strength_data
    now = context.now
    days = context.days
    if context.cutoff is not None and not data.empty:
        data = data[data["training"] >= context.cutoff]
    print("Training summary:")
    num_trainings = len(data["training"].unique())
    if num_trainings == 0:
        period = f" in the last {days} days" if days > 0 else ""
        print(f"No trainings found{period}.")
        return
    print(f"Total number of trainings: {num_trainings}")
    print(f"Total reps: {data['moves'].sum()}")
    first_training = data["training"].min()
    print(f"First training: {first_training} ({format_delta(now - first_training)} ago)")
    last_training = data["training"].max()
    print(f"Last training: {last_training} ({format_delta(now - last_training)} ago)")
    print(f"Average time between trainings: {format_delta((last_training - first_training) / num_trainings)}")
    print(f"Total work: {data['work'].sum() / 3600:.2f} kWh")
    total_duration = datetime.timedelta(seconds=int(data["duration"].sum()))
    print(f"Total active time: {format_delta(total_duration)}")
    print(f"Active time per training: {format_delta(total_duration / num_trainings)}")


"""Plot cumulative pedestrian arrivals and video-relative train arrivals.

Run this script from the folder containing Pedestrian Data.xlsx and
Train Arrival.xlsx. Requires: pip install numpy matplotlib openpyxl

Outputs: Arrival_curves/<session>.png and .pdf, a multipage PDF, and a
session summary CSV. The Excel files are opened read-only and never saved.

The curve counts arrivals recorded on the first pedestrian worksheet,
including Walk arrivals and people whose cycle assignments need review.
Cycle labels and crossing-start times are not used. Each plot stops at the
last recorded pedestrian arrival; it does not assert the video ended there.
Train times come directly from Video Time, without a walking-time offset.
Monday South 8am-9am has no train markers because it was filmed separately.
"""

import argparse
import csv
import math
import re
from collections import defaultdict
from datetime import time, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from openpyxl import load_workbook


TRAIN_SHEETS = {"monday": "Monday Arrivals", "saturday": "Saturday Arrivals"}
TRAIN_EXCLUDED = {("monday", "south", "8am-9am")}
TRAIN_COLOUR = "#777777"
CROSSING_COLOURS = {"south": "tab:blue", "east": "tab:orange"}


def normalise_period(value):
    period = re.sub(r"\s+", "", str(value).lower())
    period = period.replace(":00", "").replace("\u2013", "-").replace("\u2014", "-")
    return {"2:55pm-4pm": "3pm-4pm"}.get(period, period)


def session_key(day, crossing, period):
    return str(day).strip().casefold(), str(crossing).strip().casefold(), normalise_period(period)


def seconds_from_video_time(value):
    """Numeric cells are seconds, NOT Excel fractions of a day."""
    if value is None or isinstance(value, bool):
        raise ValueError("missing or invalid video-relative time")
    if isinstance(value, timedelta):
        result = value.total_seconds()
    elif isinstance(value, time):
        result = value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1e6
    else:
        text = str(value).strip()
        try:
            result = float(text)
        except ValueError:
            sign = -1 if text.startswith("-") else 1
            parts = text.lstrip("+-").split(":")
            if len(parts) not in (2, 3):
                raise ValueError(f"invalid video-relative time: {value!r}")
            result = sign * sum(float(part) * 60 ** power
                                for power, part in enumerate(reversed(parts)))
    if not math.isfinite(result):
        raise ValueError(f"non-finite video-relative time: {value!r}")
    return result


def find_header(sheet, required):
    for row_number, row in enumerate(sheet.iter_rows(max_row=20, values_only=True), 1):
        columns = {str(value).strip(): index for index, value in enumerate(row) if value is not None}
        if set(required).issubset(columns):
            return row_number, columns
    raise ValueError(f"{sheet.title}: missing column headers {required}")


def load_pedestrians(path):
    sessions = defaultdict(list)
    seen_ids = set()
    book = load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = book.worksheets[0]
        header, columns = find_header(sheet, ["Pedestrian ID", "Day", "Crossing", "Time Period", "Arrival Time"])
        for number, row in enumerate(sheet.iter_rows(min_row=header + 1, values_only=True), header + 1):
            pedestrian_id = row[columns["Pedestrian ID"]]
            if pedestrian_id is None:
                continue  # Blank rows and summary/footer cells are not pedestrians.
            if pedestrian_id in seen_ids:
                raise ValueError(f"Duplicate pedestrian ID {pedestrian_id!r} at row {number}")
            seen_ids.add(pedestrian_id)
            labels = [row[columns[name]] for name in ["Day", "Crossing", "Time Period"]]
            if any(value is None or not str(value).strip() for value in labels):
                raise ValueError(f"Pedestrian row {number}: missing session label")
            try:
                seconds = seconds_from_video_time(row[columns["Arrival Time"]])
            except (ValueError, TypeError) as error:
                raise ValueError(f"Pedestrian row {number}: {error}") from error
            if seconds < 0:
                raise ValueError(f"Pedestrian row {number}: arrival precedes video start")
            sessions[session_key(*labels)].append(seconds)
    finally:
        book.close()
    if not sessions:
        raise ValueError("No pedestrian records found")
    return {key: np.sort(times) for key, times in sessions.items()}


def load_trains(path):
    trains = defaultdict(list)
    book = load_workbook(path, read_only=True, data_only=True)
    try:
        for day, sheet_name in TRAIN_SHEETS.items():
            sheet = book[sheet_name]
            header, columns = find_header(sheet, ["Requested window", "Direction", "Video Time"])
            for number, row in enumerate(sheet.iter_rows(min_row=header + 1, values_only=True), header + 1):
                direction = row[columns["Direction"]]
                if direction is None:
                    continue
                direction = re.sub(r"[\s\-\u2013\u2014]+", " ", str(direction).strip().casefold())
                label = {"city bound": "City-bound", "cronulla bound": "Cronulla-bound"}.get(direction)
                if label is None:
                    raise ValueError(f"{sheet_name}, row {number}: unknown direction {direction!r}")
                window = row[columns["Requested window"]]
                if window is None:
                    raise ValueError(f"{sheet_name}, row {number}: missing Requested window")
                try:
                    seconds = seconds_from_video_time(row[columns["Video Time"]])
                except (ValueError, TypeError) as error:
                    raise ValueError(f"{sheet_name}, row {number}: {error}. Save calculated cells in Excel first.") from error
                trains[(day, normalise_period(window))].append((seconds, label))
    finally:
        book.close()
    for events in trains.values():
        if len(events) != len(set(events)):
            raise ValueError("Duplicate train time/direction within a session")
        events.sort()
    return trains


def session_sort(key):
    day, crossing, period = key
    period_order = {"8am-9am": 0, "10am-11am": 1, "12pm-1pm": 2, "3pm-4pm": 3}
    return ({"monday": 0, "saturday": 1}.get(day, 2), period_order.get(period, 99), crossing)


def make_figure(key, arrivals, trains):
    day, crossing, period = key
    end_seconds = float(arrivals[-1])
    excluded = key in TRAIN_EXCLUDED
    train_key = (day, period)
    if not excluded and train_key not in trains:
        raise ValueError(f"No matching train session for {key}; check workbook labels")
    events = [] if excluded else [event for event in trains[train_key] if 0 <= event[0] <= end_seconds]
    colour = CROSSING_COLOURS.get(crossing, "black")
    fig, ax = plt.subplots(figsize=(10.8, 5.5))
    fig.subplots_adjust(left=0.10, right=0.77, bottom=0.17, top=0.85)
    times, counts = np.unique(arrivals, return_counts=True)
    ax.step(np.r_[0.0, times / 60], np.r_[0, np.cumsum(counts)], where="post",
            color=colour, linewidth=1.7, label="Pedestrian arrivals", zorder=3)
    ax.set_xlim(0, max(end_seconds / 60, 0.1))
    ax.set_ylim(0, max(len(arrivals) * 1.16, 5))
    ax.set_xlabel("Time since video start (minutes)")
    ax.set_ylabel("Cumulative pedestrian arrivals")
    ax.set_title(f"{day.title()} {crossing.title()}\n{period.replace('-', ' - ')}", fontsize=12, pad=12)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    previous_seconds = -math.inf
    previous_level = 0
    for seconds, label in events:
        position = seconds / 60
        ax.axvline(position, color=TRAIN_COLOUR, linestyle="--", linewidth=1, alpha=0.8, zorder=2)
        # Stagger labels on nearby trains and keep edge labels inside the plot.
        level = 1 - previous_level if seconds - previous_seconds < 0.08 * max(end_seconds, 1) else 0
        near_right = seconds > 0.96 * end_seconds
        ax.annotate(label, xy=(position, 0.98 - 0.28 * level), xycoords=ax.get_xaxis_transform(),
                    xytext=(-3 if near_right else 3, 0), textcoords="offset points", rotation=90,
                    ha="right" if near_right else "left", va="top", fontsize=8, color=TRAIN_COLOUR,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1})
        previous_seconds, previous_level = seconds, level
    handles = [Line2D([0], [0], color=colour, linewidth=1.7, label="Pedestrian arrivals")]
    if events:
        handles.append(Line2D([0], [0], color=TRAIN_COLOUR, linestyle="--", label="Train arrival"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
    if excluded:
        ax.text(0.02, 0.97, "Train markers omitted:\nseparate recording time", transform=ax.transAxes,
                ha="left", va="top", color=TRAIN_COLOUR, fontsize=9)
    fig.text(0.10, 0.055, f"{len(arrivals):,} recorded pedestrian arrivals. Curve ends at last recorded arrival.", fontsize=9)
    return fig, {
        "Day": day.title(), "Crossing": crossing.title(), "Time Period": period,
        "Pedestrians": len(arrivals), "First arrival (video s)": float(arrivals[0]),
        "Last arrival (video s)": end_seconds, "Train lines shown": len(events),
        "Train markers excluded": excluded,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Folder containing both source workbooks; defaults to this script's folder")
    parser.add_argument("--output-dir", type=Path, help="Defaults to <data-dir>/Arrival_curves")
    args = parser.parse_args()
    output = args.output_dir or args.data_dir / "Arrival_curves"
    sessions = load_pedestrians(args.data_dir / "Pedestrian Data.xlsx")
    trains = load_trains(args.data_dir / "Train Arrival.xlsx")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "savefig.dpi": 300})
    summary = []
    with PdfPages(output / "All_sessions_arrival_curves.pdf") as document:
        for key in sorted(sessions, key=session_sort):
            fig, row = make_figure(key, sessions[key], trains)
            filename = "_".join(key).replace("-", "_to_") + "_arrival_curve"
            fig.savefig(output / f"{filename}.png")
            fig.savefig(output / f"{filename}.pdf")
            document.savefig(fig)
            plt.close(fig)
            summary.append(row)
            print(f"{row['Day']} {row['Crossing']} {row['Time Period']}: "
                  f"{row['Pedestrians']} pedestrians, {row['Train lines shown']} train markers")
    with (output / "Session_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nSaved {len(summary)} session figures ({sum(row['Pedestrians'] for row in summary):,} pedestrians) to:\n{output.resolve()}")
    print("Times use the video origin. Monday South 8am-9am has no train markers.")
    print("Train lines use Video Time as supplied; no extra walking-time shift is applied.")


if __name__ == "__main__":
    main()

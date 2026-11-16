"""Complete replacement script for the FR/red crossing-start figure.

Replace the old script in full; do not append this to the previous code.
Time labels and day labels are drawn together in label_sessions(), once.

Run this script normally, or use:
    python "Crossing Phase Figure.py" "Pedestrian Data.xlsx" --no-show
    python "Crossing Phase Figure.py" --workbook PATH --output-dir FOLDER --no-show

Outputs: a vector PDF, a 300 dpi PNG and a CSV of counts/percentages.
The workbook is read without modification. All pedestrian records are included;
cycle-review flags do not exclude records from this phase-based description.
"""

from pathlib import Path
import argparse
import math

import numpy as np
import pandas as pd


DEFAULT_WORKBOOK = Path(
    r"C:\Users\adamm\source\repos\Thesis Data Coding\Thesis Data Coding\Pedestrian Data.xlsx"
)
DAY_TYPES = ["Weekday", "Weekend"]
PERIODS = ["Morning", "Lunch", "Afternoon"]
CROSSINGS = ["South", "East"]
GROUPS = [(day, period) for day in DAY_TYPES for period in PERIODS]


def label_sessions(ax, data, positions):
    """Draw each time range and each day heading exactly once."""
    time_labels = []
    for day_type, period in GROUPS:
        recorded_times = sorted(data.loc[
            data["Day type"].eq(day_type) & data["Period"].eq(period),
            "Time Period",
        ].dropna().unique())
        time_labels.append("\n".join(recorded_times) if recorded_times else "No data")
    ax.set_xticks(positions, time_labels)
    ax.tick_params(axis="x", length=0, pad=9)
    for centre, label, left, right in [
        (1.15, "Monday", -0.5, 2.8),
        (5.1, "Saturday", 3.45, 6.75),
    ]:
        ax.plot(
            [left, right], [-0.12, -0.12], transform=ax.get_xaxis_transform(),
            color="#777777", linewidth=0.8, clip_on=False,
        )
        ax.text(
            centre, -0.17, label, transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=11, fontweight="bold",
        )


def summarise(workbook):
    data = pd.read_excel(workbook, sheet_name="Pedestrian Data")
    data.columns = data.columns.astype(str).str.strip()
    required = ["Pedestrian ID", "Day", "Crossing", "Time Period", "Crossing Start Phase"]
    missing = [column for column in required if column not in data]
    if missing:
        raise ValueError(f"Missing columns in Pedestrian Data: {missing}")
    data["Pedestrian ID"] = pd.to_numeric(data["Pedestrian ID"], errors="coerce")
    data = data.loc[data["Pedestrian ID"].notna()].copy()
    if data.empty or data["Pedestrian ID"].duplicated().any():
        raise ValueError("No pedestrian records, or duplicate Pedestrian IDs.")
    for column in ["Day", "Crossing", "Time Period", "Crossing Start Phase"]:
        data[column] = (
            data[column].astype("string").str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )
    data["Day"] = data["Day"].str.title()
    data["Crossing"] = data["Crossing"].str.title()
    data["Time Period"] = data["Time Period"].str.lower()
    data["Crossing Start Phase"] = data["Crossing Start Phase"].str.upper()
    days = {day: "Weekday" for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]}
    days.update({"Saturday": "Weekend", "Sunday": "Weekend"})
    periods = {"8am - 9am": "Morning", "10am - 11am": "Morning",
               "12pm - 1pm": "Lunch", "3pm - 4pm": "Afternoon", "5pm - 6pm": "Afternoon"}
    data["Day type"] = data["Day"].map(days)
    data["Period"] = data["Time Period"].map(periods)
    invalid = data[["Day type", "Period"]].isna().any(axis=1) | ~data["Crossing"].isin(CROSSINGS)
    if invalid.any():
        bad = data.loc[invalid, ["Pedestrian ID", "Day", "Crossing", "Time Period"]]
        raise ValueError(f"Unrecognised session labels; update the mappings if needed:\n{bad}")

    rows = []
    for day_type, period in GROUPS:
        for crossing in CROSSINGS:
            group = data.loc[
                data["Day type"].eq(day_type) & data["Period"].eq(period)
                & data["Crossing"].eq(crossing)
            ]
            denominator = len(group)
            known = group["Crossing Start Phase"].isin(["G", "FR", "R"])
            for phase in ["FR", "R"]:
                count = int(group["Crossing Start Phase"].eq(phase).sum())
                rows.append({
                    "Day type": day_type, "Period": period, "Crossing": crossing,
                    "Crossing start phase": phase, "Pedestrians": denominator,
                    "Known crossing phases": int(known.sum()),
                    "Unknown crossing phases": int((~known).sum()),
                    "Crossing starts": count,
                    "Percentage": 100 * count / denominator if denominator else np.nan,
                })
    summary = pd.DataFrame(rows)
    # Each crossing/day/period is represented twice, once per displayed phase.
    assert summary["Pedestrians"].sum() == 2 * len(data)
    return summary, data


def plot(summary, data, plt):
    from matplotlib.colors import to_rgb
    from matplotlib.ticker import MultipleLocator, PercentFormatter

    def tint(colour, white=0.52):
        return tuple((1 - white) * component + white for component in to_rgb(colour))

    style = {
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 9, "axes.linewidth": 0.8,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(12, 5.8))
        positions = np.array([0, 1.15, 2.3, 3.95, 5.1, 6.25])
        width = 0.19
        blue, orange = "#1f77b4", "#ff7f0e"
        series = [
            ("South", "FR", tint(blue)), ("South", "R", blue),
            ("East", "FR", tint(orange)), ("East", "R", orange),
        ]
        for index, (crossing, phase, colour) in enumerate(series):
            selected = summary.loc[
                summary["Crossing"].eq(crossing)
                & summary["Crossing start phase"].eq(phase)
            ].set_index(["Day type", "Period"]).reindex(GROUPS)
            values = selected["Percentage"].to_numpy(dtype=float)
            bars = ax.bar(
                positions + (index - 1.5) * width, values, width=width,
                color=colour, edgecolor="white", linewidth=0.4,
                label=f"{crossing} - {'Flashing Red' if phase == 'FR' else 'Red'}",
            )
            for bar, value in zip(bars, values):
                if np.isfinite(value):
                    ax.annotate(
                        f"{value:.1f}%", (bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", va="bottom", rotation=90, fontsize=8,
                    )
                else:
                    ax.annotate(
                        "n.a.", (bar.get_x() + bar.get_width() / 2, 0),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", va="bottom", rotation=90, fontsize=8, color="#666666",
                    )
        maximum = summary["Percentage"].max()
        if not np.isfinite(maximum):
            raise ValueError("No groups contain pedestrian data.")
        upper = max(20, math.ceil((maximum + 7) / 10) * 10)
        ax.set_ylim(0, upper)
        ax.set_xlim(-0.65, 6.9)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.yaxis.set_major_locator(MultipleLocator(5 if upper <= 50 else 10))
        ax.set_ylabel("Percentage of All Pedestrians (%)")
        label_sessions(ax, data, positions)
        ax.grid(axis="y", color="#b0b0b0", linewidth=0.8, alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.axvline(3.125, color="#bbbbbb", linewidth=0.8, zorder=1)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.025), ncol=4, frameon=False)
        ax.legend(
            loc="upper left", bbox_to_anchor=(1.02, 1.0), ncol=1,
            frameon=True, facecolor="white", edgecolor="#b0b0b0",
            framealpha=1.0, fancybox=True, borderpad=0.8,
            labelspacing=0.7, handlelength=1.5, borderaxespad=0,
        )
        fig.suptitle("Crossing starts during flashing red and red", fontsize=14, y=0.98)
        fig.subplots_adjust(left=0.08, right=0.99, top=0.80, bottom=0.28)
        fig.subplots_adjust(left=0.08, right=0.84, top=0.88, bottom=0.20)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", nargs="?", type=Path, help="Pedestrian Data workbook")
    parser.add_argument("--workbook", dest="workbook_option", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-show", action="store_true", help="Save without opening a graph window")
    args = parser.parse_args()
    adjacent = Path(__file__).resolve().parent / "Pedestrian Data.xlsx"
    source = (args.workbook_option or args.workbook or
              (adjacent if adjacent.exists() else DEFAULT_WORKBOOK)).resolve()
    if not source.is_file():
        parser.error(f"Workbook not found: {source}. Supply its path with --workbook.")
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary, data = summarise(source)
    figure = plot(summary, data, plt)
    output = (args.output_dir or source.parent / "Descriptive_graphs").resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "Violation_share_all_arrivals"
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
        figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    summary.to_csv(stem.with_suffix(".csv"), index=False)
    print(f"Included {len(data)} pedestrians; 24 bars in 12 crossing/day/period groups.")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"\nPDF, PNG and counts CSV saved in: {output}")
    if not args.no_show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
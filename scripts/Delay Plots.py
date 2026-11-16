from pathlib import Path
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

FILE = Path(__file__).resolve().parent / "Pedestrian Data.xlsx"
SHEET = "Cycle Data (My Estimator)"

KEYS = ["Day", "Crossing", "Time Period"]
ACTUAL = "Actual Total Delay (person-s)"
ESTIMATED = "Estimated Delay (s)"

df = pd.read_excel(FILE, sheet_name=SHEET)

# Clean labels and convert numeric columns.
for column in KEYS:
    df[column] = (
        df[column].astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

for column in ["Cycle", "Cycle Length", ACTUAL, ESTIMATED]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Keep complete cycle intervals.
df = df.dropna(subset=KEYS + ["Cycle"])
df = df.loc[df["Cycle Length"].gt(0)].copy()

groups = list(df.groupby(KEYS, sort=False))

if not groups:
    raise ValueError("No complete cycles found.")

output_folder = FILE.parent / "Delay_graphs"
output_folder.mkdir(exist_ok=True)

ncols = 3
nrows = math.ceil(len(groups) / ncols)

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(18, 4 * nrows),
    squeeze=False
)
axes = axes.ravel()

summary_rows = []
cycle_tables = []

for ax, (key, run) in zip(axes, groups):
    day, crossing, period = key
    run = run.sort_values("Cycle").copy()

    if run["Cycle"].duplicated().any():
        raise ValueError(f"Duplicate cycle numbers in {key}.")

    valid = (
        np.isfinite(run[ACTUAL])
        & np.isfinite(run[ESTIMATED])
    )

    omitted = run.loc[~valid, "Cycle"].astype(int).tolist()
    data = run.loc[valid].copy()

    if omitted:
        print(f"WARNING: {key}: omitted cycles {omitted} with missing/invalid delays.")

    ax.set_title(f"{day} {crossing}\n{period}", fontsize=11)

    if data.empty:
        ax.text(
            0.5, 0.5, "No complete comparison data",
            ha="center", va="center",
            transform=ax.transAxes
        )
        continue

    negative = (data[[ACTUAL, ESTIMATED]] < 0).any(axis=1)

    if negative.any():
        print(f"\nWARNING: negative delays in {key}:")
        print(
            data.loc[
                negative, ["Cycle", ACTUAL, ESTIMATED]
            ].to_string(index=False)
        )

        # Exclude these cycles from both the plot and error calculations.
        data = data.loc[~negative].copy()

    if data.empty:
        ax.text(
            0.5, 0.5, "No valid delay data",
            ha="center", va="center",
            transform=ax.transAxes
        )
        continue

    cycles = data["Cycle"].astype(int).to_numpy()
    actual = data[ACTUAL].to_numpy(dtype=float)
    estimated = data[ESTIMATED].to_numpy(dtype=float)

    ax.plot(
        cycles, actual,
        color="black", marker="o", markersize=3,
        linewidth=1.8, label="Actual"
    )
    ax.plot(
        cycles, estimated,
        color="tab:blue", marker="s", markersize=3,
        linewidth=1.4, label="Levinson estimate"
    )

    ax.set_xlabel("Cycle number")
    ax.set_ylabel("Total delay (person-seconds)")
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    ax.grid(alpha=0.25)

    error = estimated - actual

    summary_rows.append({
        "Day": day,
        "Crossing": crossing,
        "Time Period": period,
        "Cycles compared": len(data),
        "MAE (person-s/cycle)": np.mean(np.abs(error)),
        "RMSE (person-s/cycle)": np.sqrt(np.mean(error ** 2)),
        "Mean error (person-s/cycle)": np.mean(error),
        "Actual total (person-s)": actual.sum(),
        "Estimated total (person-s)": estimated.sum(),
    })

    cycle_tables.append(
        data[KEYS + ["Cycle", ACTUAL, ESTIMATED]]
    )

for ax in axes[len(groups):]:
    ax.set_visible(False)

if not summary_rows:
    plt.close(fig)
    raise ValueError("No runs have both actual and estimated delays.")

# One shared legend.
for ax in axes:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="upper center", ncol=2,
            bbox_to_anchor=(0.5, 0.975)
        )
        break

fig.suptitle(
    "Actual and estimated pedestrian delay by cycle",
    fontsize=17, y=0.995
)
fig.tight_layout(rect=[0, 0, 1, 0.945])

# Save the combined PNG and comparison tables.
figure_path = output_folder / "All_runs_delay_comparison.png"
summary_path = output_folder / "All_runs_delay_errors.csv"
detail_path = output_folder / "All_runs_delays_compared.csv"

fig.savefig(figure_path, dpi=300, bbox_inches="tight")

results = pd.DataFrame(summary_rows)
results.to_csv(summary_path, index=False)
pd.concat(cycle_tables, ignore_index=True).to_csv(detail_path, index=False)

print("\nDelay comparison:\n")
print(
    results.to_string(
        index=False,
        float_format=lambda value: f"{value:.2f}"
    )
)
print(f"\nCombined PNG: {figure_path}")
print(f"Error table: {summary_path}")
print(f"Compared delays: {detail_path}")

# Open one combined window with zoom/pan controls.
plt.show()
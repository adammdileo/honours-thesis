"""Compare Cox-predicted and observed delay reductions by pedestrian and cycle.

Default: refit the current Cox Model.py using the current source and regression
workbooks, then compare each timing-consistent person's expected reduction with
their observed reduction. Each scatter point is one pedestrian, including
compliant pedestrians with zero observed reduction. MAE, RMSE and Pearson r
are calculated over exactly the same paired people and saved separately.

    python "Delay Comparison.py" --no-show
    python "Delay Comparison.py" --analysis cycle --no-show

The default also sums the same paired pedestrians within each signal cycle.
Pedestrian-level and cycle-level graphs are saved as separate PDF and PNG files.
The optional cycle mode retains the previous comparison described below.

Compare observed delay with single, combined-train and separated-train models.

Run beside Pedestrian Data.xlsx, Train Arrival.xlsx, Train Window Calibration.py
and Train Direction u Histograms.py:
    python "Delay Comparison.py"
    python "Delay Comparison.py" --no-show

Windows default to [30,180) seconds after each train. The shared scripts supply
the same train readers, cycle timing and window partitioning as the flow plots.
Both workbooks are read-only. All factors are delay-specific and fitted within
each session; FLOW factors are not reused. All comparisons are IN-SAMPLE.

Observed target: actual waiting of pedestrians arriving during flashing red/red,
matched by arrival timestamp. Walk arrivals are excluded, as in the old script.
Only valid positive-latency actuated cycles are retained. Monday South morning,
partial cycles before zero and uncertain pre-video train history stay excluded.

Let A=first actuation, G=next Walk, R=G-A and s(t)=train category at time t.
Base delay for state s:
    B_s = R*1{s(A)=s} + lambda_j * integral_A^G (G-t)*1{s(t)=s} dt.
Thus sum_s B_s = R + lambda_j*R**2/2, the original pooled base delay.
First-actuation waiting is allocated by actuation time; later-arrival waiting
is integrated exactly over the category intervals. Observed category delay is
allocated by pedestrian arrival time. This is an exploratory extension.

alpha_D,s = summed observed category delay / summed B_s. The combined model
merges the three train categories; the separated model uses all four. A zero
base and zero target gives an undefined factor with zero contribution. A
positive target with zero modeled base raises an explicit error.
"""

import argparse
import bisect
import math
import runpy
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from openpyxl import load_workbook


MODELS = ("Single calibration factor", "Combined train factor", "Separated train factor")
STATES = ("out", "Cronulla only", "City only", "both")


def read_source_times(path):
    book = load_workbook(path, read_only=True, data_only=True)
    try:
        raw = list(book["Cycle Data"].values)
        columns = {value: i for i, value in enumerate(raw[0]) if value is not None}
        required = ("Flashing Red", "First Actuation", "Actuation Latency (s)", "Actuation to Next Walk (s)")
        if not set(required).issubset(columns):
            raise ValueError(f"Cycle Data needs timing columns {required}")
        cycle_times = {row_number: {name: row[columns[name]] for name in required}
                       for row_number, row in enumerate(raw[1:], 2)}
        base_column = next((name for name in ("Pooled Estimated Delay (s)", "Estimated Delay (s)") if name in columns), None)
        for row_number, row in enumerate(raw[1:], 2):
            cycle_times[row_number]["Saved pooled base"] = row[columns[base_column]] if base_column else None
        rows = list(book.worksheets[0].values)
        pc = {value: i for i, value in enumerate(rows[0]) if value is not None}
        if not {"Pedestrian ID", "Crossing Start"}.issubset(pc):
            raise ValueError("First worksheet needs Pedestrian ID and Crossing Start")
        crossings = {row[pc["Pedestrian ID"]]: row[pc["Crossing Start"]] for row in rows[1:]
                     if row[pc["Pedestrian ID"]] is not None}
    finally:
        book.close()
    return cycle_times, crossings


def delay_components(actuation, next_walk, rate, windows, shared):
    wait = next_walk - actuation
    bases = {state: 0.0 for state in STATES}
    bases[shared["category"](actuation, windows)] += wait
    _, segments = shared["split_cycle"](actuation, next_walk, windows)
    audit = []
    for left, right, state in segments:
        contribution = rate * (right - left) * (next_walk - (left + right) / 2)
        bases[state] += contribution
        audit.append((left, right, state, contribution))
    total = wait + rate * wait ** 2 / 2
    if not math.isclose(sum(bases.values()), total, rel_tol=1e-10, abs_tol=1e-7):
        raise ValueError("Train-category base delays do not sum to original pooled base")
    return bases, audit, wait


def analyse(args, shared, readers):
    number, key_of, labels = shared["number"], shared["key_of"], shared["labels"]
    normalise = readers["normalise_period"]
    cycles, rates, _ = shared["load_cycles_and_rates"](args.workbook, normalise)
    trains = readers["read_trains"](args.train_file)
    pedestrians = readers["calculate_pedestrian_u"](args.workbook, trains)
    timing, crossings = read_source_times(args.workbook)
    issues, segments, factors, sessions = [], [], [], []
    windows_by_key, retained = {}, {}
    for key, run in cycles.items():
        times = {d: trains.get((key[0], key[2], d), []) for d in readers["DIRECTIONS"]}
        windows = {d: shared["merge_windows"]([(t + args.train_start, t + args.train_end) for t in ts])
                   for d, ts in times.items()}
        windows_by_key[key] = windows
        rate, cell = rates.get(key, (None, ""))
        session_reason = ""
        if key in readers["EXCLUDED"]:
            session_reason = "Excluded: Monday South morning recorded at a different time"
        elif any(not ts for ts in times.values()):
            session_reason = "Missing train times for one or both directions"
        elif not number(rate) or rate <= 0:
            raise ValueError(f"Missing/invalid stored pooled rate for {key} at {cell}")
        history_start = max([0.0] + [0.0 if ts and ts[0] <= 0 else min(ts[0], args.train_end)
                                     if ts else args.train_end for ts in times.values()])
        for cycle in run:
            if session_reason:
                cycle["status"] = session_reason
            elif cycle["status"] != "Pending":
                continue
            elif cycle["start"] < 0:
                cycle["status"] = "Cycle begins before video zero"
            elif cycle["start"] < history_start - 1e-7:
                cycle["status"] = "Cycle overlaps uncertain pre-video train history"
            else:
                row = timing[cycle["row"]]
                flash, actuation, latency, wait = [row[name] for name in
                    ("Flashing Red", "First Actuation", "Actuation Latency (s)", "Actuation to Next Walk (s)")]
                if not all(number(value) for value in (flash, actuation, latency, wait)):
                    cycle["status"] = "Missing actuation/phase timing; pooled delay unavailable"
                elif not (cycle["start"] <= flash < actuation <= cycle["end"]) or latency <= 0 or wait < 0:
                    cycle["status"] = "Invalid phase/actuation timing or nonpositive latency"
                elif not (math.isclose(latency, actuation-flash, abs_tol=1e-7)
                          and math.isclose(wait, cycle["end"]-actuation, abs_tol=1e-7)):
                    cycle["status"] = "Stored latency/wait disagrees with timestamps"
                else:
                    cycle["status"] = "Included"
                    cycle.update({"flash": flash, "actuation": actuation, "wait": wait})
                    cycle["times"], _ = shared["split_cycle"](cycle["start"], cycle["end"], windows)
                    cycle["bases"], pieces, first_wait = delay_components(actuation, cycle["end"], rate, windows, shared)
                    cycle["actual"] = {s: 0.0 for s in STATES}
                    cycle["pedestrians"] = {s: 0 for s in STATES}
                    cycle["base"] = sum(cycle["bases"].values())
                    saved = row["Saved pooled base"]
                    if number(saved) and not math.isclose(saved, cycle["base"], rel_tol=1e-8, abs_tol=1e-7):
                        issues.append({**labels(key), "Cycle": cycle["cycle"], "Issue": "Saved pooled base differs from timestamp/rate calculation",
                                       "Saved base": saved, "Recomputed base": cycle["base"]})
                    segments.append({**labels(key), "Cycle": cycle["cycle"], "Component": "First actuation wait",
                                     "Category": shared["category"](actuation, windows),
                                     "Start (video s)": actuation, "End exclusive (video s)": cycle["end"],
                                     "Uncalibrated delay contribution (person-s)": first_wait})
                    segments.extend({**labels(key), "Cycle": cycle["cycle"], "Component": "Expected subsequent arrivals",
                                     "Category": state, "Start (video s)": left, "End exclusive (video s)": right,
                                     "Uncalibrated delay contribution (person-s)": amount} for left,right,state,amount in pieces)
        retained[key] = [c for c in run if c["status"] == "Included"]
        sessions.append({**labels(key), "Stored lambda_j (events/s)": rate, "Lambda source cell": cell,
                         "Earliest reliable train context (video s)": history_start,
                         "Cycles compared": len(retained[key]), "Status": session_reason or "Analysed"})
    timelines = {key: sorted([c for c in run if c["end"] is not None], key=lambda c: c["start"])
                 for key, run in cycles.items()}
    starts = {key: [c["start"] for c in run] for key, run in timelines.items()}
    for p in pedestrians:
        key = key_of(p["Day"], p["Crossing"], p["Time Period"], normalise)
        arrival = p["Arrival Time (video s)"]
        p.update({"Delay analysis status": "", "Matched cycle": "", "Train category": "", "Observed delay (person-s)": ""})
        if key in readers["EXCLUDED"]:
            p["Delay analysis status"] = "Excluded: Monday South morning recorded at a different time"
            continue
        if not number(arrival):
            p["Delay analysis status"] = "Missing/invalid arrival time"
            continue
        index = bisect.bisect_right(starts.get(key, []), arrival) - 1
        if index < 0 or arrival >= timelines[key][index]["end"]:
            p["Delay analysis status"] = "Outside complete cycle intervals; source cycle label retained"
            continue
        cycle = timelines[key][index]
        p["Matched cycle"] = cycle["cycle"]
        if cycle["status"] != "Included":
            p["Delay analysis status"] = cycle["status"]
            continue
        if arrival < cycle["flash"]:
            p["Delay analysis status"] = "Walk arrival; outside restrictive-phase delay target"
            continue
        crossing = crossings[p["Pedestrian ID"]]
        if not number(crossing) or crossing < arrival:
            raise ValueError(f"Missing/negative observed waiting time for pedestrian {p['Pedestrian ID']}")
        observed = crossing - arrival
        state = shared["category"](arrival, windows_by_key[key])
        p.update({"Delay analysis status": "Included", "Train category": state, "Observed delay (person-s)": observed})
        cycle["actual"][state] += observed
        cycle["pedestrians"][state] += 1
    for session in sessions:
        key = key_of(session["Day"], session["Crossing"], session["Time Period"], normalise)
        run = retained[key]
        if not run:
            continue
        bases = {s: sum(c["bases"][s] for c in run) for s in STATES}
        target = {s: sum(c["actual"][s] for c in run) for s in STATES}
        definitions = {
            MODELS[0]: {"all": STATES},
            MODELS[1]: {"out": ("out",), "in": STATES[1:]},
            MODELS[2]: {s: (s,) for s in STATES},
        }
        calibrated = {}
        for model, groups in definitions.items():
            coefficients = {}
            for group, states in groups.items():
                base = sum(bases[s] for s in states)
                actual = sum(target[s] for s in states)
                if base <= 0 and actual > 1e-8:
                    raise ValueError(f"{key}, {model}, {group}: observed delay > 0 but modeled base is zero; cannot estimate alpha")
                alpha = actual / base if base > 0 else None
                coefficients[group] = alpha
                factors.append({**shared["labels"](key), "Model": model, "Category": group,
                                "Observed total delay (person-s)": actual, "Pooled base total (person-s)": base,
                                "alpha_D": alpha, "Restrictive-phase pedestrians": sum(c["pedestrians"][s] for c in run for s in states),
                                "Cycles with base contribution": sum(sum(c["bases"][s] for s in states) > 0 for c in run),
                                "Status": "Undefined: no base exposure" if alpha is None else "Estimated"})
            calibrated[model] = (groups, coefficients)
        for cycle in run:
            cycle["predictions"] = {}
            for model, (groups, coefficients) in calibrated.items():
                prediction = sum((coefficients[group] or 0.0) * sum(cycle["bases"][s] for s in states)
                                 for group, states in groups.items())
                cycle["predictions"][model] = prediction
        session.update({"Observed delay (person-s)": sum(target.values()), "Pooled base delay (person-s)": sum(bases.values()),
                        "Restrictive-phase pedestrians": sum(sum(c["pedestrians"].values()) for c in run)})
        for model in MODELS:
            if not math.isclose(sum(c["predictions"][model] for c in run), sum(target.values()), abs_tol=1e-6):
                raise ValueError(f"Delay calibration totals failed to reconcile for {key}, {model}")
    all_cycles = [c for run in retained.values() for c in run]
    observed_from_people = sum(p["Observed delay (person-s)"] for p in pedestrians if p["Delay analysis status"] == "Included")
    if not math.isclose(observed_from_people, sum(sum(c["actual"].values()) for c in all_cycles), abs_tol=1e-6):
        raise ValueError("Pedestrian and cycle observed delay totals do not reconcile")
    return cycles, retained, pedestrians, sessions, factors, segments, issues


def errors(run, scope):
    rows = []
    for model in MODELS:
        if not run:
            continue
        e = np.array([c["predictions"][model] - sum(c["actual"].values()) for c in run])
        rows.append({**scope, "Model": model, "Cycles compared": len(run),
                     "MAE (person-s/cycle)": float(np.abs(e).mean()), "RMSE (person-s/cycle)": float(np.sqrt(np.mean(e**2))),
                     "Mean error (person-s/cycle)": float(e.mean()),
                     "Observed total (person-s)": sum(sum(c["actual"].values()) for c in run),
                     "Estimated total (person-s)": sum(c["predictions"][model] for c in run), "Evaluation": "In-sample"})
    return rows


def save(args, shared, result):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.ticker import MaxNLocator
    cycles, retained, pedestrians, sessions, factors, segments, issues = result
    output = args.output_dir or args.workbook.parent / "Delay_comparisons"
    output.mkdir(parents=True, exist_ok=True)
    folder = output / "Session_delay_comparisons"
    folder.mkdir(exist_ok=True)
    rows, session_errors = [], []
    for key, run in cycles.items():
        for c in run:
            row = {**shared["labels"](key), "Cycle": c["cycle"], "Cycle Data row": c["row"],
                   "Green (video s)": c["start"], "Next Walk (video s)": c["end"], "T_ij (s)": c["duration"], "Status": c["status"]}
            if c["status"] == "Included":
                row.update({"Flashing Red (video s)": c["flash"], "First Actuation (video s)": c["actuation"],
                            "Actuation-to-Walk wait (s)": c["wait"], "Pooled base delay (person-s)": c["base"],
                            "Restrictive-phase pedestrians": sum(c["pedestrians"].values()),
                            "Observed delay (person-s)": sum(c["actual"].values())})
                row.update({f"T_ij^{s} (s)": c["times"][s] for s in STATES})
                row.update({f"Base delay: {s} (person-s)": c["bases"][s] for s in STATES})
                row.update({f"Observed delay: {s} (person-s)": c["actual"][s] for s in STATES})
                row.update(c["predictions"])
            rows.append(row)
        session_errors.extend(errors(retained[key], shared["labels"](key)))
    all_cycles = [c for run in retained.values() for c in run]
    overall = errors(all_cycles, {"Day": "ALL", "Crossing": "ALL", "Time Period": "ALL"})
    files = {
        "Cycle_delay_comparison.csv": rows, "Delay_calibration_factors.csv": factors,
        "Session_delay_summary.csv": sessions, "Session_delay_errors.csv": session_errors,
        "Overall_delay_errors.csv": overall, "Delay_component_audit.csv": segments,
        "Pedestrian_delay_assignments.csv": pedestrians, "Validation_issues.csv": issues,
        "Excluded_cycles.csv": [r for r in rows if r["Status"] != "Included"],
        "Excluded_pedestrians.csv": [p for p in pedestrians if p["Delay analysis status"] != "Included"],
    }
    for name, content in files.items():
        shared["write_csv"](output / name, content)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "savefig.dpi": 300})
    groups = [(key, sorted(run, key=lambda c: c["cycle"])) for key, run in sorted(retained.items()) if run]
    overview, axes = plt.subplots(math.ceil(len(groups)/3), 3, figsize=(18, 4*math.ceil(len(groups)/3)), squeeze=False)
    def plot(ax, key, run):
        x = [c["cycle"] for c in run]
        ax.plot(x, [sum(c["actual"].values()) for c in run], color="black", marker="o", ms=3, lw=1.7, label="Observed delay")
        for model, colour, marker in zip(MODELS, ("tab:blue", "tab:green", "tab:orange"), ("s", "D", "^")):
            ax.plot(x, [c["predictions"][model] for c in run], color=colour, marker=marker, ms=3, lw=1.2, label=model)
        ax.set_title(f"{key[0].title()} {key[1].title()}\n{key[2].replace('-', ' - ')}", fontsize=12)
        ax.set_xlabel("Cycle number")
        ax.set_ylabel("Total delay (person-seconds)")
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
        ax.grid(alpha=0.25)
    with PdfPages(output / "All_sessions_delay_comparison.pdf") as document:
        for index, (key, run) in enumerate(groups):
            plot(axes.ravel()[index], key, run)
            fig, ax = plt.subplots(figsize=(11.5, 5.2))
            fig.subplots_adjust(left=0.09, right=0.66, bottom=0.14, top=0.85)
            plot(ax, key, run)
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
            filename = "_".join(key).replace("-", "_to_") + "_delay_comparison"
            for extension in ("png", "pdf"):
                fig.savefig(folder / f"{filename}.{extension}")
            document.savefig(fig)
            plt.close(fig)
    for ax in axes.ravel()[len(groups):]:
        ax.set_visible(False)
    handles, legend_labels = axes.ravel()[0].get_legend_handles_labels()
    overview.legend(handles, legend_labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.98), frameon=True)
    overview.suptitle("Observed and estimated pedestrian delay by cycle", y=0.998, fontsize=16)
    overview.tight_layout(rect=(0, 0, 1, 0.95))
    for extension in ("png", "pdf"):
        overview.savefig(output / f"All_runs_delay_comparison.{extension}")
    plt.close(overview)
    (output / "Train_delay_method.txt").write_text(
        __doc__ + f"\nApplied train window: [{args.train_start:g}, {args.train_end:g}) seconds.\n"
        "The first-actuation term is assigned by actuation time; actual delays are assigned by pedestrian arrival time.\n"
        "Each later-arrival contribution integrates lambda_j*(next Walk-time) exactly over its train-category segment.\n"
        "Category base components reconcile to R+lambda_j*R^2/2 for every retained cycle.\n"
        "The supplied session lambda_j values are preserved; their censoring treatment is not changed.\n"
        "No-exposure factors are blank. The three models use exactly the same cycles and actual delay target.\n"
        "Delay calibration absorbs observed crossing behaviour; it does not make these forecasts or causal effects.\n"
        "Same-session fitting and evaluation are in-sample. Delay factors differ from flow factors.\n"
        "Shared helpers are loaded from Train Window Calibration.py and Train Direction u Histograms.py.\n"
        "Figures are saved automatically; --no-show is retained for compatibility.\n", encoding="utf-8")
    print(f"Compared {len(all_cycles)} cycles across {len(groups)} sessions.")
    print(f"Included {sum(p['Delay analysis status']=='Included' for p in pedestrians)} restrictive-phase pedestrians.")
    print("Delay-specific factors are fitted separately within each session; all errors are in-sample.")
    for row in overall:
        print(f"{row['Model']}: MAE={row['MAE (person-s/cycle)']:.3f}, RMSE={row['RMSE (person-s/cycle)']:.3f} person-s/cycle")
    print("Pedestrian exclusions:")
    for status, count in Counter(p["Delay analysis status"] for p in pedestrians if p["Delay analysis status"] != "Included").items():
        print(f"  {count}: {status}")
    print(f"Validation issues: {len(issues)}. Saved to {output.resolve()}")


def reduction_metrics(frame):
    """All metrics use the same individual observed/predicted pairs."""
    from scipy.stats import pearsonr
    observed = frame["Observed early-crossing reduction (s)"].to_numpy(dtype=float)
    predicted = frame["Cox predicted early-crossing reduction (s)"].to_numpy(dtype=float)
    if not len(observed) or not np.isfinite(np.column_stack([observed, predicted])).all():
        raise ValueError("Delay metrics require nonempty, finite matched pedestrian pairs")
    error = predicted - observed
    correlation = (float(pearsonr(observed, predicted).statistic)
                   if len(observed) >= 2 and np.ptp(observed) > 0 and np.ptp(predicted) > 0 else np.nan)
    return {"Pedestrians": len(frame), "MAE (s/pedestrian)": float(np.abs(error).mean()),
            "RMSE (s/pedestrian)": float(np.sqrt(np.mean(error ** 2))),
            "Pearson r": correlation, "Bias (s/pedestrian)": float(error.mean()),
            "Observed total reduction (person-s)": float(observed.sum()),
            "Predicted total reduction (person-s)": float(predicted.sum())}


def save_cycle_reduction_comparison(paired, fit_folder, output):
    """Aggregate identical observed/predicted pedestrian pairs within each cycle."""
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr

    keys = ["Day", "Crossing", "Time Period", "Cycle"]
    observed_column = "Observed total reduction (person-s)"
    predicted_column = "Cox predicted total reduction (person-s)"
    cycles = paired.groupby(keys, sort=True, dropna=False).agg(**{
        "Matched pedestrians": ("Pedestrian ID", "size"),
        observed_column: ("Observed early-crossing reduction (s)", "sum"),
        predicted_column: ("Cox predicted early-crossing reduction (s)", "sum"),
    }).reset_index()
    reference = pd.read_csv(fit_folder / "Delay_reduction_comparison" / "Cycle_delay_reduction_comparison.csv")
    cycles = cycles.merge(reference[keys + ["Source restrictive arrivals", "Restrictive arrivals outside comparison"]],
                          on=keys, how="left", validate="one_to_one")
    if cycles["Source restrictive arrivals"].isna().any():
        raise ValueError("Missing source coverage for an aggregated cycle")
    if not np.array_equal((cycles["Source restrictive arrivals"] - cycles["Matched pedestrians"]).to_numpy(),
                          cycles["Restrictive arrivals outside comparison"].to_numpy()):
        raise ValueError("Cycle coverage counts do not reconcile")
    if cycles["Restrictive arrivals outside comparison"].lt(0).any():
        raise ValueError("Matched cycle count exceeds its source population")
    cycles["Coverage"] = np.where(cycles["Restrictive arrivals outside comparison"].eq(0),
                                   "Complete coverage", "Partial coverage")
    if (int(cycles["Matched pedestrians"].sum()) != len(paired)
            or not np.isclose(cycles[observed_column].sum(), paired["Observed early-crossing reduction (s)"].sum())
            or not np.isclose(cycles[predicted_column].sum(), paired["Cox predicted early-crossing reduction (s)"].sum())):
        raise ValueError("Pedestrian and cycle totals do not reconcile")
    cycles.to_csv(output / "Cycle_delay_reduction_comparison.csv", index=False, encoding="utf-8-sig")
    metric_rows = []
    for label, group in [("All matched cycles", cycles)] + list(cycles.groupby("Coverage", sort=True)):
        observed = group[observed_column].to_numpy(dtype=float)
        predicted = group[predicted_column].to_numpy(dtype=float)
        error = predicted - observed
        correlation = (float(pearsonr(observed, predicted).statistic)
                       if len(group) > 1 and np.ptp(observed) > 0 and np.ptp(predicted) > 0 else np.nan)
        metric_rows.append({"Coverage": label, "Cycles": len(group),
                            "Matched pedestrians": int(group["Matched pedestrians"].sum()),
                            "MAE (person-s/cycle)": float(np.abs(error).mean()),
                            "RMSE (person-s/cycle)": float(np.sqrt(np.mean(error**2))),
                            "Pearson r": correlation, "Bias (person-s/cycle)": float(error.mean())})
    pd.DataFrame(metric_rows).to_csv(output / "Cycle_delay_reduction_metrics.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(9.8, 7.0))
    ax.scatter(cycles[observed_column], cycles[predicted_column], s=32, alpha=0.7,
               color="tab:orange", edgecolors="none")
    maximum = max(float(cycles[[observed_column, predicted_column]].to_numpy().max()), 1.0)*1.05
    ax.plot([0, maximum], [0, maximum], color="0.45", linestyle="--", linewidth=1.2)
    ax.set(xlim=(0, maximum), ylim=(0, maximum),
           xlabel="Observed total delay reduction (person-seconds)",
           ylabel="Cox-predicted total delay reduction (person-seconds)",
           title="Cycle-level delay reduction (matched pedestrians)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_axisbelow(True)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"Cycle_delay_reduction_comparison.{extension}", bbox_inches="tight")
    plt.close(fig)
    print(f"Cycle plot: {len(cycles)} cycles; {int(cycles['Coverage'].eq('Complete coverage').sum())} with complete source coverage.")


def compare_individual_reductions(args):
    """Refit the current model and compare individual reductions and cycle sums."""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regression = (args.regression or args.workbook.with_name("Pedestrian Regression Data.xlsx")).resolve()
    cox_script = (args.cox_script or args.helper_dir / "Cox Model.py").resolve()
    inputs = [args.workbook, regression, cox_script, cox_script.with_name("hazard_correlations.py")]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    output = (args.output_dir or args.workbook.parent / "Delay_comparisons" /
              "Pedestrian_delay_reduction").resolve()
    output.mkdir(parents=True, exist_ok=True)
    fit_folder = output / "Cox_fit"
    command = [sys.executable, "-B", str(cox_script), "--source", str(args.workbook),
               "--regression", str(regression), "--output-dir", str(fit_folder), "--no-show"]
    print("Refitting the current Cox model for individual pedestrian delay reductions...", flush=True)
    with (output / "Cox_fit.log").open("w", encoding="utf-8") as log:
        fitted = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
    if fitted.returncode:
        raise RuntimeError(f"Cox fit failed; see {output / 'Cox_fit.log'}")
    for path in inputs:
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path)]:
            raise RuntimeError(f"Input changed during the analysis: {path}")
    comparison = fit_folder / "Delay_reduction_comparison"
    paired = pd.read_csv(comparison / "Pedestrian_delay_reduction_predictions.csv")
    required = ["Pedestrian ID", "Day", "Crossing", "Time Period", "Cycle", "Time to green (s)",
                "Observed waiting time (s)", "Observed early-crossing reduction (s)",
                "Observed extra wait after Walk (s)", "Cox expected wait up to Walk (s)",
                "Cox predicted early-crossing reduction (s)"]
    if not set(required).issubset(paired.columns) or paired.empty:
        raise ValueError("Fresh Cox output has no complete individual delay-reduction pairs")
    if paired["Pedestrian ID"].isna().any() or paired["Pedestrian ID"].duplicated().any():
        raise ValueError("Each comparison row must represent one unique pedestrian")
    phase_mapping = pd.read_excel(regression, sheet_name="Pedestrians", usecols=["Pedestrian ID", "Phase at Arrival"])
    phase_mapping = phase_mapping.loc[phase_mapping["Pedestrian ID"].notna()]
    paired = paired.merge(phase_mapping, on="Pedestrian ID", how="left", validate="one_to_one", sort=False)
    if not paired["Phase at Arrival"].isin([0, 1]).all():
        raise ValueError("Each compared pedestrian must have a known FR/R arrival phase")
    paired["Arrival phase"] = paired.pop("Phase at Arrival").map({0: "FR", 1: "R"})
    numbers = paired[required[5:]].to_numpy(dtype=float)
    if not np.isfinite(numbers).all():
        raise ValueError("Missing/non-finite values in individual delay reductions")
    horizon = paired["Time to green (s)"].to_numpy(dtype=float)
    observed_wait = paired["Observed waiting time (s)"].to_numpy(dtype=float)
    observed = paired["Observed early-crossing reduction (s)"].to_numpy(dtype=float)
    expected_wait = paired["Cox expected wait up to Walk (s)"].to_numpy(dtype=float)
    predicted = paired["Cox predicted early-crossing reduction (s)"].to_numpy(dtype=float)
    if (np.any(horizon <= 0) or np.any(observed_wait < 0) or np.any(predicted < -1e-7)
            or np.any(predicted > horizon + 1e-7)
            or not np.allclose(observed, np.maximum(horizon - observed_wait, 0), atol=1e-7, rtol=0)
            or not np.allclose(expected_wait + predicted, horizon, atol=1e-7, rtol=0)):
        raise ValueError("Individual reduction bounds or waiting-time identities failed")
    paired.to_csv(output / "Pedestrian_delay_reduction_comparison.csv", index=False, encoding="utf-8-sig")
    exclusions = comparison / "Comparison_exclusions.csv"
    shutil.copyfile(exclusions, output / "Pedestrian_comparison_exclusions.csv")
    metrics = reduction_metrics(paired)
    rows = [{"Scope": "Overall", "Arrival phase": "ALL", "Day": "ALL", "Crossing": "ALL", "Time Period": "ALL", **metrics}]
    for phase_code, group in paired.groupby("Arrival phase", sort=True):
        rows.append({"Scope": "Arrival phase", "Arrival phase": phase_code, "Day": "ALL", "Crossing": "ALL",
                     "Time Period": "ALL", **reduction_metrics(group)})
    for (day, crossing, period), group in paired.groupby(["Day", "Crossing", "Time Period"], sort=True):
        rows.append({"Scope": "Session", "Arrival phase": "ALL", "Day": day, "Crossing": crossing,
                     "Time Period": period, **reduction_metrics(group)})
    pd.DataFrame(rows).to_csv(output / "Pedestrian_delay_reduction_metrics.csv", index=False, encoding="utf-8-sig")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "savefig.dpi": 300})
    correlation_text = "undefined" if not np.isfinite(metrics["Pearson r"]) else f"{metrics['Pearson r']:.3f}"
    with plt.rc_context():
        fig, ax = plt.subplots(figsize=(9.8, 7.0))
        for phase_code, group in paired.groupby("Arrival phase", sort=True):
            ax.scatter(group["Observed early-crossing reduction (s)"],
                       group["Cox predicted early-crossing reduction (s)"],
                       s=17, alpha=0.45, edgecolors="none",
                       color={"FR": "tab:orange", "R": "tab:blue"}[phase_code],
                       label=f"{'Flashing red' if phase_code == 'FR' else 'Solid red'} ({phase_code})")
        maximum = max(float(max(observed.max(), predicted.max())), 1.0) * 1.05
        ax.set(xlim=(0, maximum), ylim=(0, maximum),
               xlabel="Observed delay reduction (seconds)",
               ylabel="Cox-predicted expected delay reduction (seconds)",
               title="Individual pedestrian delay reduction")
        ax.set_aspect("equal", adjustable="box")
        ax.set_axisbelow(True)
        ax.grid(alpha=0.2)
        ax.legend(title="Arrival phase", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(output / f"Pedestrian_delay_reduction_comparison.{extension}", bbox_inches="tight")
        plt.close(fig)
        save_cycle_reduction_comparison(paired, fit_folder, output)

        # Error and correlation have different units, so plot them on separate axes.
        fig, (error_ax, corr_ax) = plt.subplots(1, 2, figsize=(9.2, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
        values = [metrics["MAE (s/pedestrian)"], metrics["RMSE (s/pedestrian)"]]
        bars = error_ax.bar(["MAE", "RMSE"], values, color=["#3978A8", "#DE8C35"], width=0.55)
        error_ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=5)
        error_ax.set(ylabel="Error (seconds)", ylim=(0, max(max(values), 1.0)*1.2),
                     title="Prediction error")
        error_ax.set_axisbelow(True)
        error_ax.grid(axis="y", alpha=0.2)
        corr_ax.axhline(0, color="0.65", linewidth=1)
        if np.isfinite(metrics["Pearson r"]):
            corr_ax.scatter([0], [metrics["Pearson r"]], s=85, color="#3978A8", zorder=3)
            corr_ax.annotate(f"r = {correlation_text}", (0, metrics["Pearson r"]),
                            xytext=(0, 12), textcoords="offset points", ha="center")
        else:
            corr_ax.text(0.5, 0.5, "Pearson r undefined", transform=corr_ax.transAxes, ha="center")
        corr_ax.set(xlim=(-0.6, 0.6), ylim=(-1.1, 1.1), xticks=[], ylabel="Pearson correlation coefficient",
                    title="Observed versus predicted")
        corr_ax.grid(axis="y", alpha=0.2)
        fig.suptitle(f"Individual delay-reduction accuracy (n = {len(paired):,}; in-sample)", fontsize=12)
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(output / f"Pedestrian_delay_reduction_metrics.{extension}", bbox_inches="tight")
        plt.close(fig)

    (output / "Method_and_limitations.txt").write_text(
        "Individual pedestrian early-crossing delay-reduction comparison\n\n"
        "The current Cox Model.py is refitted on each run using the current input workbooks.\n"
        "The six predictors include solid-red duration and the direction-specific train indicators; no pooled pedestrian rate.\n"
        "For person i, g_i is time from arrival until next Walk and S_i(t) is the fitted survival curve.\n"
        "Expected reduction = g_i - integral from 0 to g_i of S_i(t) dt.\n"
        "The Cox script integrates the Breslow step curve exactly between hazard jumps, without extrapolation.\n"
        "Observed reduction = max(g_i - observed waiting time_i, 0).\n"
        "Each dot and each row represents one unique timing-consistent pedestrian, including zero observed reductions.\n"
        "Scatter colours distinguish arrival phases: flashing red (FR) and solid red (R), matched by unique pedestrian ID.\n"
        "Identical/overlapping points are retained; no visual jitter is used.\n"
        "The cycle plot sums the same matched pedestrians' observed and predicted reductions within recording/cycle.\n"
        "Cycle points are totals, not averages. Their unit is person-seconds; the individual plot uses seconds.\n"
        "Complete coverage means every source restrictive-phase arrival in that cycle has a valid matched prediction.\n"
        "Partial coverage means the point includes only the matched subset on BOTH axes; it is not a complete-cycle total.\n"
        "No empty or unsupported cycle is silently assigned a zero; coverage and cycle-level metrics are exported.\n"
        "MAE = mean(abs(predicted-observed)); RMSE = sqrt(mean((predicted-observed)^2)).\n"
        "Pearson r uses exactly the same paired individual reductions; undefined for constant inputs.\n"
        "Extra waiting after Walk is recorded separately; the target is early-crossing reduction, not net total delay.\n"
        "Missing predictors and timing/event/phase disagreements follow the current Cox script's exported exclusions.\n"
        "Fitting and evaluation are in-sample; the displayed errors are not independent predictive validation.\n"
        "The current Cox model's PH limitations remain; see Cox_fit/Delay_reduction_comparison/Comparison_notes.txt.\n"
        "Source workbooks are read-only. The previous pooled cycle comparison remains available via --analysis cycle.\n",
        encoding="utf-8")
    (output / "Input_manifest.json").write_text(json.dumps({"Input SHA256": hashes,
        "Comparison pedestrians": len(paired), "Metrics": metrics}, indent=2), encoding="utf-8")
    print(f"Compared {len(paired):,} individual pedestrians, including {int((observed == 0).sum()):,} zero observed reductions.")
    print(f"MAE = {metrics['MAE (s/pedestrian)']:.3f} s; RMSE = {metrics['RMSE (s/pedestrian)']:.3f} s; Pearson r = {correlation_text}")
    print(f"Results: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workbook", type=Path, default=Path(__file__).resolve().parent / "Pedestrian Data.xlsx")
    parser.add_argument("--analysis", choices=("pedestrian", "cycle"), default="pedestrian",
                        help="Default: pedestrian and cycle Cox delay-reduction plots; cycle: legacy calibrated total-delay analysis")
    parser.add_argument("--regression", type=Path, help="Regression workbook; defaults beside --workbook")
    parser.add_argument("--cox-script", type=Path, help="Current Cox Model.py; defaults inside --helper-dir")
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--helper-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--train-start", type=float, default=30.0)
    parser.add_argument("--train-end", type=float, default=180.0)
    parser.add_argument("--no-show", action="store_true", help="Compatibility option; figures are always saved without opening windows")
    args = parser.parse_args()
    args.workbook = args.workbook.resolve()
    if args.analysis == "pedestrian":
        compare_individual_reductions(args)
        return
    if not (math.isfinite(args.train_start) and math.isfinite(args.train_end) and 0 <= args.train_start < args.train_end):
        parser.error("Require 0 <= --train-start < --train-end")
    args.workbook = args.workbook.resolve()
    args.train_file = args.train_file or args.workbook.parent / "Train Arrival.xlsx"
    shared = runpy.run_path(str(args.helper_dir / "Train Window Calibration.py"))
    readers = runpy.run_path(str(args.helper_dir / "Train Direction u Histograms.py"))
    result = analyse(args, shared, readers)
    if not any(result[1].values()):
        raise ValueError("No valid cycles for delay comparison")
    save(args, shared, result)


if __name__ == "__main__":
    main()

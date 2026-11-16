"""Fit the five Cox predictors in thesis Table 3.3.

Inputs are read-only. Default cohort: flashing-red and solid-red arrivals.
Turning and through rates are separate predictors, divided by 5 lanes for South
and 2 lanes for East. Green arrivals are excluded; phase is a binary contrast.
Pre-train means 0 <= time to a listed train < 180 seconds, in either service
direction and, by default, for all walking directions (matching Table 3.3).

Run beside the three workbooks, or supply --source, --regression and --trains.
Use --arrival-phase red for the optional solid-red-only sensitivity analysis.
"""
from pathlib import Path
from collections import defaultdict, Counter
from datetime import time, timedelta
import argparse
import contextlib
import hashlib
import io
import json
import re
import warnings

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
from scipy.stats import chi2, pearsonr

LANES = {"south": 5, "east": 2}
KEYS = ["Day", "Crossing", "Time Period", "Cycle"]
PREDICTORS = ["RedTime", "TurningRate", "ThroughRate", "ArrivedOnRed", "TrainWithin3Min"]
LABELS = {
    "RedTime": "Full solid-red duration (per second)",
    "TurningRate": "Turning flow (vehicles/lane/second)",
    "ThroughRate": "Through flow (vehicles/lane/second)",
    "ArrivedOnRed": "Arrival phase (solid red vs flashing red)",
    "TrainWithin3Min": "Arrival within 180 seconds before a train",
}


def normal(value):
    return " ".join(str(value).strip().casefold().split())


def period(value):
    value = re.sub(r"\s+", "", normal(value)).replace(":00", "")
    value = value.replace("\u2013", "-").replace("\u2014", "-")
    return {"2:55pm-4pm": "3pm-4pm"}.get(value, value)


def seconds(value):
    if isinstance(value, timedelta):
        result = value.total_seconds()
    elif isinstance(value, time):
        result = value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1e6
    else:
        result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"Invalid train timestamp: {value!r}")
    return result


def direction(value):
    value = re.sub(r"[\s\-\u2013\u2014]+", " ", normal(value))
    return {"city bound": "city", "cronulla bound": "cronulla"}.get(value)


def read_train_clocks(path):
    clocks, special, audit = defaultdict(list), defaultdict(list), []
    book = load_workbook(path, read_only=True, data_only=True)
    try:
        for day, preferred in [("monday", "Monday Arrivals"), ("saturday", "Saturday Arrivals")]:
            name = preferred if preferred in book.sheetnames else "Monday Arrivals (except 8am S)"
            sheet = book[name]
            header = None
            for n, row in enumerate(sheet.iter_rows(max_row=20, values_only=True), 1):
                cols = {str(v).strip(): k for k, v in enumerate(row) if v is not None}
                if {"Requested window", "Direction", "Video Time"}.issubset(cols):
                    header = n
                    break
            if header is None:
                raise ValueError(f"Train headers not found in {name}")
            for n, row in enumerate(sheet.iter_rows(min_row=header + 1, values_only=True), header + 1):
                raw = row[cols["Direction"]]
                if raw is None:
                    continue
                d = direction(raw)
                if d is None:
                    raise ValueError(f"Unknown train direction in {name}, row {n}")
                p = period(row[cols["Requested window"]])
                timestamp = seconds(row[cols["Video Time"]])
                clocks[(day, p, d)].append(timestamp)
                audit.append({"Sheet": name, "Excel row": n, "Day": day, "Time Period": p,
                              "Service direction": d, "Video seconds": timestamp})
        name = "Monday Arrivals (8am S)"
        if name in book.sheetnames:
            for n, row in enumerate(book[name].values, 1):
                d = direction(row[2]) if len(row) > 6 and row[2] is not None else None
                if d:
                    timestamp = seconds(row[6])
                    special[d].append(timestamp)
                    audit.append({"Sheet": name, "Excel row": n, "Day": "monday",
                                  "Time Period": "8am-9am", "Service direction": d,
                                  "Video seconds": timestamp})
    finally:
        book.close()
    for values in list(clocks.values()) + list(special.values()):
        if len(values) != len(set(values)):
            raise ValueError("Duplicate train timestamp in a session/direction")
        values.sort()
    return clocks, special, pd.DataFrame(audit)


def train_indicator(row, clocks, special, walking="all"):
    if walking == "towards" and str(row["Direction"]).strip().upper() not in ["S", "E"]:
        return 0.0, np.nan, "Outside towards-station walking directions"
    day, crossing, p = normal(row["Day"]), normal(row["Crossing"]), period(row["Time Period"])
    a = row["Arrival Time"]
    if not np.isfinite(a):
        return np.nan, np.nan, "Missing arrival time"
    upcoming = []
    for d in ["city", "cronulla"]:
        values = (special.get(d, []) if (day, crossing, p) == ("monday", "south", "8am-9am")
                  else clocks.get((day, p, d), []))
        index = np.searchsorted(values, a, side="left")
        upcoming.append(float(values[index]) - a if index < len(values) else np.nan)
    known = [v for v in upcoming if np.isfinite(v)]
    nearest = min(known) if known else np.nan
    if any(0 <= v < 180 for v in known):
        return 1.0, nearest, "Train listed within [0,180) seconds"
    if len(known) == 2:
        return 0.0, nearest, "Next train known in both service directions; neither within window"
    return np.nan, nearest, "Insufficient future train coverage to determine zero"


def resolve_duplicate_source_ids(ped, reg):
    """Resolve source duplicates only by a unique recorded waiting-time match."""
    ped = ped.copy()
    audit = []
    remove = []
    duplicated = ped.loc[ped["Pedestrian ID"].duplicated(keep=False)]
    for person, group in duplicated.groupby("Pedestrian ID", sort=False):
        target = reg.loc[reg["Pedestrian ID"].eq(person), "Delay"]
        if len(target) != 1:
            raise ValueError(f"Cannot uniquely match duplicated source pedestrian ID {person}")
        target_delay = pd.to_numeric(target, errors="raise").iloc[0]
        source_delay = pd.to_numeric(group["Waiting Time"], errors="coerce")
        matches = np.isclose(source_delay, target_delay, rtol=0, atol=1e-7, equal_nan=False)
        if int(matches.sum()) != 1:
            raise ValueError(f"Source ID {person}: Waiting Time does not uniquely match regression Delay")
        chosen = group.index[matches][0]
        for index, row in group.iterrows():
            selected = index == chosen
            audit.append({"Pedestrian ID": int(person), "Source Excel row": int(index) + 2,
                          "Selected": int(selected), "Source waiting time": row["Waiting Time"],
                          "Regression delay": target_delay,
                          **{column: row[column] for column in ["Day", "Crossing", "Time Period"]},
                          "Rule": "Unique Waiting Time = regression Delay match; source workbook unchanged"})
            if not selected:
                remove.append(index)
    return ped.drop(index=remove), pd.DataFrame(audit, columns=[
        "Pedestrian ID", "Source Excel row", "Selected", "Source waiting time", "Regression delay",
        "Day", "Crossing", "Time Period", "Rule"])


def predicted_delay_reductions(cph, covariates, time_to_green):
    """Integrate each Breslow step survival curve up to that person's Walk.

    E[(g - T)+] = g - integral_0^g S(t | x) dt. The curve is held constant
    between baseline jumps, rather than interpolating across those jumps.
    Horizons beyond the observed follow-up are rejected, not extrapolated.
    """
    baseline = cph.baseline_cumulative_hazard_
    if baseline.shape[1] != 1:
        raise ValueError("Delay integration currently requires an unstratified Cox model.")
    times = baseline.index.to_numpy(dtype=float)
    hazards = baseline.iloc[:, 0].to_numpy(dtype=float)
    horizons = np.asarray(time_to_green, dtype=float)
    risks = np.asarray(cph.predict_partial_hazard(covariates), dtype=float).reshape(-1)
    if len(horizons) != len(covariates) or len(risks) != len(horizons):
        raise ValueError("Predictions and pedestrian horizons must have matching lengths.")
    if (not np.isfinite(times).all() or not np.isfinite(hazards).all()
            or np.any(np.diff(times) <= 0) or np.any(times < 0)
            or np.any(hazards < 0) or np.any(np.diff(hazards) < -1e-10)):
        raise ValueError("Invalid baseline cumulative hazard.")
    follow_up = float(np.max(cph.durations))
    if (not np.isfinite(horizons).all() or np.any(horizons < 0)
            or np.any(horizons > follow_up + 1e-7)
            or not np.isfinite(risks).all() or np.any(risks <= 0)):
        raise ValueError("Invalid horizon/risk, or horizon exceeds observed model follow-up.")
    waits, reductions, probabilities = [], [], []
    for horizon, risk in zip(horizons, risks):
        grid = np.unique(np.r_[0.0, times[(times >= 0) & (times < horizon)], horizon])
        indices = np.searchsorted(times, grid[:-1], side="right") - 1
        left_hazard = np.zeros(len(indices))
        present = indices >= 0
        left_hazard[present] = hazards[indices[present]]
        wait = float(np.dot(np.diff(grid), np.exp(-risk * left_hazard)))
        if not -1e-7 <= wait <= horizon + 1e-7:
            raise ValueError("Predicted wait is outside its allowed time-to-green horizon.")
        wait = float(np.clip(wait, 0, horizon))
        before_green = np.searchsorted(times, horizon, side="left") - 1
        hazard = hazards[before_green] if before_green >= 0 else 0.0
        waits.append(wait)
        reductions.append(horizon - wait)
        probabilities.append(float(-np.expm1(-risk * hazard)))
    return np.asarray(waits), np.asarray(reductions), np.asarray(probabilities)



def load_inputs(args, out):
    reg = pd.read_excel(args.regression, sheet_name="Pedestrians")
    ped = pd.read_excel(args.source, sheet_name="Pedestrian Data")
    cycles = pd.read_excel(args.source, sheet_name="Cycle Data")
    for frame in [reg, ped, cycles]:
        frame.columns = frame.columns.astype(str).str.strip()
    for frame in [reg, ped]:
        frame["Pedestrian ID"] = pd.to_numeric(frame["Pedestrian ID"], errors="coerce")
        frame.dropna(subset=["Pedestrian ID"], inplace=True)
        frame["Pedestrian ID"] = frame["Pedestrian ID"].astype(int)
    if reg["Pedestrian ID"].duplicated().any():
        raise ValueError("Duplicate regression pedestrian IDs")
    ped, duplicates = resolve_duplicate_source_ids(ped, reg)
    duplicates.to_csv(out / "Duplicate_source_ID_resolution.csv", index=False)
    for frame in [ped, cycles]:
        for col in KEYS[:3]:
            frame[col] = frame[col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
        frame["Cycle"] = pd.to_numeric(frame["Cycle"], errors="coerce")
    timing = cycles.dropna(subset=KEYS).copy()
    if timing.duplicated(KEYS).any():
        raise ValueError("Duplicate signal cycle keys")
    timing = timing[KEYS + ["Green", "Flashing Red", "Red", "Cycle Length", "Red Time"]].rename(
        columns={"Green": "Cycle green", "Flashing Red": "Cycle FR", "Red": "Cycle red",
                 "Cycle Length": "Cycle length", "Red Time": "Source red duration"})
    source_cols = ["Pedestrian ID"] + KEYS + ["Direction", "Arrival Time", "Arrival Phase",
                   "Crossing Start", "Waiting Time", "Notes"]
    data = reg.merge(ped[source_cols], on="Pedestrian ID", how="left", validate="one_to_one", indicator=True)
    if data["_merge"].ne("both").any():
        raise ValueError("Regression pedestrian ID absent from source workbook")
    data.drop(columns="_merge", inplace=True)
    data = data.merge(timing, on=KEYS, how="left", validate="many_to_one")
    data["Arrival Phase"] = data["Arrival Phase"].astype("string").str.strip().str.upper()
    numeric = ["Delay", "Hazard", "Phase at Arrival", "Red Time", "Arrival Time", "Crossing Start",
               "Waiting Time", "Cycle green", "Cycle FR", "Cycle red", "Cycle length", "Source red duration",
               "Conflicting Vehicles (Turning)", "Conflicting Vehicles (Through)"]
    for col in numeric:
        data[col] = pd.to_numeric(data[col].astype("string").str.strip().replace("", pd.NA),
                                  errors="raise").astype(float)
    data["Lane count"] = data["Crossing"].map(lambda v: LANES.get(normal(v), np.nan))
    if data["Lane count"].isna().any():
        raise ValueError("Unknown crossing: expected South or East")
    data["Cycle end"] = data["Cycle green"] + data["Cycle length"]
    data["Time to green"] = data["Cycle end"] - data["Arrival Time"]
    # Full solid-red duration is computed from the source phase boundaries.
    data["RedTime"] = data["Cycle end"] - data["Cycle red"]
    raw_through, raw_turning = "Conflicting Vehicles (Through)", "Conflicting Vehicles (Turning)"
    east = data["Crossing"].map(normal).eq("east")
    data["Raw through rate (vehicles/s)"] = data[raw_through]
    data["Raw turning rate (vehicles/s)"] = data[raw_turning]
    data.loc[east & data[raw_through].fillna(0).ne(0), ["Pedestrian ID", raw_through]].to_csv(
        out / "East_through_overrides.csv", index=False)
    data.loc[east, "Raw through rate (vehicles/s)"] = 0.0
    data["TurningRate"] = data["Raw turning rate (vehicles/s)"] / data["Lane count"]
    data["ThroughRate"] = data["Raw through rate (vehicles/s)"] / data["Lane count"]
    data["ArrivedOnRed"] = data["Arrival Phase"].map({"FR": 0, "R": 1})
    clocks, special, train_audit = read_train_clocks(args.trains)
    train_audit.to_csv(out / "Train_times_used.csv", index=False)
    before = data.apply(lambda row: train_indicator(row, clocks, special, args.pretrain_direction), axis=1)
    data[["TrainWithin3Min", "Seconds to next listed train", "Train coverage status"]] = pd.DataFrame(
        before.tolist(), index=data.index)
    data["TrainWithin3Min"] = pd.to_numeric(data["TrainWithin3Min"])
    reasons = [[] for _ in range(len(data))]

    def exclude(mask, reason):
        for index in np.flatnonzero(pd.array(mask, dtype="boolean").fillna(True).to_numpy(dtype=bool)):
            reasons[index].append(reason)

    allowed = ["R"] if args.arrival_phase == "red" else ["FR", "R"]
    exclude(~data["Arrival Phase"].isin(allowed), "Outside selected arrival cohort")
    expected_code = data["Arrival Phase"].map({"G": 0, "FR": 1, "R": 2})
    exclude(data["Phase at Arrival"].ne(expected_code), "Regression/source arrival phase disagreement")
    timing_ok = (data["Cycle green"].le(data["Cycle FR"]) & data["Cycle FR"].le(data["Cycle red"])
                 & data["Cycle red"].lt(data["Cycle end"]) & data["Cycle length"].gt(0))
    exclude(~timing_ok, "Missing or invalid signal timing")
    in_restrictive = data["Arrival Time"].ge(data["Cycle FR"]) & data["Arrival Time"].lt(data["Cycle end"])
    exclude(data["Arrival Phase"].isin(["FR", "R"]) & ~in_restrictive,
            "Arrival timestamp outside the assigned cycle restrictive phase")
    phase_time_ok = ((data["Arrival Phase"].eq("FR") & data["Arrival Time"].lt(data["Cycle red"]))
                     | (data["Arrival Phase"].eq("R") & data["Arrival Time"].ge(data["Cycle red"])))
    exclude(data["Arrival Phase"].isin(["FR", "R"]) & ~phase_time_ok, "Arrival phase/timestamp disagreement")
    red_ok = (np.isclose(data["RedTime"], data["Red Time"], atol=1e-6, rtol=0)
              & np.isclose(data["RedTime"], data["Source red duration"], atol=1e-6, rtol=0))
    exclude(pd.Series(~red_ok, index=data.index), "Regression/source full solid-red duration disagreement")
    wait_ok = (data["Delay"].ge(0) & np.isclose(data["Delay"], data["Waiting Time"], atol=1e-6, rtol=0)
               & np.isclose(data["Delay"], data["Crossing Start"] - data["Arrival Time"], atol=1e-6, rtol=0))
    exclude(~wait_ok, "Invalid or inconsistent observed waiting time")
    event_ok = data["Hazard"].isin([0, 1]) & data["Hazard"].eq(data["Delay"].lt(data["Time to green"]).astype(int))
    exclude(~event_ok, "Recorded event disagrees with crossing before next green")
    exclude(data["Notes"].fillna("").str.casefold().str.contains("review: cycle label retained"),
            "Unresolved source cycle assignment")
    if "Train within next 3 min" in data:
        data["Previous pre-train indicator"] = data["Train within next 3 min"]
    selected = PREDICTORS if args.arrival_phase == "combined" else [p for p in PREDICTORS if p != "ArrivedOnRed"]
    exclude(data[selected].isna().any(axis=1), "Missing model predictor or train coverage")
    exclude(~np.isfinite(data[selected]).all(axis=1), "Nonfinite model predictor")
    exclude(data[["TurningRate", "ThroughRate"]].lt(0).any(axis=1), "Negative traffic rate")
    exclude(data["Time to green"].le(0), "Nonpositive time to green")
    data["Exclusion reasons"] = ["; ".join(r) for r in reasons]
    data.to_csv(out / "Source_mapping_and_exclusions.csv", index=False)
    data.loc[data["Exclusion reasons"].ne("")].to_csv(out / "Cohort_exclusions.csv", index=False)
    fit = data.loc[data["Exclusion reasons"].eq("")].copy()
    fit["Event"] = fit["Hazard"].astype(int)
    fit["Duration"] = np.where(fit["Event"].eq(1), fit["Delay"], fit["Time to green"])
    fit["Cluster"] = fit[KEYS].astype(str).agg("|".join, axis=1)
    model = fit[["Duration", "Event"] + selected].copy()
    if not np.isfinite(model.to_numpy()).all() or len(fit) < 2 or not model["Event"].any():
        raise ValueError("Insufficient valid finite model data")
    if model[selected].nunique().lt(2).any():
        raise ValueError("At least one selected predictor is constant")
    if np.linalg.matrix_rank(np.column_stack([np.ones(len(model)), model[selected]])) != len(selected) + 1:
        raise ValueError("Predictors are linearly dependent")
    fit[["Pedestrian ID"] + KEYS + ["Lane count", "Raw turning rate (vehicles/s)",
         "Raw through rate (vehicles/s)", "TurningRate", "ThroughRate"]].to_csv(
        out / "Traffic_flow_calculation.csv", index=False)
    fit.to_csv(out / "Included_pedestrians.csv", index=False)
    model.to_csv(out / "Model_data.csv", index=False)
    return data, fit, model


def fit_model(model):
    cph = CoxPHFitter(penalizer=0.0)
    cph.fit(model, duration_col="Duration", event_col="Event",
            fit_options={"step_size": 0.1, "max_steps": 1000})
    return cph


def save_coefficient_table(cph, out, n, events, pretrain_direction="all"):
    table = cph.summary.copy()
    table.index.name = "Predictor"
    table.reset_index().to_csv(out / "Cox_results.csv", index=False)
    import matplotlib.pyplot as plt
    rows, latex = [], []
    for name, row in table.iterrows():
        label = LABELS[name]
        p = "<0.001" if row["p"] < .001 else f"{row['p']:.3f}"
        values = [label, f"{row['coef']:.3f}", f"{row['exp(coef)']:.3f}",
                  f"{row['exp(coef) lower 95%']:.3f}--{row['exp(coef) upper 95%']:.3f}", p]
        rows.append(values)
        latex.append(" & ".join(values) + r" \\")
    fig, ax = plt.subplots(figsize=(13, 2.3 + .30 * len(rows)))
    ax.axis("off")
    tab = ax.table(cellText=rows, colLabels=["Predictor", "Coefficient", "Hazard ratio", "95% HR interval", "p"],
                   colWidths=[.46, .12, .12, .20, .10], loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.scale(1, 1.7)
    for (r, c), cell in tab.get_celld().items():
        cell.set_linewidth(.4)
        if r == 0:
            cell.set_facecolor("#e7edf4")
            cell.set_text_props(weight="bold")
        if c == 0:
            cell.set_text_props(ha="left")
    ax.set_title(f"Cox proportional hazards model: {n:,} pedestrians, {events:,} early crossings", pad=12)
    fig.text(.5, .07, "Traffic rates are per lane. Model-based standard errors; in-sample estimates.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .08, 1, 1])
    fig.savefig(out / "Cox_results_table.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    phase_note = ("Phase reference: flashing red; Green arrivals excluded. " if "ArrivedOnRed" in table.index
                  else "Solid-red arrivals only; arrival phase is not a predictor. ")
    walking_note = ("All walking directions. " if pretrain_direction == "all"
                    else "Pre-train indicator requires walking S/E towards the station. ")
    note = (phase_note +
            "Turning and through rates are separate, in vehicles per lane per second "
            "(South: 5 lanes; East: 2 lanes). Pre-train: $0\\leq\\Delta t<180$ seconds. "
            + walking_note + "Model-based standard errors; fitting and evaluation are in-sample.")
    tex = [r"% Requires booktabs and graphicx", r"\begin{table}[htbp]", r"\centering\small",
           r"\caption{Cox proportional hazards model coefficient estimates.}",
           r"\label{tab:cox-five-predictors}", r"\resizebox{\textwidth}{!}{%",
           r"\begin{tabular}{lrrrr}\toprule",
           r"Predictor & Coefficient ($\beta$) & Hazard ratio & 95\% HR interval & $p$ \\",
           r"\midrule", *latex, r"\bottomrule\end{tabular}}", r"\par\smallskip",
           rf"\begin{{minipage}}{{\textwidth}}\footnotesize $n={n}$; {events} early crossings. " + note,
           r"\end{minipage}\end{table}", ""]
    (out / "Cox_results_table.tex").write_text("\n".join(tex), encoding="utf-8")
    return table


def phase_assessment(cph, fit, model, out):
    if "ArrivedOnRed" not in model:
        return {"arrival_phase": "Not estimable within a solid-red-only cohort"}
    import matplotlib.pyplot as plt
    reduced = fit_model(model.drop(columns="ArrivedOnRed"))
    statistic = max(0.0, 2 * (cph.log_likelihood_ - reduced.log_likelihood_))
    comparison = {"Same fitted pedestrians": len(model), "LR statistic adding arrival phase": statistic,
                  "LR df": 1, "LR p": float(chi2.sf(statistic, 1)),
                  "Full concordance (in-sample)": float(cph.concordance_index_),
                  "Without phase concordance (in-sample)": float(reduced.concordance_index_),
                  "Full partial AIC": float(cph.AIC_partial_), "Without phase partial AIC": float(reduced.AIC_partial_)}
    pd.DataFrame([comparison]).to_csv(out / "Arrival_phase_comparison.csv", index=False)
    observed = fit.groupby("Arrival Phase").agg(Pedestrians=("Event", "size"),
                                                Early_crossings=("Event", "sum"),
                                                Mean_wait_seconds=("Delay", "mean"))
    observed["Early_crossing_fraction"] = observed["Early_crossings"] / observed["Pedestrians"]
    observed.to_csv(out / "Arrival_phase_observed.csv")
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for phase, label in [("FR", "Flashing-red arrivals"), ("R", "Solid-red arrivals")]:
        group = fit[fit["Arrival Phase"].eq(phase)]
        km = KaplanMeierFitter(label=label).fit(group["Duration"], event_observed=group["Event"])
        km.plot_survival_function(ax=ax, ci_show=False)
    ax.set(xlabel="Seconds since pedestrian arrival", ylabel="Estimated probability of no early crossing",
           title="Observed early-crossing survival by arrival phase", ylim=(0, 1.02))
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "Arrival_phase_survival.png", dpi=200)
    plt.close(fig)
    return comparison


def residual_plots(cph, model, out):
    """Inspect scaled Schoenfeld residuals without LOWESS on tied zero times."""
    import matplotlib.pyplot as plt
    residuals = cph.compute_residuals(model, kind="scaled_schoenfeld")
    residuals.to_csv(out / "Scaled_Schoenfeld_residuals.csv")
    columns = list(residuals.columns)
    fig, axes = plt.subplots(len(columns), 1, figsize=(8, 2.7 * len(columns)), squeeze=False)
    times = model.loc[residuals.index, "Duration"].to_numpy()
    for ax, col in zip(axes.flat, columns):
        values = residuals[col].to_numpy()
        ax.scatter(times, values, s=9, alpha=.20, color="#345d80")
        summary = pd.DataFrame({"time": times, "residual": values})
        summary["bin"] = pd.cut(summary["time"], bins=8, include_lowest=True)
        means = summary.groupby("bin", observed=True)[["time", "residual"]].mean()
        ax.plot(means["time"], means["residual"], "o-", color="#bb5723", label="Means in time bins")
        ax.axhline(0, color="black", linestyle="--", linewidth=.8)
        ax.set(title=LABELS[col], xlabel="Waiting time at early-crossing event (seconds)",
               ylabel="Scaled residual")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "PH_residuals.png", dpi=150)
    plt.close(fig)


def delay_comparison(cph, fit, model, out):
    folder = out / "Delay_reduction_comparison"
    folder.mkdir(exist_ok=True)
    eligible = fit["Time to green"].le(float(np.max(cph.durations)) + 1e-7)
    excluded = fit.loc[~eligible, ["Pedestrian ID"] + KEYS + ["Time to green"]].copy()
    excluded["Reason"] = "Horizon beyond observed model follow-up; no survival extrapolation"
    excluded.to_csv(folder / "Comparison_exclusions.csv", index=False)
    model = model.loc[eligible].copy()
    fit = fit.loc[eligible].copy()
    covariates = model.drop(columns=["Duration", "Event"])
    wait, reduction, probability = predicted_delay_reductions(cph, covariates, fit["Time to green"])
    pairs = fit[["Pedestrian ID"] + KEYS + ["Arrival Phase", "Delay", "Time to green"]].copy()
    pairs["Observed reduction (s)"] = np.maximum(pairs["Time to green"] - pairs["Delay"], 0)
    pairs["Expected wait up to green (s)"] = wait
    pairs["Predicted reduction (s)"] = reduction
    pairs["Probability of early crossing"] = probability
    # Preserve the established column names used by the thesis delay scripts.
    pairs["Time to green (s)"] = pairs["Time to green"]
    pairs["Observed waiting time (s)"] = pairs["Delay"]
    pairs["Observed early-crossing reduction (s)"] = pairs["Observed reduction (s)"]
    pairs["Observed extra wait after Walk (s)"] = np.maximum(pairs["Delay"] - pairs["Time to green"], 0)
    pairs["Cox expected wait up to Walk (s)"] = wait
    pairs["Cox predicted early-crossing reduction (s)"] = reduction
    pairs["Cox probability of crossing strictly before Walk"] = probability
    cph.baseline_cumulative_hazard_.to_csv(folder / "Fitted_baseline_cumulative_hazard.csv")
    if not np.allclose(wait + reduction, pairs["Time to green"], atol=1e-7, rtol=0):
        raise ValueError("Predicted wait and reduction do not reconcile")
    pairs.to_csv(folder / "Pedestrian_delay_reduction_predictions.csv", index=False)
    cycles = pairs.groupby(KEYS, dropna=False).agg(
        Pedestrians=("Pedestrian ID", "size"), Observed=("Observed reduction (s)", "sum"),
        Predicted=("Predicted reduction (s)", "sum")).reset_index()
    cycles.to_csv(folder / "Cycle_delay_reduction_predictions.csv", index=False)
    rows = []
    for label, obs, pred, units in [
        ("Matched fitted pedestrians", pairs["Observed reduction (s)"], pairs["Predicted reduction (s)"], "seconds"),
        ("Matched cycle subsets", cycles["Observed"], cycles["Predicted"], "person-seconds")]:
        err = np.asarray(pred) - np.asarray(obs)
        rows.append({"Scope": label, "N": len(err), "MAE": np.abs(err).mean(),
                     "RMSE": np.sqrt(np.mean(err ** 2)), "Mean error": err.mean(),
                     "Pearson r": pearsonr(obs, pred).statistic, "Units": units})
    for phase in ["FR", "R"]:
        sub = pairs[pairs["Arrival Phase"].eq(phase)]
        if len(sub) > 1 and sub["Observed reduction (s)"].nunique() > 1:
            err = sub["Predicted reduction (s)"] - sub["Observed reduction (s)"]
            rows.append({"Scope": phase + " arrivals", "N": len(sub), "MAE": err.abs().mean(),
                         "RMSE": np.sqrt(np.mean(err ** 2)), "Mean error": err.mean(),
                         "Pearson r": pearsonr(sub["Observed reduction (s)"], sub["Predicted reduction (s)"]).statistic,
                         "Units": "seconds"})
    pd.DataFrame(rows).to_csv(folder / "Delay_reduction_metrics.csv", index=False)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parent
    parser.add_argument("--regression", type=Path, default=root / "Pedestrian Regression Data.xlsx")
    parser.add_argument("--source", type=Path, default=root / "Pedestrian Data.xlsx")
    parser.add_argument("--trains", type=Path, default=root / "Train Arrival.xlsx")
    parser.add_argument("--output-dir", type=Path, default=root / "Cox_Five_Predictor_results")
    parser.add_argument("--arrival-phase", choices=["combined", "red"], default="combined")
    parser.add_argument("--pretrain-direction", choices=["all", "towards"], default="all",
                        help="all matches Table 3.3; towards restricts the indicator to S/E walkers")
    parser.add_argument("--no-show", action="store_true", help="Accepted for compatibility; figures are always saved")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.regression, args.source, args.trains]}
    data, fit, model = load_inputs(args, out)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        cph = fit_model(model)
        table = save_coefficient_table(cph, out, len(model), int(model["Event"].sum()), args.pretrain_direction)
        ph = proportional_hazard_test(cph, model, time_transform="rank").summary
        ph.to_csv(out / "PH_diagnostic.csv")
        phase = phase_assessment(cph, fit, model, out)
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            cph.check_assumptions(model, p_value_threshold=.05, show_plots=False)
        (out / "PH_assumption_report.txt").write_text(log.getvalue(), encoding="utf-8")
        residual_plots(cph, model, out)
        plt.close("all")
        metrics = delay_comparison(cph, fit, model, out)
    warning_counts = Counter(str(w.message) for w in recorded)
    (out / "Warnings.txt").write_text("\n".join(f"{n} occurrence(s): {message}"
                                               for message, n in warning_counts.items()), encoding="utf-8")
    from hazard_correlations import save_correlations
    save_correlations(model.drop(columns=["Duration", "Event"]), out, True, labels=LABELS)
    ph_flags = ph.index[ph["p"].lt(.05)].tolist()
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in hashes.items()):
        raise ValueError("An input file changed during the run")
    fit_summary = {"Input SHA256": hashes, "Lane counts": LANES, "Arrival cohort": args.arrival_phase,
                   "Pre-train walking directions": args.pretrain_direction,
                   "Predictors": list(model.columns[2:]), "Fitted pedestrians": len(fit),
                   "Events": int(fit["Event"].sum()), "Censored": int(model["Event"].eq(0).sum()),
                   "Zero-duration observations": int(model["Duration"].eq(0).sum()),
                   "Shared signal cycles": int(fit["Cluster"].nunique()),
                   "Concordance (in-sample)": float(cph.concordance_index_),
                   "PH flags at .05": ph_flags, "Phase comparison": phase, "Delay metrics": metrics,
                   "Standard errors": "Model-based, not adjusted for dependence within shared signal cycles",
                   "Train definition": "0 <= time until either listed service <180 seconds; missing future coverage stays missing",
                   "Traffic definition": "Stored aggregate vehicle/s rate divided by total crossing lanes once",
                   "Scope": "Retrospective, fitted-sample associations; covariates include full-cycle information"}
    (out / "Fit_summary.json").write_text(json.dumps(fit_summary, indent=2), encoding="utf-8")
    print(table[["coef", "exp(coef)", "p"]].to_string())
    print(json.dumps({k: v for k, v in fit_summary.items() if k not in ["Input SHA256", "Delay metrics"]}, indent=2))


if __name__ == "__main__":
    main()

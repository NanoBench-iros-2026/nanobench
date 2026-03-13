# -*- coding: utf-8 -*-
"""
task1_sysid.py — System Identification Benchmark
=================================================

Scientific question: Does a voltage-conditioned thrust model outperform models
that ignore battery state?

Four models:
  1. Constant gain:        F = k_const * sum(u_i^2)
  2. Time-indexed:         F = k(t)    * sum(u_i^2),  k polynomial in time
  3. Linear regression:    a_z = w * [m1,m2,m3,m4, Vbat]  (black-box)
  4. Voltage-conditioned:  F = k(Vbat) * sum(u_i^2),  k quadratic in voltage [PROPOSED]

Ground truth: vertical acceleration derived by double-differentiating Vicon pz
              and adding gravity  →  a_z_gt = d²pz/dt² + 9.81

Outputs
-------
  results/task1_thrust_curve.pdf
  results/task1_prediction_error_vs_voltage.pdf
  results/task1_timeseries_comparison.pdf
  results/tables/task1_model_comparison_table.tex
  results/tables/task1_model_comparison_table.csv
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression

# Allow importing from benchmarks/ and src/
_BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(_BENCH))
sys.path.insert(0, str(_BENCH.parent / "src"))

from utils.data_loader import (
    TrajectoryData,
    discover_datasets,
    get_best_trajectory,
    MOTOR_COLS,
)
from utils.metrics import rmse, r2_score, bin_by_voltage, VOLTAGE_BINS
from utils.plotting import (
    setup_plotting,
    save_fig,
    SINGLE_COL,
    DOUBLE_COL,
    COLORS,
    MODEL_COLORS,
    MODEL_LINESTYLES,
)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
G = 9.81          # m/s²
MOTOR_MAX = 65535.0
SG_WINDOW = 15    # Savitzky-Golay window for double-differentiation of pz
SG_ORDER = 3      # polynomial order


# ---------------------------------------------------------------------------
# Ground truth acceleration
# ---------------------------------------------------------------------------

def compute_az_gt(df: pd.DataFrame) -> np.ndarray:
    """
    Derive vertical acceleration ground truth from Vicon pz.

    Uses Savitzky-Golay double differentiation (window=15, polyorder=3).
    Returns a_z_gt = d²pz/dt² + g  (effective thrust acceleration, m/s²).
    """
    pz = df["pz"].values.copy()
    t = df["t"].values
    dt = float(np.median(np.diff(t)))
    if dt <= 0 or not np.isfinite(dt):
        dt = 0.005  # fallback: 200 Hz

    # Clamp window to odd value ≤ len(pz)
    win = min(SG_WINDOW, len(pz) if len(pz) % 2 == 1 else len(pz) - 1)
    win = max(win, SG_ORDER + 2 if (SG_ORDER + 2) % 2 == 1 else SG_ORDER + 3)

    az = savgol_filter(pz, window_length=win, polyorder=SG_ORDER, deriv=2, delta=dt)
    return az + G  # add gravity: at hover az≈0, F_total = m*g


def compute_u_sq(df: pd.DataFrame) -> np.ndarray:
    """Sum of squared normalized motor commands: sum_i (m_i / 65535)^2."""
    motors = df[MOTOR_COLS].values / MOTOR_MAX
    return np.sum(motors**2, axis=1)


def compute_k_empirical(az_gt: np.ndarray, u_sq: np.ndarray, mass_kg: float) -> np.ndarray:
    """
    Empirical thrust gain k = F_total / sum(u_i^2).

    F_total = mass * az_gt.
    """
    F = mass_kg * az_gt
    k = F / (u_sq + 1e-9)
    return k


# ---------------------------------------------------------------------------
# Train/test split (temporal, 70/30 within flight rows)
# ---------------------------------------------------------------------------

def train_test_split(flight_df: pd.DataFrame, train_frac: float = 0.70):
    """Split flight_df into train (first 70%) and test (last 30%) by time."""
    n = len(flight_df)
    n_train = int(n * train_frac)
    return flight_df.iloc[:n_train], flight_df.iloc[n_train:]


# ---------------------------------------------------------------------------
# Model 1: Constant gain
# ---------------------------------------------------------------------------

def fit_model_const(az_gt_tr: np.ndarray, u_sq_tr: np.ndarray,
                    mass_kg: float) -> dict:
    k_vals = compute_k_empirical(az_gt_tr, u_sq_tr, mass_kg)
    finite = np.isfinite(k_vals)
    k_const = float(np.mean(k_vals[finite]))
    return {"k_const": k_const}


def predict_model_const(params: dict, u_sq: np.ndarray, vbat: np.ndarray,
                        t_rel: np.ndarray, mass_kg: float) -> np.ndarray:
    return (params["k_const"] * u_sq) / mass_kg - G


# ---------------------------------------------------------------------------
# Model 2: Time-indexed gain (polynomial in elapsed time)
# ---------------------------------------------------------------------------

def fit_model_time(az_gt_tr: np.ndarray, u_sq_tr: np.ndarray,
                   t_rel_tr: np.ndarray, mass_kg: float, deg: int = 3) -> dict:
    k_vals = compute_k_empirical(az_gt_tr, u_sq_tr, mass_kg)
    finite = np.isfinite(k_vals)
    coeffs = np.polyfit(t_rel_tr[finite], k_vals[finite], deg)
    return {"coeffs": coeffs}


def predict_model_time(params: dict, u_sq: np.ndarray, vbat: np.ndarray,
                       t_rel: np.ndarray, mass_kg: float) -> np.ndarray:
    k_t = np.polyval(params["coeffs"], t_rel)
    return (k_t * u_sq) / mass_kg - G


# ---------------------------------------------------------------------------
# Model 3: Black-box linear regression
# ---------------------------------------------------------------------------

def fit_model_linear(az_gt_tr: np.ndarray, X_tr: np.ndarray) -> dict:
    """X_tr columns: [m1, m2, m3, m4, Vbat] (raw motor values)."""
    finite = np.isfinite(az_gt_tr)
    reg = LinearRegression().fit(X_tr[finite], az_gt_tr[finite])
    return {"model": reg}


def predict_model_linear(params: dict, X: np.ndarray,
                         u_sq: np.ndarray = None, vbat: np.ndarray = None,
                         t_rel: np.ndarray = None,
                         mass_kg: float = None) -> np.ndarray:
    return params["model"].predict(X).ravel()


# ---------------------------------------------------------------------------
# Model 4: Voltage-conditioned gain (quadratic in Vbat) [PROPOSED]
# ---------------------------------------------------------------------------

def fit_model_voltage(az_gt_tr: np.ndarray, u_sq_tr: np.ndarray,
                      vbat_tr: np.ndarray, mass_kg: float,
                      deg: int = 2) -> dict:
    k_vals = compute_k_empirical(az_gt_tr, u_sq_tr, mass_kg)
    finite = np.isfinite(k_vals)
    coeffs = np.polyfit(vbat_tr[finite], k_vals[finite], deg)
    return {"coeffs": coeffs, "deg": deg}


def predict_model_voltage(params: dict, u_sq: np.ndarray, vbat: np.ndarray,
                          t_rel: np.ndarray, mass_kg: float) -> np.ndarray:
    k_v = np.polyval(params["coeffs"], vbat)
    return (k_v * u_sq) / mass_kg - G


# ---------------------------------------------------------------------------
# Per-trajectory evaluation
# ---------------------------------------------------------------------------

def evaluate_trajectory(traj: TrajectoryData) -> Optional[Dict]:
    """
    Fit and evaluate all four models on a single trajectory.

    Returns a dict with:
      - per-model RMSE / R² on test set
      - per-model RMSE binned by voltage
      - raw arrays for plotting
    """
    fdf = traj.flight_df
    if len(fdf) < 200:
        logger.warning(f"{traj.traj_id}: too few flight samples ({len(fdf)}), skipping")
        return None

    train_df, test_df = train_test_split(fdf, train_frac=0.70)

    # --- Ground truth ---
    az_gt_all = compute_az_gt(fdf)
    t_all = fdf["t"].values
    u_sq_all = compute_u_sq(fdf)
    vbat_all = fdf["pwr_pm_vbat"].values
    t_rel_all = t_all - t_all[0]

    n_tr = len(train_df)
    az_gt_tr  = az_gt_all[:n_tr]
    az_gt_te  = az_gt_all[n_tr:]
    u_sq_tr   = u_sq_all[:n_tr]
    u_sq_te   = u_sq_all[n_tr:]
    vbat_tr   = vbat_all[:n_tr]
    vbat_te   = vbat_all[n_tr:]
    t_rel_tr  = t_rel_all[:n_tr]
    t_rel_te  = t_rel_all[n_tr:]

    # Feature matrix for linear regression
    raw_motors_tr = train_df[MOTOR_COLS].values
    raw_motors_te = test_df[MOTOR_COLS].values
    X_tr = np.column_stack([raw_motors_tr, vbat_tr])
    X_te = np.column_stack([raw_motors_te, vbat_te])

    mass = traj.mass_kg

    # --- Fit ---
    p_const   = fit_model_const(az_gt_tr, u_sq_tr, mass)
    p_time    = fit_model_time(az_gt_tr, u_sq_tr, t_rel_tr, mass)
    p_linear  = fit_model_linear(az_gt_tr, X_tr)
    p_voltage = fit_model_voltage(az_gt_tr, u_sq_tr, vbat_tr, mass)

    # --- Predict on test set ---
    az_pred = {
        "Constant gain":       predict_model_const(p_const, u_sq_te, vbat_te, t_rel_te, mass),
        "Time-indexed":        predict_model_time(p_time, u_sq_te, vbat_te, t_rel_te, mass),
        "Linear regression":   predict_model_linear(p_linear, X_te),
        "Voltage-conditioned": predict_model_voltage(p_voltage, u_sq_te, vbat_te, t_rel_te, mass),
    }

    # --- Metrics ---
    metrics = {}
    for name, pred in az_pred.items():
        finite = np.isfinite(az_gt_te) & np.isfinite(pred)
        binned = bin_by_voltage(
            np.abs(pred[finite] - az_gt_te[finite]), vbat_te[finite]
        )
        metrics[name] = {
            "rmse_overall": rmse(pred, az_gt_te),
            "r2":           r2_score(pred, az_gt_te),
        }
        for _, _, label in VOLTAGE_BINS:
            arr = binned.get(label, np.array([]))
            metrics[name][f"rmse_{label}"] = float(np.sqrt(np.mean(arr**2))) if len(arr) > 0 else float("nan")

    return {
        "traj_id":     traj.traj_id,
        "traj_type":   traj.trajectory_type,
        "metrics":     metrics,
        # Raw arrays (on full flight data) for plotting
        "t":           t_all,
        "t_rel":       t_rel_all,
        "az_gt":       az_gt_all,
        "u_sq":        u_sq_all,
        "vbat":        vbat_all,
        "k_emp":       compute_k_empirical(az_gt_all, u_sq_all, mass),
        "az_pred_full": {
            name: (
                predict_model_const(p_const, u_sq_all, vbat_all, t_rel_all, mass)
                if name == "Constant gain" else
                predict_model_time(p_time, u_sq_all, vbat_all, t_rel_all, mass)
                if name == "Time-indexed" else
                predict_model_linear(p_linear,
                                     np.column_stack([fdf[MOTOR_COLS].values, vbat_all]))
                if name == "Linear regression" else
                predict_model_voltage(p_voltage, u_sq_all, vbat_all, t_rel_all, mass)
            )
            for name in az_pred
        },
        "n_train":    n_tr,
        "n_test":     len(az_gt_te),
        "volt_params": p_voltage,
    }


# ---------------------------------------------------------------------------
# Voltage segmentation (for error-vs-voltage plot)
# ---------------------------------------------------------------------------

def compute_voltage_segments(result: Dict, segment_s: float = 10.0) -> Dict:
    """
    Divide flight data into 10s segments.
    Return per-segment mean Vbat and per-model RMSE.
    """
    t = result["t"]
    dt = float(np.median(np.diff(t)))
    seg_samples = max(1, int(segment_s / dt))

    n = len(t)
    seg_vbat, seg_metrics = [], {name: [] for name in result["az_pred_full"]}

    for start in range(0, n - seg_samples + 1, seg_samples):
        end = start + seg_samples
        seg_vbat.append(float(np.mean(result["vbat"][start:end])))
        gt_seg = result["az_gt"][start:end]
        for name, pred_full in result["az_pred_full"].items():
            pred_seg = pred_full[start:end]
            seg_metrics[name].append(rmse(pred_seg, gt_seg))

    return {"vbat": np.array(seg_vbat),
            "rmse": {k: np.array(v) for k, v in seg_metrics.items()}}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_thrust_curve(results: List[Dict], volt_params: dict) -> plt.Figure:
    """Figure 1: k_empirical scatter + fitted quadratic k(Vbat)."""
    fig, ax = plt.subplots(figsize=SINGLE_COL)

    # Scatter empirical gains from all trajectories
    for res in results:
        k = res["k_emp"]
        v = res["vbat"]
        finite = np.isfinite(k) & (np.abs(k) < 1e4)
        # Subsample for readability
        step = max(1, int(len(k) / 500))
        ax.scatter(v[finite][::step], k[finite][::step],
                   s=4, alpha=0.25, color=COLORS["ekf"],
                   label=res["traj_type"] if res == results[0] else None)

    # Best-fit quadratic from most recent volt_params
    if volt_params is not None:
        v_range = np.linspace(3.3, 4.3, 200)
        k_fit = np.polyval(volt_params["coeffs"], v_range)
        ax.plot(v_range, k_fit, color=COLORS["our_model"], linewidth=2.0,
                label=r"$k(V) = aV^2+bV+c$ (fitted)")

    # Voltage bin boundaries
    for v_thresh, ls in [(4.0, "--"), (3.8, ":")]:
        ax.axvline(v_thresh, color="grey", linestyle=ls, linewidth=0.8, alpha=0.7)

    ax.set_xlabel("Battery voltage (V)")
    ax.set_ylabel(r"Thrust gain $k$ (N)")
    ax.set_title("Voltage-conditioned thrust gain")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_error_vs_voltage(all_segments: List[Dict]) -> plt.Figure:
    """Figure 2: RMSE per 10s segment vs mean battery voltage, 4 model lines."""
    fig, ax = plt.subplots(figsize=SINGLE_COL)

    # Aggregate segments across all trajectories
    agg_vbat: List[float] = []
    agg_rmse: Dict[str, List[float]] = {}

    for seg in all_segments:
        agg_vbat.extend(seg["vbat"].tolist())
        for name, vals in seg["rmse"].items():
            agg_rmse.setdefault(name, []).extend(vals.tolist())

    if not agg_vbat:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return fig

    agg_vbat = np.array(agg_vbat)
    order = np.argsort(agg_vbat)

    for name, vals in agg_rmse.items():
        vals_arr = np.array(vals)[order]
        v_sorted = agg_vbat[order]
        # Smooth with a rolling mean (window=3)
        if len(v_sorted) >= 3:
            from numpy.lib.stride_tricks import sliding_window_view
            v_sm = v_sorted[1:-1]
            r_sm = np.array([np.mean(vals_arr[max(0,i-1):i+2])
                             for i in range(1, len(vals_arr)-1)])
        else:
            v_sm, r_sm = v_sorted, vals_arr

        ax.plot(v_sm, r_sm,
                color=MODEL_COLORS.get(name, COLORS["default"]),
                linestyle=MODEL_LINESTYLES.get(name, "-"),
                label=name)

    ax.set_xlabel("Battery voltage (V)")
    ax.set_ylabel(r"Acceleration RMSE (m/s²)")
    ax.set_title("Prediction error vs battery voltage")
    ax.legend(fontsize=8, loc="upper right")
    ax.invert_xaxis()  # depleted on right side
    fig.tight_layout()
    return fig


def plot_timeseries_comparison(result: Dict, traj_label: str) -> plt.Figure:
    """Figure 3: Ground truth vs model predictions in last 30% of flight."""
    t = result["t"]
    t_rel = result["t_rel"]
    az_gt = result["az_gt"]
    az_preds = result["az_pred_full"]
    n = len(t)
    start = int(0.70 * n)

    fig, ax = plt.subplots(figsize=DOUBLE_COL)
    t_plot = t_rel[start:] - t_rel[start]  # re-zero

    ax.plot(t_plot, az_gt[start:], color=COLORS["vicon"],
            linewidth=1.8, label="Ground truth (Vicon)", zorder=5)

    for name, pred in az_preds.items():
        ax.plot(t_plot, pred[start:],
                color=MODEL_COLORS.get(name, COLORS["default"]),
                linestyle=MODEL_LINESTYLES.get(name, "-"),
                linewidth=1.2, label=name, alpha=0.85)

    ax.set_xlabel("Time (s) — depleted battery region")
    ax.set_ylabel(r"Vertical acceleration (m/s²)")
    ax.set_title(f"Model comparison on {traj_label} (last 30% of flight, low battery)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def build_table(all_results: List[Dict]) -> pd.DataFrame:
    """
    Build model comparison table aggregated across all trajectories.

    Rows = models, Columns = RMSE per voltage bin + overall RMSE + R².
    """
    model_names = ["Constant gain", "Time-indexed", "Linear regression", "Voltage-conditioned"]
    bin_labels = [label for _, _, label in VOLTAGE_BINS]

    # Aggregate per model across trajectories
    agg: Dict[str, Dict[str, List[float]]] = {m: {} for m in model_names}
    for res in all_results:
        for m in model_names:
            if m not in res["metrics"]:
                continue
            mmet = res["metrics"][m]
            for key, val in mmet.items():
                agg[m].setdefault(key, []).append(val)

    rows = []
    for m in model_names:
        row = {"Model": m}
        for label in bin_labels:
            key = f"rmse_{label}"
            vals = [v for v in agg[m].get(key, []) if np.isfinite(v)]
            row[f"RMSE {label}"] = f"{np.mean(vals):.3f}" if vals else "—"
        overall_vals = [v for v in agg[m].get("rmse_overall", []) if np.isfinite(v)]
        r2_vals      = [v for v in agg[m].get("r2", []) if np.isfinite(v)]
        row["RMSE overall (m/s²)"] = f"{np.mean(overall_vals):.3f}" if overall_vals else "—"
        row["R²"]                  = f"{np.mean(r2_vals):.3f}"      if r2_vals      else "—"
        rows.append(row)

    return pd.DataFrame(rows).set_index("Model")


def save_table(df: pd.DataFrame, name: str, results_dir: Path) -> None:
    """Save table as both .csv and .tex."""
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(tables_dir / f"{name}.csv")

    # LaTeX with booktabs
    latex = df.to_latex(
        escape=False,
        caption=(
            "System identification model comparison. "
            "RMSE of predicted vertical acceleration (m/s²). "
            f"n={{}}"
        ),
        label="tab:sysid",
    )
    # Bold minimum RMSE per column (simple string replacement)
    # We'll bold the row with minimum overall RMSE
    with open(tables_dir / f"{name}.tex", "w") as f:
        f.write(latex)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(trajectories: Optional[List[TrajectoryData]] = None) -> Dict:
    """Run Task 1 end-to-end. Returns summary dict."""
    setup_plotting()
    results_dir = Path(__file__).resolve().parent / "results"

    if trajectories is None:
        trajectories = discover_datasets()

    if not trajectories:
        logger.error("Task 1: No trajectories found, aborting.")
        return {}

    logger.info(f"Task 1: evaluating {len(trajectories)} trajectory/ies")

    all_results = []
    all_segments = []
    last_volt_params = None

    for traj in trajectories:
        logger.info(f"  Processing {traj.traj_id} ...")
        res = evaluate_trajectory(traj)
        if res is None:
            continue
        all_results.append(res)
        segs = compute_voltage_segments(res)
        all_segments.append(segs)
        last_volt_params = res["volt_params"]

    if not all_results:
        logger.error("Task 1: No valid results produced.")
        return {}

    # --- Figure 1: Thrust curve ---
    fig1 = plot_thrust_curve(all_results, last_volt_params)
    p1 = save_fig(fig1, "task1_thrust_curve.pdf")
    logger.info(f"  Saved {p1}")

    # --- Figure 2: Error vs voltage ---
    fig2 = plot_error_vs_voltage(all_segments)
    p2 = save_fig(fig2, "task1_prediction_error_vs_voltage.pdf")
    logger.info(f"  Saved {p2}")

    # --- Figure 3: Time series comparison (best trajectory = widest voltage range) ---
    best = get_best_trajectory(trajectories, prefer_type="circle")
    best_result = next((r for r in all_results if r["traj_id"] == best.traj_id), all_results[0])
    fig3 = plot_timeseries_comparison(best_result, best.traj_id)
    p3 = save_fig(fig3, "task1_timeseries_comparison.pdf")
    logger.info(f"  Saved {p3}")

    # --- Table ---
    table_df = build_table(all_results)
    save_table(table_df, "task1_model_comparison_table", results_dir)
    logger.info("  Saved task1_model_comparison_table.tex/.csv")

    # --- Summary numbers ---
    summary = {}
    for res in all_results:
        m = res["metrics"]
        # RMSE reduction at depleted battery: const vs voltage-conditioned
        bin_key = f"rmse_{VOLTAGE_BINS[2][2]}"  # "<3.8V (depleted)"
        r_const = m.get("Constant gain", {}).get(bin_key, float("nan"))
        r_volt  = m.get("Voltage-conditioned", {}).get(bin_key, float("nan"))
        if np.isfinite(r_const) and np.isfinite(r_volt) and r_const > 0:
            pct = (r_const - r_volt) / r_const * 100
            summary[res["traj_id"]] = {"rmse_reduction_depleted_pct": pct}

    return {
        "all_results": all_results,
        "table": table_df,
        "summary": summary,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run()

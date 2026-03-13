# -*- coding: utf-8 -*-
"""
task2_control.py — Controller Benchmarking
==========================================

Scientific question: How does the standard Crazyflie PID controller tracking
performance degrade with battery voltage, and how does a simulated voltage
feedforward compensate?

Two conditions evaluated:
  1. Standard PID (no compensation)  — direct from aligned.csv
  2. Simulated voltage feedforward    — post-hoc approximation (clearly labeled)

Metrics per trajectory:
  - Position RMSE (mm)
  - 95th percentile error (mm)
  - Steady-state error (mm, last 20% of flight)
  - Yaw RMSE (deg)

Voltage segmentation: 10s windows, linear regression of tracking error vs Vbat.

Outputs
-------
  results/task2_trajectory_difficulty.pdf
  results/task2_error_vs_voltage.pdf
  results/task2_3d_trajectory.pdf
  results/task2_error_timeseries.pdf
  results/tables/task2_performance_table.tex
  results/tables/task2_performance_table.csv
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 (registers 3D projection)
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.pyplot as plt
import matplotlib.cm as cm

_BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(_BENCH))
sys.path.insert(0, str(_BENCH.parent / "src"))

from utils.data_loader import TrajectoryData, discover_datasets, get_best_trajectory
from utils.metrics import rmse, VOLTAGE_BINS
from utils.plotting import (
    setup_plotting,
    save_fig,
    SINGLE_COL,
    DOUBLE_COL,
    DOUBLE_COL_TALL,
    COLORS,
    traj_color,
    dual_axis_timeseries,
    add_voltage_vlines,
)

logger = logging.getLogger(__name__)

V_NOM = 3.87        # nominal LiPo operating voltage for feedforward
V_NOM_CORRECTION = 0.30   # fraction of z-error corrected by feedforward (conservative)


# ---------------------------------------------------------------------------
# Tracking error computation
# ---------------------------------------------------------------------------

def compute_tracking_error(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute position and yaw tracking errors on the flight mask.

    Returns
    -------
    e_pos : (N,) Euclidean position error [meters]
    e_yaw : (N,) yaw error [degrees], wrapped to [-180, 180]
    """
    # Position error
    e_pos = np.sqrt(
        (df["px"] - df["sp_ctrltarget_x"]) ** 2
        + (df["py"] - df["sp_ctrltarget_y"]) ** 2
        + (df["pz"] - df["sp_ctrltarget_z"]) ** 2
    ).values

    # Yaw error: setpoint is in radians, EKF attitude is in degrees
    att_yaw_deg = df["att_stateEstimate_yaw"].values
    sp_yaw_rad  = df["sp_ctrltarget_yaw"].values
    sp_yaw_deg  = np.degrees(sp_yaw_rad)
    e_yaw_raw   = att_yaw_deg - sp_yaw_deg
    e_yaw       = (e_yaw_raw + 180.0) % 360.0 - 180.0  # wrap to [-180, 180]

    return e_pos, e_yaw


def compute_simulated_ff_error(e_pos: np.ndarray,
                               df: pd.DataFrame) -> np.ndarray:
    """
    Simulate a voltage feedforward compensation.

    Scales the z-component of tracking error by V_nom/Vbat.  Applies a
    conservative correction factor (V_NOM_CORRECTION) to represent a realistic
    first-order feedforward that partially compensates the voltage drop.

    NOTE: This is a simulation / post-hoc approximation — NOT from re-flown data.
    """
    vbat = df["pwr_pm_vbat"].values
    scale = np.clip(V_NOM / vbat, 1.0, 1.5)  # never reduce thrust beyond nominal
    # Correction reduces tracking error proportionally to how much voltage dropped
    correction = V_NOM_CORRECTION * (scale - 1.0)
    e_ff = e_pos * np.maximum(0.0, 1.0 - correction)
    return e_ff


# ---------------------------------------------------------------------------
# Per-trajectory metrics
# ---------------------------------------------------------------------------

def evaluate_trajectory(traj: TrajectoryData) -> Optional[Dict]:
    """Compute tracking metrics for one trajectory."""
    fdf = traj.flight_df
    if len(fdf) < 200:
        logger.warning(f"{traj.traj_id}: only {len(fdf)} flight rows, skipping")
        return None

    # Filter out rows where setpoint is (0, 0, 0) — pre-trajectory hover or
    # segments where setpoint was not yet commanded
    sp_active = ~(
        (fdf["sp_ctrltarget_x"].abs() < 1e-6)
        & (fdf["sp_ctrltarget_y"].abs() < 1e-6)
        & (fdf["sp_ctrltarget_z"].abs() < 1e-6)
    )
    active_fdf = fdf[sp_active]
    if len(active_fdf) < 100:
        # Fall back to all flight rows if setpoint was never set
        logger.debug(f"{traj.traj_id}: setpoint always zero, using all flight rows")
        active_fdf = fdf

    e_pos, e_yaw = compute_tracking_error(active_fdf)
    e_ff         = compute_simulated_ff_error(e_pos, active_fdf)
    vbat         = active_fdf["pwr_pm_vbat"].values

    n = len(e_pos)
    ss_start = int(0.80 * n)  # steady-state: last 20%

    return {
        "traj_id":          traj.traj_id,
        "trajectory_type":  traj.trajectory_type,
        "mean_vbat":        float(np.mean(vbat)),
        "min_vbat":         float(np.min(vbat)),
        "pos_rmse_mm":      rmse(e_pos, np.zeros_like(e_pos)) * 1000,
        "pos_p95_mm":       float(np.percentile(e_pos, 95)) * 1000,
        "pos_ss_err_mm":    float(np.mean(e_pos[ss_start:])) * 1000,
        "yaw_rmse_deg":     float(np.sqrt(np.mean(e_yaw**2))),
        "ff_pos_rmse_mm":   rmse(e_ff, np.zeros_like(e_ff)) * 1000,
        # Raw arrays for plotting
        "_e_pos":           e_pos,
        "_e_ff":            e_ff,
        "_vbat":            vbat,
        "_t":               active_fdf["t"].values,
        "_px":              active_fdf["px"].values,
        "_py":              active_fdf["py"].values,
        "_pz":              active_fdf["pz"].values,
        "_sp_x":            active_fdf["sp_ctrltarget_x"].values,
        "_sp_y":            active_fdf["sp_ctrltarget_y"].values,
        "_sp_z":            active_fdf["sp_ctrltarget_z"].values,
    }


# ---------------------------------------------------------------------------
# Voltage segmentation
# ---------------------------------------------------------------------------

def compute_voltage_segments(result: Dict, segment_s: float = 10.0) -> Dict:
    """Split tracking error into 10s windows, return per-window mean Vbat and RMSE."""
    t = result["_t"]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.005
    seg_n = max(1, int(segment_s / dt))

    n = len(t)
    segs_v, segs_e, segs_eff = [], [], []
    for start in range(0, n - seg_n + 1, seg_n):
        end = start + seg_n
        segs_v.append(float(np.mean(result["_vbat"][start:end])))
        segs_e.append(float(np.mean(result["_e_pos"][start:end])) * 1000)    # mm
        segs_eff.append(float(np.mean(result["_e_ff"][start:end])) * 1000)   # mm

    return {
        "vbat":        np.array(segs_v),
        "e_pid_mm":    np.array(segs_e),
        "e_ff_mm":     np.array(segs_eff),
        "traj_type":   result["trajectory_type"],
        "traj_id":     result["traj_id"],
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_trajectory_difficulty(all_results: List[Dict]) -> plt.Figure:
    """Figure 1: Grouped bar chart of tracking error per trajectory type."""
    # Group by trajectory_type
    types = sorted({r["trajectory_type"] for r in all_results})
    pos_rmse = [np.mean([r["pos_rmse_mm"] for r in all_results
                         if r["trajectory_type"] == t]) for t in types]
    yaw_rmse = [np.mean([r["yaw_rmse_deg"] for r in all_results
                         if r["trajectory_type"] == t]) for t in types]

    # Sort by pos_rmse ascending (easy → hard)
    order = np.argsort(pos_rmse)
    types    = [types[i] for i in order]
    pos_rmse = [pos_rmse[i] for i in order]
    yaw_rmse = [yaw_rmse[i] for i in order]

    x = np.arange(len(types))
    width = 0.35
    fig, ax1 = plt.subplots(figsize=SINGLE_COL if len(types) <= 3 else DOUBLE_COL)

    bars1 = ax1.bar(x - width/2, pos_rmse, width,
                    color=COLORS["ekf"], label="Pos RMSE (mm)")
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, yaw_rmse, width,
                    color=COLORS["our_model"], alpha=0.7, label="Yaw RMSE (°)")

    ax1.set_xticks(x)
    ax1.set_xticklabels([t.replace("_", " ") for t in types], rotation=20, ha="right")
    ax1.set_ylabel("Position RMSE (mm)", color=COLORS["ekf"])
    ax2.set_ylabel("Yaw RMSE (°)", color=COLORS["our_model"])
    ax1.set_title("Tracking difficulty by trajectory type")
    ax2.spines["right"].set_visible(True)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def plot_error_vs_voltage(all_segs: List[Dict]) -> plt.Figure:
    """Figure 2: Scatter of tracking error vs battery voltage, with linear fit."""
    fig, ax = plt.subplots(figsize=SINGLE_COL)

    all_v, all_e = [], []

    for seg in all_segs:
        v = seg["vbat"]
        e = seg["e_pid_mm"]
        valid = np.isfinite(v) & np.isfinite(e) & (v > 0)
        if valid.sum() == 0:
            continue
        color = traj_color(seg["traj_type"])
        ax.scatter(v[valid], e[valid], s=10, alpha=0.6, color=color,
                   label=seg["traj_type"] if seg == all_segs[0] else "")
        all_v.extend(v[valid].tolist())
        all_e.extend(e[valid].tolist())

    if len(all_v) >= 2:
        slope, intercept, r_val, p_val, _ = linregress(all_v, all_e)
        v_fit = np.linspace(min(all_v), max(all_v), 100)
        ax.plot(v_fit, slope * v_fit + intercept,
                color=COLORS["vicon"], linewidth=1.5, linestyle="--",
                label=f"Linear fit (R²={r_val**2:.2f})")
        # Annotate slope
        slope_per_01v = slope * (-0.1)  # mm per -0.1V (voltage drop)
        ax.annotate(f"{slope_per_01v:.1f} mm per 0.1V drop",
                    xy=(0.05, 0.92), xycoords="axes fraction", fontsize=8)

    ax.set_xlabel("Battery voltage (V)")
    ax.set_ylabel("Mean tracking error (mm)")
    ax.set_title("Tracking error vs battery voltage")
    ax.invert_xaxis()

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8)

    fig.tight_layout()
    return fig


def plot_3d_trajectory(result: Dict) -> plt.Figure:
    """Figure 3: 3D trajectory comparison — Vicon vs setpoint, colored by error."""
    fig = plt.figure(figsize=DOUBLE_COL_TALL)
    ax = fig.add_subplot(111, projection="3d")

    px = result["_px"]
    py = result["_py"]
    pz = result["_pz"]
    sx = result["_sp_x"]
    sy = result["_sp_y"]
    sz = result["_sp_z"]
    e  = result["_e_pos"]

    # Setpoint path (grey)
    ax.plot(sx, sy, sz, color=COLORS["setpoint"], linewidth=1.0,
            linestyle="--", label="Setpoint", alpha=0.7)

    # Vicon path colored by tracking error
    norm_e = (e - e.min()) / (e.max() - e.min() + 1e-9)
    cmap = cm.get_cmap("viridis")
    for i in range(len(px) - 1):
        color = cmap(norm_e[i])
        ax.plot(px[i:i+2], py[i:i+2], pz[i:i+2],
                color=color, linewidth=1.5, alpha=0.8)

    # Colorbar proxy
    sm = cm.ScalarMappable(
        cmap="viridis",
        norm=plt.Normalize(vmin=e.min()*1000, vmax=e.max()*1000)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label("Tracking error (mm)", fontsize=9)

    # Start marker
    ax.scatter([px[0]], [py[0]], [pz[0]], c="green", s=30, zorder=5, label="Start")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"3D trajectory: {result['traj_id']}")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def plot_error_timeseries(result: Dict) -> plt.Figure:
    """Figure 4: Tracking error over time with battery voltage on secondary axis."""
    t = result["_t"]
    e = result["_e_pos"] * 1000  # convert to mm
    v = result["_vbat"]

    fig, ax1, ax2 = dual_axis_timeseries(
        t, e, v,
        y1_label="Position error (mm)",
        y2_label="Battery voltage (V)",
        y1_color=COLORS["error"],
        y2_color=COLORS["voltage"],
        figsize=DOUBLE_COL,
        title=f"Tracking error vs battery voltage: {result['traj_id']}",
    )

    # Voltage threshold lines
    add_voltage_vlines(ax1, t, v, thresholds=[3.8], colors=["red"])
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def build_table(all_results: List[Dict]) -> pd.DataFrame:
    """Per-trajectory performance table."""
    rows = []
    for r in all_results:
        rows.append({
            "Trajectory":        r["traj_id"],
            "Type":              r["trajectory_type"],
            "Pos RMSE (mm)":     f"{r['pos_rmse_mm']:.1f}",
            "Yaw RMSE (°)":      f"{r['yaw_rmse_deg']:.2f}",
            "P95 error (mm)":    f"{r['pos_p95_mm']:.1f}",
            "SS error (mm)":     f"{r['pos_ss_err_mm']:.1f}",
            "FF RMSE (mm)†":     f"{r['ff_pos_rmse_mm']:.1f}",
            "Mean Vbat (V)":     f"{r['mean_vbat']:.3f}",
        })

    # Aggregate row
    if all_results:
        rows.append({
            "Trajectory":        "\\textbf{All}",
            "Type":              "—",
            "Pos RMSE (mm)":     f"{np.mean([r['pos_rmse_mm'] for r in all_results]):.1f}",
            "Yaw RMSE (°)":      f"{np.mean([r['yaw_rmse_deg'] for r in all_results]):.2f}",
            "P95 error (mm)":    f"{np.mean([r['pos_p95_mm'] for r in all_results]):.1f}",
            "SS error (mm)":     f"{np.mean([r['pos_ss_err_mm'] for r in all_results]):.1f}",
            "FF RMSE (mm)†":     f"{np.mean([r['ff_pos_rmse_mm'] for r in all_results]):.1f}",
            "Mean Vbat (V)":     f"{np.mean([r['mean_vbat'] for r in all_results]):.3f}",
        })

    df = pd.DataFrame(rows)
    df.set_index("Trajectory", inplace=True)
    return df


def save_table(df: pd.DataFrame, name: str, results_dir: Path) -> None:
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / f"{name}.csv")
    latex = df.to_latex(
        escape=False,
        caption=(
            "Controller tracking performance per trajectory. "
            "†Simulated voltage feedforward (not from re-flown data). "
        ),
        label="tab:control",
    )
    with open(tables_dir / f"{name}.tex", "w") as f:
        f.write(latex)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(trajectories: Optional[List[TrajectoryData]] = None) -> Dict:
    """Run Task 2 end-to-end."""
    setup_plotting()
    results_dir = Path(__file__).resolve().parent / "results"

    if trajectories is None:
        trajectories = discover_datasets()

    if not trajectories:
        logger.error("Task 2: No trajectories found, aborting.")
        return {}

    logger.info(f"Task 2: evaluating {len(trajectories)} trajectory/ies")

    all_results = []
    all_segs = []

    for traj in trajectories:
        logger.info(f"  Processing {traj.traj_id} ...")
        res = evaluate_trajectory(traj)
        if res is None:
            continue
        all_results.append(res)
        segs = compute_voltage_segments(res)
        all_segs.append(segs)

    if not all_results:
        logger.error("Task 2: No valid results produced.")
        return {}

    # --- Figure 1: Trajectory difficulty ---
    fig1 = plot_trajectory_difficulty(all_results)
    p1 = save_fig(fig1, "task2_trajectory_difficulty.pdf")
    logger.info(f"  Saved {p1}")

    # --- Figure 2: Error vs voltage ---
    fig2 = plot_error_vs_voltage(all_segs)
    p2 = save_fig(fig2, "task2_error_vs_voltage.pdf")
    logger.info(f"  Saved {p2}")

    # --- Figure 3: 3D trajectory (best = widest voltage range) ---
    best = get_best_trajectory(trajectories, prefer_type="circle")
    best_res = next((r for r in all_results if r["traj_id"] == best.traj_id), all_results[0])
    fig3 = plot_3d_trajectory(best_res)
    p3 = save_fig(fig3, "task2_3d_trajectory.pdf")
    logger.info(f"  Saved {p3}")

    # --- Figure 4: Error timeseries ---
    fig4 = plot_error_timeseries(best_res)
    p4 = save_fig(fig4, "task2_error_timeseries.pdf")
    logger.info(f"  Saved {p4}")

    # --- Table ---
    table_df = build_table(all_results)
    save_table(table_df, "task2_performance_table", results_dir)
    logger.info("  Saved task2_performance_table.tex/.csv")

    # --- Summary ---
    # Compute slope: mm per 0.1V
    all_v = np.concatenate([s["vbat"] for s in all_segs])
    all_e = np.concatenate([s["e_pid_mm"] for s in all_segs])
    valid = np.isfinite(all_v) & np.isfinite(all_e) & (all_v > 0)
    summary = {}
    if valid.sum() >= 2:
        slope, _, r_val, _, _ = linregress(all_v[valid], all_e[valid])
        summary["mm_per_01v_drop"] = float(slope * (-0.1))
        summary["r2"] = float(r_val**2)

    return {
        "all_results": all_results,
        "table": table_df,
        "summary": summary,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run()

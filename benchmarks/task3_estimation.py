# -*- coding: utf-8 -*-
"""
task3_estimation.py — State Estimation Validation
==================================================

Scientific question: Does onboard EKF accuracy degrade with battery voltage,
and how does it compare to simpler baselines?

Three estimators evaluated:
  1. Dead reckoning — IMU-only integration, expected to drift significantly
  2. Complementary filter — attitude only (alpha=0.98 gyro/accel fusion)
  3. Onboard EKF — est_stateEstimate columns from aligned.csv

Metrics follow TUM/EuRoC format:
  - ATE (Absolute Trajectory Error) after Umeyama alignment
  - RTE (Relative Trajectory Error) at 0.5m, 1.0m, 2.0m path-length windows
  - Velocity estimation RMSE
  - Windowed ATE vs battery voltage

Outputs
-------
  results/task3_trajectory_overlay.pdf
  results/task3_ate_vs_voltage.pdf
  results/task3_error_breakdown.pdf
  results/task3_velocity_error.pdf
  results/tables/task3_evaluation_table.tex
  results/tables/task3_evaluation_table.csv
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.stats import linregress
import matplotlib.pyplot as plt

_BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(_BENCH))
sys.path.insert(0, str(_BENCH.parent / "src"))

from utils.data_loader import TrajectoryData, discover_datasets, get_best_trajectory
from utils.metrics import (
    rmse,
    compute_ate,
    compute_rte,
    compute_windowed_ate,
    umeyama_align,
)
from utils.plotting import (
    setup_plotting,
    save_fig,
    SINGLE_COL,
    DOUBLE_COL,
    DOUBLE_COL_TALL,
    COLORS,
    dual_axis_timeseries,
    add_voltage_vlines,
)

logger = logging.getLogger(__name__)

G_VEC = np.array([0.0, 0.0, 9.81])  # gravity in world frame (NED convention: z-up = positive)
G_SCALAR = 9.81
ALPHA_CF = 0.98  # complementary filter gyro weight


# ---------------------------------------------------------------------------
# Dead reckoning: IMU integration
# ---------------------------------------------------------------------------

def dead_reckoning(df: pd.DataFrame) -> np.ndarray:
    """
    Integrate IMU to produce a position estimate.

    Assumptions:
    - IMU accelerometer values are in g units → multiply by 9.81 for m/s²
    - IMU is in body frame; initial orientation from Vicon quaternion at row 0
    - Gravity is subtracted in world frame after rotation

    Returns
    -------
    dr_pos : (N, 3) dead-reckoning position [meters], starts from Vicon initial position
    """
    n = len(df)
    t = df["t"].values

    # Initial state from Vicon
    pos = np.array([df["px"].iloc[0], df["py"].iloc[0], df["pz"].iloc[0]], dtype=float)
    vel = np.zeros(3, dtype=float)
    q   = Rotation.from_quat([
        df["qx"].iloc[0], df["qy"].iloc[0],
        df["qz"].iloc[0], df["qw"].iloc[0],   # scipy [x,y,z,w]
    ])

    dr_pos = np.zeros((n, 3), dtype=float)
    dr_pos[0] = pos

    for i in range(1, n):
        dt = float(t[i] - t[i-1])
        if dt <= 0 or dt > 0.5:   # guard against bad timestamps / gaps
            dt = 0.005

        # Body-frame acceleration (g units → m/s²)
        acc_body = np.array([
            df["imu_acc_x"].iloc[i],
            df["imu_acc_y"].iloc[i],
            df["imu_acc_z"].iloc[i],
        ]) * G_SCALAR

        # Gyroscope (rad/s)
        gyro = np.array([
            df["imu_gyro_x"].iloc[i],
            df["imu_gyro_y"].iloc[i],
            df["imu_gyro_z"].iloc[i],
        ])

        # Rotate acceleration to world frame, remove gravity
        acc_world = q.apply(acc_body) - G_VEC

        # Integrate
        vel += acc_world * dt
        pos  = pos + vel * dt
        dr_pos[i] = pos

        # Update orientation via gyro integration (small-angle approximation)
        angle = np.linalg.norm(gyro) * dt
        if angle > 1e-10:
            axis = gyro / (np.linalg.norm(gyro) + 1e-12)
            dq   = Rotation.from_rotvec(axis * angle)
            q    = q * dq

    return dr_pos


# ---------------------------------------------------------------------------
# Complementary filter: attitude only
# ---------------------------------------------------------------------------

def complementary_filter_attitude(df: pd.DataFrame) -> np.ndarray:
    """
    Complementary filter for attitude estimation.

    Fuses gyroscope (high-frequency) with accelerometer tilt correction
    (low-frequency).  Position is not estimated here — use Vicon position
    for position-based metrics.

    Returns
    -------
    cf_euler : (N, 3) roll/pitch/yaw in degrees [roll, pitch, yaw]
    """
    n = len(df)
    t = df["t"].values

    # Initialize from Vicon quaternion
    q = Rotation.from_quat([
        df["qx"].iloc[0], df["qy"].iloc[0],
        df["qz"].iloc[0], df["qw"].iloc[0],
    ])

    cf_euler = np.zeros((n, 3), dtype=float)
    euler0 = q.as_euler("xyz", degrees=True)
    cf_euler[0] = euler0  # roll, pitch, yaw

    for i in range(1, n):
        dt = float(t[i] - t[i-1])
        if dt <= 0 or dt > 0.5:
            dt = 0.005

        gyro = np.array([
            df["imu_gyro_x"].iloc[i],
            df["imu_gyro_y"].iloc[i],
            df["imu_gyro_z"].iloc[i],
        ])
        acc_body = np.array([
            df["imu_acc_x"].iloc[i],
            df["imu_acc_y"].iloc[i],
            df["imu_acc_z"].iloc[i],
        ])

        # Gyro integration
        angle = np.linalg.norm(gyro) * dt
        if angle > 1e-10:
            axis = gyro / (np.linalg.norm(gyro) + 1e-12)
            dq   = Rotation.from_rotvec(axis * angle)
            q_gyro = q * dq
        else:
            q_gyro = q

        # Accelerometer tilt correction (only when magnitude is reasonable)
        acc_mag = np.linalg.norm(acc_body)
        if 0.5 < acc_mag < 2.0:  # in g units; ~0.5g to 2g acceptable
            acc_norm = acc_body / acc_mag
            # Estimated gravity direction in world frame from gyro-integrated orientation
            g_est = q_gyro.apply(np.array([0.0, 0.0, 1.0]))  # z-axis in world → what IMU should see
            # Cross product gives correction axis
            corr_axis = np.cross(g_est, acc_norm)
            corr_angle = np.arcsin(np.clip(np.linalg.norm(corr_axis), 0.0, 1.0))
            if np.linalg.norm(corr_axis) > 1e-10:
                corr_axis_n = corr_axis / (np.linalg.norm(corr_axis) + 1e-12)
                q_corr = Rotation.from_rotvec(
                    corr_axis_n * corr_angle * (1.0 - ALPHA_CF)
                )
                q = q_corr * q_gyro
            else:
                q = q_gyro
        else:
            q = q_gyro

        euler = q.as_euler("xyz", degrees=True)
        cf_euler[i] = euler

    return cf_euler  # (N, 3): roll, pitch, yaw in degrees


# ---------------------------------------------------------------------------
# Per-trajectory evaluation
# ---------------------------------------------------------------------------

def evaluate_trajectory(traj: TrajectoryData) -> Optional[Dict]:
    """Evaluate all three estimators on a single trajectory."""
    df = traj.flight_df  # active-flight rows only
    if len(df) < 200:
        logger.warning(f"{traj.traj_id}: too few flight samples ({len(df)}), skipping")
        return None

    logger.info(f"  {traj.traj_id}: computing dead reckoning ...")
    dr_pos = dead_reckoning(df)

    logger.info(f"  {traj.traj_id}: computing complementary filter ...")
    cf_euler = complementary_filter_attitude(df)  # (N, 3) degrees

    # Ground truth
    p_gt  = df[["px", "py", "pz"]].values                       # (N, 3)
    v_gt  = df[["vx", "vy", "vz"]].values                       # (N, 3)
    t     = df["t"].values                                        # (N,)

    # Vicon Euler (for attitude comparison)
    gt_roll  = df["roll"].values
    gt_pitch = df["pitch"].values
    gt_yaw   = df["yaw"].values

    # EKF
    p_ekf = df[["est_stateEstimate_x", "est_stateEstimate_y",
                "est_stateEstimate_z"]].values
    v_ekf = df[["est_stateEstimate_vx", "est_stateEstimate_vy",
                "est_stateEstimate_vz"]].values

    # EKF attitude (degrees from aligned.csv)
    ekf_roll  = df["att_stateEstimate_roll"].values
    ekf_pitch = df["att_stateEstimate_pitch"].values
    ekf_yaw   = df["att_stateEstimate_yaw"].values

    # Voltage
    vbat = df["pwr_pm_vbat"].values

    # --- ATE ---
    ate_dr  = compute_ate(dr_pos, p_gt, align=True)
    ate_ekf = compute_ate(p_ekf, p_gt, align=True)

    # --- RTE ---
    rte_dr  = compute_rte(dr_pos, p_gt, window_lengths_m=[0.5, 1.0, 2.0])
    rte_ekf = compute_rte(p_ekf,  p_gt, window_lengths_m=[0.5, 1.0, 2.0])

    # --- Velocity RMSE (EKF only — dead reckoning velocity is noisy) ---
    v_err = np.linalg.norm(v_ekf - v_gt, axis=1)
    vel_rmse = float(np.sqrt(np.mean(v_err**2)))

    # --- Attitude RMSE (EKF and complementary filter): sqrt of mean of (roll² + pitch² + yaw²) per axis RMSE ---
    att_err_ekf = np.sqrt(
        (np.mean(_angle_diff(ekf_roll,  gt_roll)**2) +
         np.mean(_angle_diff(ekf_pitch, gt_pitch)**2) +
         np.mean(_angle_diff(ekf_yaw,   gt_yaw)**2)) / 3.0
    )
    att_err_cf = np.sqrt(
        (np.mean(_angle_diff(cf_euler[:,0], gt_roll)**2) +
         np.mean(_angle_diff(cf_euler[:,1], gt_pitch)**2) +
         np.mean(_angle_diff(cf_euler[:,2], gt_yaw)**2)) / 3.0
    )

    # --- Windowed ATE vs voltage ---
    win_n  = min(1000, len(p_gt) // 4)  # 5s at 200Hz; adapt to short trajectories
    step_n = win_n // 2
    win_centers, win_ate = compute_windowed_ate(p_ekf, p_gt,
                                                window_samples=win_n,
                                                step_samples=step_n)
    win_vbat = np.array([float(np.mean(vbat[c - win_n//2: c + win_n//2]))
                         for c in win_centers])

    # --- Aligned EKF for error breakdown ---
    _, _, p_ekf_aligned = umeyama_align(p_ekf, p_gt)
    e_xyz = p_ekf_aligned - p_gt   # (N, 3): ex, ey, ez

    return {
        "traj_id":       traj.traj_id,
        "traj_type":     traj.trajectory_type,
        # ATE
        "ate_dr":        ate_dr,
        "ate_ekf":       ate_ekf,
        # RTE (1m window for table)
        "rte_dr_1m":     rte_dr.get("1.0", {}),
        "rte_ekf_1m":    rte_ekf.get("1.0", {}),
        # Velocity
        "vel_rmse_ekf":  vel_rmse,
        # Attitude
        "att_rmse_ekf":  float(att_err_ekf),
        "att_rmse_cf":   float(att_err_cf),
        # Raw arrays for plotting
        "_t":            t,
        "_p_gt":         p_gt,
        "_p_dr":         dr_pos,
        "_p_ekf":        p_ekf,
        "_p_ekf_aligned":p_ekf_aligned,
        "_v_gt":         v_gt,
        "_v_ekf":        v_ekf,
        "_e_xyz":        e_xyz,
        "_v_err":        v_err,
        "_vbat":         vbat,
        "_win_ate":      win_ate,
        "_win_vbat":     win_vbat,
        "_att_err_ekf":  np.sqrt(_angle_diff(ekf_roll, gt_roll)**2 +
                                 _angle_diff(ekf_pitch, gt_pitch)**2 +
                                 _angle_diff(ekf_yaw, gt_yaw)**2),
        "_cf_euler":     cf_euler,
        "_gt_euler":     np.column_stack([gt_roll, gt_pitch, gt_yaw]),
        # Voltage binning for table
        "_vbat_full":    vbat,
        "_ate_fresh":    _ate_subset(p_ekf, p_gt, vbat, lo=4.0),
        "_ate_depleted": _ate_subset(p_ekf, p_gt, vbat, hi=3.8),
    }


def _angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angular difference wrapped to [-180, 180] degrees."""
    diff = np.asarray(a) - np.asarray(b)
    return (diff + 180.0) % 360.0 - 180.0


def _ate_subset(p_est: np.ndarray, p_gt: np.ndarray,
                vbat: np.ndarray, lo: float = 0.0,
                hi: float = float("inf")) -> Dict:
    """Compute ATE on the voltage-filtered subset."""
    mask = (vbat >= lo) & (vbat < hi)
    if mask.sum() < 10:
        return {"mean": float("nan"), "rmse": float("nan")}
    return compute_ate(p_est[mask], p_gt[mask], align=False)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_trajectory_overlay(result: Dict) -> plt.Figure:
    """Figure 1: Top-down (XY) view — Vicon vs EKF vs dead reckoning."""
    fig, ax = plt.subplots(figsize=DOUBLE_COL)

    p_gt  = result["_p_gt"]
    p_dr  = result["_p_dr"]
    p_ekf = result["_p_ekf"]

    # Axis limits: clip to 1.5× Vicon extent (dead reckoning can diverge far)
    x_range = p_gt[:, 0].max() - p_gt[:, 0].min()
    y_range = p_gt[:, 1].max() - p_gt[:, 1].min()
    cx, cy = p_gt[:, 0].mean(), p_gt[:, 1].mean()
    margin = max(x_range, y_range) * 0.8
    ax.set_xlim(cx - margin, cx + margin)
    ax.set_ylim(cy - margin, cy + margin)

    ax.plot(p_gt[:, 0],  p_gt[:, 1],  color=COLORS["vicon"],
            linewidth=1.5, label="Vicon (ground truth)", zorder=5)
    ax.plot(p_ekf[:, 0], p_ekf[:, 1], color=COLORS["ekf"],
            linewidth=1.2, label="Onboard EKF", zorder=4, alpha=0.9)
    ax.plot(p_dr[:, 0],  p_dr[:, 1],  color=COLORS["dead_reckoning"],
            linewidth=0.8, label="Dead reckoning", zorder=3, alpha=0.6)

    ax.scatter(p_gt[0, 0], p_gt[0, 1], c="green", s=40, zorder=6, label="Start")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Trajectory overlay (top-down view): {result['traj_id']}")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_ate_vs_voltage(result: Dict) -> plt.Figure:
    """Figure 2: Windowed ATE and attitude error vs battery voltage (2 subplots)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=DOUBLE_COL)

    win_ate  = result["_win_ate"]
    win_vbat = result["_win_vbat"]
    att_err  = result["_att_err_ekf"]
    t        = result["_t"]
    vbat     = result["_vbat"]

    # Left: position ATE per window
    valid = np.isfinite(win_vbat) & np.isfinite(win_ate)
    if valid.sum() >= 2:
        ax1.scatter(win_vbat[valid], win_ate[valid] * 1000,
                    s=12, alpha=0.7, color=COLORS["ekf"])
        slope1, intercept1, r1, _, _ = linregress(win_vbat[valid], win_ate[valid] * 1000)
        v_fit = np.linspace(win_vbat[valid].min(), win_vbat[valid].max(), 100)
        ax1.plot(v_fit, slope1 * v_fit + intercept1,
                 color=COLORS["our_model"], linewidth=1.5,
                 label=f"Fit (R²={r1**2:.2f})")
        ax1.annotate(f"{slope1*(-0.1):.1f} mm/0.1V",
                     xy=(0.05, 0.92), xycoords="axes fraction", fontsize=8)

    ax1.set_xlabel("Battery voltage (V)")
    ax1.set_ylabel("Position ATE (mm)")
    ax1.set_title("EKF position ATE vs voltage")
    ax1.invert_xaxis()
    ax1.legend(fontsize=8)

    # Right: attitude error per sample (subsampled) vs vbat
    step = max(1, len(att_err) // 200)
    valid2 = np.isfinite(att_err[::step]) & np.isfinite(vbat[::step])
    v_sub  = vbat[::step][valid2]
    e_sub  = att_err[::step][valid2]
    if len(v_sub) >= 2:
        ax2.scatter(v_sub, e_sub, s=6, alpha=0.4, color=COLORS["complementary"],
                    edgecolors="none")
        slope2, intercept2, r2, _, _ = linregress(v_sub, e_sub)
        v_fit2 = np.linspace(v_sub.min(), v_sub.max(), 100)
        ax2.plot(v_fit2, slope2 * v_fit2 + intercept2,
                 color=COLORS["our_model"], linewidth=1.5,
                 label=f"Fit (R²={r2**2:.2f})")

    ax2.set_xlabel("Battery voltage (V)")
    ax2.set_ylabel("Attitude RMS error (°)")
    ax2.set_title("EKF attitude error vs voltage")
    ax2.invert_xaxis()
    ax2.legend(fontsize=8)

    fig.tight_layout()
    return fig


def plot_error_breakdown(result: Dict) -> plt.Figure:
    """Figure 3: EKF position error components over time."""
    t     = result["_t"]
    e_xyz = result["_e_xyz"]  # (N, 3): ex, ey, ez [meters]
    e_tot = np.linalg.norm(e_xyz, axis=1)
    vbat  = result["_vbat"]
    t_s   = t - t[0]

    fig, axes = plt.subplots(4, 1, figsize=DOUBLE_COL_TALL, sharex=True)
    labels = ["x-error (m)", "y-error (m)", "z-error (m)", "||error|| (m)"]
    data   = [e_xyz[:, 0], e_xyz[:, 1], e_xyz[:, 2], e_tot]
    clrs   = [COLORS["ekf"], COLORS["ekf"], COLORS["ekf"], COLORS["our_model"]]

    for ax, y, lab, col in zip(axes, data, labels, clrs):
        ax.plot(t_s, y, color=col, linewidth=1.0)
        ax.set_ylabel(lab, fontsize=9)
        # Voltage threshold line
        add_voltage_vlines(ax, t, vbat, thresholds=[3.8], colors=["red"])

    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title(f"EKF error breakdown: {result['traj_id']}")
    fig.tight_layout()
    return fig


def plot_velocity_error(result: Dict) -> plt.Figure:
    """Figure 4: Velocity error magnitude vs time with battery voltage overlay."""
    t     = result["_t"]
    v_err = result["_v_err"] * 1000  # → mm/s... actually keep in m/s
    v_err = result["_v_err"]         # m/s
    vbat  = result["_vbat"]

    fig, ax1, ax2 = dual_axis_timeseries(
        t, v_err, vbat,
        y1_label="Velocity error ||v_ekf - v_gt|| (m/s)",
        y2_label="Battery voltage (V)",
        y1_color=COLORS["ekf"],
        y2_color=COLORS["voltage"],
        figsize=SINGLE_COL,
        title=f"Velocity estimation error: {result['traj_id']}",
    )
    return fig


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def build_table(all_results: List[Dict]) -> pd.DataFrame:
    """TUM/EuRoC-style evaluation table."""
    rows = []
    for res in all_results:
        def _fmt(d, key, scale=1.0, fmt=".3f"):
            val = d.get(key, float("nan"))
            if isinstance(val, dict):
                val = val.get("mean", float("nan"))
            return f"{float(val)*scale:{fmt}}" if np.isfinite(float(val)) else "—"

        # Dead reckoning
        rows.append({
            "Estimator":         "Dead Reckoning",
            "Trajectory":        res["traj_id"],
            "ATE mean (m)":      _fmt(res["ate_dr"],    "mean"),
            "ATE std (m)":       _fmt(res["ate_dr"],    "std"),
            "ATE RMSE (m)":      _fmt(res["ate_dr"],    "rmse"),
            "RTE 1m trans (m)":  _fmt(res["rte_dr_1m"], "trans_mean"),
            "Vel RMSE (m/s)":    "—",
            "Att RMSE (°)":      "—",
        })
        # Complementary filter
        rows.append({
            "Estimator":         "Comp. Filter (att.)",
            "Trajectory":        res["traj_id"],
            "ATE mean (m)":      "† (Vicon pos)",
            "ATE std (m)":       "—",
            "ATE RMSE (m)":      "—",
            "RTE 1m trans (m)":  "—",
            "Vel RMSE (m/s)":    "—",
            "Att RMSE (°)":      f"{res['att_rmse_cf']:.2f}",
        })
        # EKF
        rows.append({
            "Estimator":         "Onboard EKF",
            "Trajectory":        res["traj_id"],
            "ATE mean (m)":      _fmt(res["ate_ekf"],    "mean"),
            "ATE std (m)":       _fmt(res["ate_ekf"],    "std"),
            "ATE RMSE (m)":      _fmt(res["ate_ekf"],    "rmse"),
            "RTE 1m trans (m)":  _fmt(res["rte_ekf_1m"], "trans_mean"),
            "Vel RMSE (m/s)":    f"{res['vel_rmse_ekf']:.3f}",
            "Att RMSE (°)":      f"{res['att_rmse_ekf']:.2f}",
        })

    df = pd.DataFrame(rows)
    df.set_index(["Estimator", "Trajectory"], inplace=True)
    return df


def save_table(df: pd.DataFrame, name: str, results_dir: Path) -> None:
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / f"{name}.csv")
    latex = df.to_latex(
        escape=False,
        caption=(
            "State estimation evaluation (TUM/EuRoC format). "
            "†Complementary filter uses Vicon position; only attitude is evaluated. "
            "ATE computed after Umeyama alignment."
        ),
        label="tab:estimation",
    )
    with open(tables_dir / f"{name}.tex", "w") as f:
        f.write(latex)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(trajectories: Optional[List[TrajectoryData]] = None) -> Dict:
    """Run Task 3 end-to-end."""
    setup_plotting()
    results_dir = Path(__file__).resolve().parent / "results"

    if trajectories is None:
        trajectories = discover_datasets()

    if not trajectories:
        logger.error("Task 3: No trajectories found, aborting.")
        return {}

    logger.info(f"Task 3: evaluating {len(trajectories)} trajectory/ies")

    all_results = []
    for traj in trajectories:
        logger.info(f"  Processing {traj.traj_id} ...")
        res = evaluate_trajectory(traj)
        if res is None:
            continue
        all_results.append(res)

    if not all_results:
        logger.error("Task 3: No valid results produced.")
        return {}

    # Pick best trajectory for single-trajectory plots
    best = get_best_trajectory(trajectories, prefer_type="circle")
    best_res = next((r for r in all_results if r["traj_id"] == best.traj_id), all_results[0])

    # --- Figure 1: Trajectory overlay ---
    fig1 = plot_trajectory_overlay(best_res)
    p1 = save_fig(fig1, "task3_trajectory_overlay.pdf")
    logger.info(f"  Saved {p1}")

    # --- Figure 2: ATE vs voltage ---
    fig2 = plot_ate_vs_voltage(best_res)
    p2 = save_fig(fig2, "task3_ate_vs_voltage.pdf")
    logger.info(f"  Saved {p2}")

    # --- Figure 3: Error breakdown ---
    fig3 = plot_error_breakdown(best_res)
    p3 = save_fig(fig3, "task3_error_breakdown.pdf")
    logger.info(f"  Saved {p3}")

    # --- Figure 4: Velocity error ---
    fig4 = plot_velocity_error(best_res)
    p4 = save_fig(fig4, "task3_velocity_error.pdf")
    logger.info(f"  Saved {p4}")

    # --- Table ---
    table_df = build_table(all_results)
    save_table(table_df, "task3_evaluation_table", results_dir)
    logger.info("  Saved task3_evaluation_table.tex/.csv")

    # --- Summary ---
    summary = {}
    for res in all_results:
        fresh_ate     = res.get("_ate_fresh",    {}).get("rmse", float("nan"))
        depleted_ate  = res.get("_ate_depleted", {}).get("rmse", float("nan"))
        if np.isfinite(fresh_ate) and np.isfinite(depleted_ate) and fresh_ate > 0:
            pct = (depleted_ate - fresh_ate) / fresh_ate * 100
            summary[res["traj_id"]] = {
                "ate_fresh_mm":    float(fresh_ate) * 1000,
                "ate_depleted_mm": float(depleted_ate) * 1000,
                "pct_degradation": pct,
            }

    return {
        "all_results": all_results,
        "table": table_df,
        "summary": summary,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run()

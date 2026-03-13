# -*- coding: utf-8 -*-
"""
run_all.py — NanoBench: Run all three benchmark tasks
======================================================

Usage:
    cd benchmarks/
    python run_all.py

Runs Task 1 (system identification), Task 2 (controller benchmarking), and
Task 3 (state estimation validation) in sequence.  Each task is run inside a
try/except so a failure in one task does not prevent the others from running.

Outputs all figures and tables to benchmarks/results/ and writes a
human-readable summary_report.txt with the key quantified findings.
"""

import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict

# Ensure benchmarks/ and visualization/ are on the path
_BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(_BENCH))
sys.path.insert(0, str(_BENCH.parent / "src"))
sys.path.insert(0, str(_BENCH.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_all")

# ---------------------------------------------------------------------------
# Import tasks
# ---------------------------------------------------------------------------
from utils.data_loader import discover_datasets

import task1_sysid
import task2_control
import task3_estimation

try:
    from visualization import plot_dataset as _plot_dataset_module
    _HAS_PLOT_DATASET = True
except Exception:
    _HAS_PLOT_DATASET = False

RESULTS_DIR = _BENCH / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_task(name: str, task_module, trajectories) -> Dict:
    """Run a task module's run() function, catching any exceptions."""
    logger.info("=" * 60)
    logger.info(f"  Starting {name}")
    logger.info("=" * 60)
    try:
        result = task_module.run(trajectories)
        logger.info(f"  {name} completed successfully.")
        return result or {}
    except Exception:
        logger.error(f"  {name} FAILED:")
        logger.error(traceback.format_exc())
        return {}


def write_summary(r1: Dict, r2: Dict, r3: Dict) -> None:
    """Write a human-readable text summary of all key findings."""
    lines = [
        "=" * 70,
        "  NanoBench Benchmark Summary Report",
        f"  Generated: {datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "=" * 70,
        "",
        "TASK 1: SYSTEM IDENTIFICATION",
        "-" * 40,
    ]

    if r1.get("summary"):
        for traj_id, s in r1["summary"].items():
            pct = s.get("rmse_reduction_depleted_pct", float("nan"))
            lines.append(
                f"  [{traj_id}] Voltage-conditioned model reduces acceleration "
                f"prediction RMSE by {pct:.1f}% vs constant-gain baseline "
                f"at depleted battery levels (Vbat < 3.8V)."
            )
        if r1.get("table") is not None:
            table = r1["table"]
            try:
                # Find overall RMSE for const and voltage models
                r_const = table.loc["Constant gain", "RMSE overall (m/s²)"]
                r_volt  = table.loc["Voltage-conditioned", "RMSE overall (m/s²)"]
                lines.append(
                    f"  Overall RMSE: Constant gain = {r_const} m/s², "
                    f"Voltage-conditioned = {r_volt} m/s²."
                )
            except Exception:
                pass
    else:
        lines.append("  [No results produced — check logs above]")

    lines += [
        "",
        "TASK 2: CONTROLLER BENCHMARKING",
        "-" * 40,
    ]

    if r2.get("summary"):
        s = r2["summary"]
        mm_per_01v = s.get("mm_per_01v_drop", float("nan"))
        r2_val     = s.get("r2", float("nan"))
        lines.append(
            f"  Each 0.1V drop in battery voltage corresponds to "
            f"{mm_per_01v:.2f} mm increase in position tracking error "
            f"(linear fit R² = {r2_val:.3f})."
        )
    if r2.get("all_results"):
        res_list = r2["all_results"]
        mean_rmse = sum(r["pos_rmse_mm"] for r in res_list) / len(res_list)
        lines.append(
            f"  Mean position RMSE across {len(res_list)} trajectory/ies: "
            f"{mean_rmse:.1f} mm."
        )
    if not r2.get("summary") and not r2.get("all_results"):
        lines.append("  [No results produced — check logs above]")

    lines += [
        "",
        "TASK 3: STATE ESTIMATION VALIDATION",
        "-" * 40,
    ]

    if r3.get("summary"):
        for traj_id, s in r3["summary"].items():
            ate_f  = s.get("ate_fresh_mm",    float("nan"))
            ate_d  = s.get("ate_depleted_mm", float("nan"))
            pct    = s.get("pct_degradation", float("nan"))
            lines.append(
                f"  [{traj_id}] EKF ATE: {ate_f:.1f} mm (fresh battery) → "
                f"{ate_d:.1f} mm (depleted), a {pct:.1f}% degradation."
            )
    if r3.get("all_results"):
        for res in r3["all_results"]:
            ate = res["ate_ekf"]["rmse"]
            lines.append(
                f"  [{res['traj_id']}] EKF ATE RMSE (full trajectory) = "
                f"{ate*1000:.1f} mm."
            )
            lines.append(
                f"  [{res['traj_id']}] Dead reckoning ATE RMSE = "
                f"{res['ate_dr']['rmse']*1000:.1f} mm (expected large drift)."
            )
    if not r3.get("summary") and not r3.get("all_results"):
        lines.append("  [No results produced — check logs above]")

    lines += [
        "",
        "OUTPUT FILES",
        "-" * 40,
    ]

    # List all produced files
    for f in sorted(RESULTS_DIR.rglob("*")):
        if f.is_file():
            lines.append(f"  {f.relative_to(RESULTS_DIR)}")

    lines.append("")
    report = "\n".join(lines)

    report_path = RESULTS_DIR / "summary_report.txt"
    with open(report_path, "w") as fh:
        fh.write(report)

    logger.info(f"\n{report}")
    logger.info(f"Summary written to {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Discover datasets once and share across all tasks
    logger.info("Discovering datasets ...")
    trajectories = discover_datasets()

    if not trajectories:
        logger.warning(
            "No datasets found. Make sure aligned.csv files exist under "
            f"{_BENCH.parent / 'datasets'} or the other candidate directories."
        )
        # Still run tasks — they will each print their own error messages
    else:
        logger.info(f"Found {len(trajectories)} trajectory/ies:")
        for t in trajectories:
            logger.info(f"  {t}")

    # Dataset visualization (static figures — no display required)
    if _HAS_PLOT_DATASET:
        logger.info("=" * 60)
        logger.info("  Starting Dataset Visualization")
        logger.info("=" * 60)
        try:
            _plot_dataset_module.plot_all(trajectories)
            logger.info("  Dataset Visualization completed successfully.")
        except Exception:
            logger.error("  Dataset Visualization FAILED:")
            logger.error(traceback.format_exc())

    r1 = run_task("Task 1: System Identification",   task1_sysid,     trajectories)
    r2 = run_task("Task 2: Controller Benchmarking", task2_control,   trajectories)
    r3 = run_task("Task 3: State Estimation",        task3_estimation, trajectories)

    write_summary(r1, r2, r3)


if __name__ == "__main__":
    main()

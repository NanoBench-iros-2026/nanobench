# NanoBench Benchmarks

Three reproducible benchmark tasks for the IROS 2026 paper on voltage-aware nano-quadrotor modelling.

**Central thesis:** Single-cell LiPo battery voltage decline measurably degrades thrust generation, tracking performance, and state estimation accuracy on the Crazyflie 2.1 — and this dataset captures it across the full discharge cycle.

---

## Quick Start

```bash
cd /home/coral/crazyswarm/ros_ws/src/nanobench/benchmarks
python3 run_all.py
```

All 19 output files land in `benchmarks/results/`. Run time: ~10 seconds on two trajectories.

Run individual tasks:
```bash
python3 task1_sysid.py
python3 task2_control.py
python3 task3_estimation.py
```

---

## Directory Layout

```
benchmarks/
├── run_all.py              # Entry point — runs all three tasks in sequence
├── task1_sysid.py          # Task 1: system identification
├── task2_control.py        # Task 2: controller benchmarking
├── task3_estimation.py     # Task 3: state estimation validation
├── utils/
│   ├── data_loader.py      # Dataset discovery and loading
│   ├── metrics.py          # RMSE, R², ATE, RTE, Umeyama alignment
│   └── plotting.py         # IEEE-ready matplotlib style, colors, save helpers
└── results/                # All outputs land here (auto-created)
    ├── tables/             # .tex and .csv tables
    ├── *.pdf               # Figures
    └── summary_report.txt  # Key numbers in plain text
```

---

## Input

### What the code reads

Every `aligned.csv` file found under these directories (searched in order):

| Priority | Path | Notes |
|---|---|---|
| 1 | `nanobench/datasets/` | Primary location |
| 2 | `nanobench/../../datasets/` | Alternate from experiment.yaml |
| 3 | `../crazyswarm/scripts/nanobench_dataset/` | Legacy location |

Each `aligned.csv` must sit inside a trajectory directory that also contains `metadata.yaml`.

### Required columns

| Column group | Columns | Source |
|---|---|---|
| Time | `t` | Vicon timestamp (Unix seconds) |
| Vicon ground truth | `px, py, pz, qx, qy, qz, qw, roll, pitch, yaw, vx, vy, vz` | Motion capture @ 200 Hz |
| IMU | `imu_acc_x/y/z` (g units), `imu_gyro_x/y/z` (rad/s) | Onboard @ 100 Hz |
| Motors | `motor_motor_m1/m2/m3/m4` (0–65535 PWM) | Onboard @ 100 Hz |
| EKF state | `est_stateEstimate_x/y/z/vx/vy/vz` | Onboard Kalman filter |
| EKF acceleration | `est_stateEstimate_ax/ay/az` | Onboard EKF (if logged) |
| EKF attitude | `att_stateEstimate_roll/pitch/yaw` (deg), `att_stateEstimate_qx/qy/qz/qw` | Onboard |
| Setpoint | `sp_ctrltarget_x/y/z/yaw` | Commanded positions |
| Battery | `pwr_pm_vbat` (V) | Onboard @ 10 Hz |

**Note:** If `att_stateEstimate_qw` is not in the aligned CSV (older datasets), it is automatically reconstructed as `sqrt(max(0, 1-qx²-qy²-qz²))`.

### What trajectories are skipped

- Fewer than 500 total rows
- All motor commands are zero (ground recording with no flight)
- `aligned.csv` missing or unreadable

---

## Output

### Files produced

```
results/
├── task1_thrust_curve.pdf                    # k(V) scatter + fitted quadratic
├── task1_prediction_error_vs_voltage.pdf     # RMSE vs voltage, 4 model lines
├── task1_timeseries_comparison.pdf           # gt acceleration vs 4 models
├── task2_trajectory_difficulty.pdf           # tracking RMSE per trajectory type
├── task2_error_vs_voltage.pdf                # tracking error scatter vs voltage
├── task2_3d_trajectory.pdf                   # 3D Vicon vs setpoint, colored by error
├── task2_error_timeseries.pdf                # error + voltage vs time
├── task3_trajectory_overlay.pdf             # XY top-down: Vicon, EKF, dead-reckoning
├── task3_ate_vs_voltage.pdf                  # windowed ATE vs voltage (2 panels)
├── task3_error_breakdown.pdf                 # EKF error components over time
├── task3_velocity_error.pdf                  # velocity estimation error vs time
├── tables/task1_model_comparison_table.tex/csv
├── tables/task2_performance_table.tex/csv
├── tables/task3_evaluation_table.tex/csv
└── summary_report.txt                        # key numbers in plain English
```

---

## Task 1 — System Identification

### What it does

Fits a **voltage-conditioned thrust model** and compares it against three baselines. Ground truth vertical acceleration is derived by double-differentiating Vicon `pz` with a Savitzky-Golay filter.

| Model | Description | Expected performance |
|---|---|---|
| Constant gain | `F = k × Σuᵢ²`, one scalar for all voltages | Degrades at low battery |
| Time-indexed | `F = k(t) × Σuᵢ²`, polynomial in time | Better than const, worse than voltage |
| Linear regression | `az = w·[m1,m2,m3,m4,Vbat]`, black-box | Competitive if Vbat correlation is strong |
| **Voltage-conditioned** | `F = k(V) × Σuᵢ²`, quadratic in Vbat | Best at depleted battery |

### Expected results

| Metric | Typical value | Notes |
|---|---|---|
| Constant gain RMSE | 2–10 m/s² | Increases as battery drains |
| Voltage-conditioned RMSE | 1–5 m/s² | Should be flat across voltage range |
| R² (voltage-conditioned) | > 0.85 | Below 0.6 means noisy data or short trajectory |
| RMSE improvement at < 3.8V | 20–60% | This is the paper's key claim |

### Ideal result

- **Thrust curve** (`task1_thrust_curve.pdf`): The scatter cloud of empirical `k` values should show a clear downward slope as voltage drops. The fitted quadratic should pass through the centre of the cloud. If the cloud is flat (no slope), the voltage effect is weak in your dataset.
- **Error vs voltage** (`task1_prediction_error_vs_voltage.pdf`): The constant-gain and time-indexed lines should diverge upward as voltage drops (right side of plot). The voltage-conditioned line should stay flat or decline least.
- **Time series** (`task1_timeseries_comparison.pdf`): In the depleted battery region, the voltage-conditioned model should track the ground truth most closely.

A good paper-quality result shows **≥ 30% RMSE reduction** at depleted battery for the voltage-conditioned model vs. constant gain.

---

## Task 2 — Controller Benchmarking

### What it does

Quantifies Crazyflie PID tracking performance across trajectory types and battery levels. Computes a **voltage-to-error regression** that quantifies how much tracking degrades per volt drop.

| Condition | Description |
|---|---|
| Standard PID | Direct from data: `e = ||p_actual - p_setpoint||` |
| Simulated feedforward | Post-hoc z-correction proportional to `V_nom / V_bat` (labeled as simulation) |

### Expected results

| Metric | Hover | Circle (slow) | Figure-eight (fast) |
|---|---|---|---|
| Position RMSE | 5–30 mm | 20–80 mm | 40–150 mm |
| Yaw RMSE | < 5° | 2–10° | 5–20° |
| 95th percentile error | 1.5–3× RMSE | typical | typical |

**Voltage slope:** A healthy result is **2–15 mm per 0.1V drop** with R² > 0.4. Values above 50 mm/0.1V usually indicate the recording includes a deep-discharge segment (battery below 3.3V) which is outside the normal operating range.

### Ideal result

- **Error timeseries** (`task2_error_timeseries.pdf`): You should see the orange battery voltage curve declining over time, and the blue error curve trending upward — the visual correlation is the most intuitive figure in the paper.
- **3D trajectory** (`task2_3d_trajectory.pdf`): The setpoint path (grey dashed) and Vicon path (colored) should be tightly aligned at the start of the trajectory (fresh battery) and diverge slightly at the end (depleted battery). The color gradient from blue→yellow shows where errors accumulate.
- **Error vs voltage scatter** (`task2_error_vs_voltage.pdf`): Points should cluster at low error (top left = fresh battery) and spread to higher error (bottom right = depleted). A clear positive linear trend with R² > 0.5 is publication-quality.

---

## Task 3 — State Estimation Validation

### What it does

Evaluates the onboard EKF against Vicon ground truth using TUM/EuRoC format metrics. Dead reckoning (IMU integration only) and a complementary attitude filter serve as baselines.

| Estimator | Position | Attitude |
|---|---|---|
| Dead reckoning | Integrated from IMU — drifts badly | Gyro integration |
| Complementary filter | Uses Vicon (not estimated) | Gyro + accel fusion (α=0.98) |
| **Onboard EKF** | `est_stateEstimate_x/y/z` | `att_stateEstimate_roll/pitch/yaw` |

### Expected results

| Estimator | ATE RMSE | Notes |
|---|---|---|
| Dead reckoning | Diverges to 100s of meters | Expected — shows why feedback is essential |
| Complementary filter (att) | 1–5° attitude error | Position not evaluated |
| Onboard EKF | **3–20 mm** | Vicon-aided, should be tight |

**EKF ATE voltage degradation:** A meaningful result is **5–30% ATE increase** from fresh to depleted battery. Less than 5% suggests the dataset lacks sufficient voltage range. More than 50% may indicate sensor issues during discharge.

### Ideal result

- **Trajectory overlay** (`task3_trajectory_overlay.pdf`): Vicon (black) and EKF (blue) lines should nearly overlap. Dead reckoning (pink) should diverge wildly off-screen within seconds — this is intentional and makes for the most visually striking figure in the paper.
- **ATE vs voltage** (`task3_ate_vs_voltage.pdf`): A positive slope on the scatter plot (higher ATE at lower voltage) with R² > 0.3 supports the paper's thesis. If R² < 0.1, the voltage effect on EKF is weak in your data (which is also a valid finding).
- **Error breakdown** (`task3_error_breakdown.pdf`): Z-error (altitude) is typically the largest component for Vicon-aided EKF. The vertical dashed red line at 3.8V should coincide with a visible step-up in error.

---

## Tunable Parameters

### `utils/data_loader.py`

| Constant | Default | Effect |
|---|---|---|
| `CRAZYFLIE_MASS_KG` | `0.027` | Mass in kg used for thrust force computation. Change if using a heavier CF (e.g. with deck: ~0.033 kg) |
| `motor_motor_m1 > 1000` (flight mask) | `1000` | Lower to capture more of takeoff; raise (e.g. 5000) to exclude slow-spin taxi |

### `task1_sysid.py`

| Constant | Default | Effect |
|---|---|---|
| `SG_WINDOW` | `15` | Savitzky-Golay window for pz double-differentiation. Larger = smoother acceleration but lags transients. Try 11–25. |
| `SG_ORDER` | `3` | Polynomial order for SG filter. Rarely needs changing. |
| `train_frac` in `train_test_split()` | `0.70` | Fraction of flight used for model fitting. Lower = more test data, but models may underfit. |
| `segment_s` in `compute_voltage_segments()` | `10.0` | Window size (seconds) for error-vs-voltage plot. Shorter = more points but noisier. |
| Polynomial degree for voltage model | `deg=2` in `fit_model_voltage()` | Quadratic. Try `deg=1` for linear, `deg=3` if you have wide voltage range. |

### `task2_control.py`

| Constant | Default | Effect |
|---|---|---|
| `V_NOM` | `3.87` | Nominal operating voltage for feedforward simulation. Change to match your typical hover voltage. |
| `V_NOM_CORRECTION` | `0.30` | How aggressively feedforward compensates (0 = no correction, 1 = full). Tune to match experimental compensation if available. |
| `segment_s` in `compute_voltage_segments()` | `10.0` | Segment length for voltage regression. |

### `task3_estimation.py`

| Constant | Default | Effect |
|---|---|---|
| `ALPHA_CF` | `0.98` | Complementary filter gyro weight. Higher = trust gyro more (less accel noise but more drift). Range: 0.95–0.999. |
| `window_lengths_m` | `[0.5, 1.0, 2.0]` | Path-length windows for RTE. Shorter windows stress short-term accuracy; longer capture cumulative drift. |
| `win_n` (windowed ATE) | `min(1000, N//4)` | Samples per ATE window (~5s at 200Hz). Increase for smoother curve; decrease if trajectory is short. |

---

## Interpreting the Current Results

The current dataset has an anomaly in **B2_circle_slow_rep01**: the battery voltage drops to **2.49V**, well below the safe minimum (3.2V from `experiment.yaml`). This suggests the recording includes a post-flight deep-discharge segment, which inflates all error metrics for that trajectory. The **B3_figure8_fast_rep01** results (EKF ATE = 4 mm) are physically realistic and represent what you should expect from healthy flights.

**What to do:**
1. Collect more trajectories across the full Category A–D library.
2. Ensure each recording captures the trajectory phase only (post-processing should trim pre/post flight).
3. For the paper, report results per voltage bin separately to isolate the voltage effect from the data quality issue.

---

## Adding More Data

Drop new experiment directories under `datasets/` following the naming convention:

```
datasets/
└── nanobench_v3/
    └── exp_YYYYMMDDTHHMMSSZ/
        └── <TRAJ_ID>_rep<NN>/
            ├── aligned.csv      ← required
            └── metadata.yaml   ← required (trajectory_type, category, etc.)
```

Re-run `python3 run_all.py` — new trajectories are discovered automatically.

---

## Including Figures in the Paper

All PDFs are IEEE column-ready. In `paper/main.tex`:

```latex
% Single-column figure (3.5 inches)
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{../benchmarks/results/task1_thrust_curve.pdf}
  \caption{Voltage-conditioned thrust gain $k(V)$.}
  \label{fig:thrust_curve}
\end{figure}

% Double-column figure (full 7.16 inches)
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{../benchmarks/results/task2_3d_trajectory.pdf}
  \caption{3D trajectory tracking error on B2\_circle\_slow.}
  \label{fig:3d_traj}
\end{figure*}
```

Tables in `results/tables/*.tex` can be included with `\input{../benchmarks/results/tables/task1_model_comparison_table.tex}`.

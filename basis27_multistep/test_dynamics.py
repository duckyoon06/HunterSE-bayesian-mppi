"""저장된 test 주행에서 모델의 연속 궤적 예측을 평가한다.


관측 변화율의 RMSE, MAE, Gaussian NLL, 95% 예측구간 포함률을 계산한다.
궤적은 기본 0.5초 간격의 시작점에서 0.5~2.5초를 예측한다.
각 구간의 첫 상태만 실제값을 사용한다. 이후에는 예측 상태와 기록된
제어 입력으로 step()을 반복한다. MPPI 제어 성능이 아닌 동역학 예측 평가다.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


from dynamics import BayesianDynamics, TARGET_NAMES, load_trajectory


STATE_NAMES = ("px", "py", "yaw", "roll", "vx", "vy", "yaw_rate")
PROJECT_DIR = Path(__file__).resolve().parent.parent
# 아래 값은 수치적으로 비정상적인 예측을 중단하기 위한 여유 있는 한계다.
# 차량의 안전 한계나 학습된 물리 파라미터가 아니다. 실행 옵션으로 변경 가능하다.
DEFAULT_LIMITS = {"speed_mps": 20.0, "yaw_rate_radps": 10.0, "roll_rad": 1.5}


def choose_windows(times, valid, horizon, count=0, start_interval=0.5):
    """구간 전체가 valid인 시작·종료 인덱스를 선택한다."""
    # 시작부터 horizon초 이상 지난 첫 샘플을 끝점으로 사용한다.
    stops = np.searchsorted(times, times + horizon)
    starts = np.flatnonzero(stops < len(times))

    # 누적합으로 [start, stop] 안에 무효한 샘플이 있는지 판정한다.
    invalid_count = np.r_[0, np.cumsum(~valid)]
    continuous = invalid_count[stops[starts] + 1] == invalid_count[starts]
    starts = starts[continuous]
    if len(starts) == 0:
        return []

    if count > 0:
        # 빠른 점검: 예전처럼 주행당 최대 count개를 균등하게 고른다.
        selected = np.linspace(0, len(starts) - 1, min(count, len(starts))).astype(int)
        starts = starts[selected]
    else:
        # 전체 평가: 인덱스 간격이 아니라 실제 경과 시간으로 시작점을 고른다.
        selected = []
        next_time = -np.inf
        for start in starts:
            if times[start] >= next_time:
                selected.append(int(start))
                next_time = times[start] + start_interval
        starts = selected
    return [(int(start), int(stops[start])) for start in starts]


def derivative_metrics(predicted, actual, std):
    """잡음이 포함된 정답을 평가하므로 observed=True의 표준편차를 받는다."""
    error = predicted - actual
    variance = np.maximum(std**2, 1e-12)
    rmse = np.sqrt(np.mean(error**2, axis=0))
    mae = np.mean(np.abs(error), axis=0)
    coverage = np.mean(np.abs(error) <= 1.96 * std, axis=0)
    nll = np.mean(0.5 * (np.log(2 * np.pi * variance) + error**2 / variance), axis=0)
    return [{
        "target": name,
        "samples": len(actual),
        "rmse": float(rmse[j]),
        "mae": float(mae[j]),
        "coverage_95": float(coverage[j]),
        "mean_observed_std": float(np.mean(std[:, j])),
        "mean_nll": float(nll[j]),
    } for j, name in enumerate(TARGET_NAMES)]


def prediction_failure(state, limits):
    if not np.isfinite(state).all():
        return "nonfinite"
    if np.hypot(state[4], state[5]) > limits["speed_mps"]:
        return "speed_limit"
    if abs(state[6]) > limits["yaw_rate_radps"]:
        return "yaw_rate_limit"
    if abs(state[3]) > limits["roll_rad"]:
        return "roll_limit"
    return None


def rollout(model, states, controls, ds, times, limits=None):
    """첫 상태만 실제값으로 주고, 이후는 모델 예측을 계속 사용한다."""
    predicted = np.full_like(states, np.nan, dtype=float)
    predicted[0] = states[0]
    if limits is None:
        limits = DEFAULT_LIMITS

    reason = prediction_failure(predicted[0], limits)
    if reason:
        return predicted, reason

    for i in range(len(times) - 1):
        dt = float(times[i + 1] - times[i])
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            next_state = model.step(predicted[i], controls[i], ds[i], dt)
        predicted[i + 1] = next_state
        reason = prediction_failure(next_state, limits)
        if reason:
            return predicted, reason

    return predicted, None


def state_errors(predicted, actual):
    errors = predicted - actual
    # 같은 방향을 나타내는 2*pi 차이가 오차로 집계되지 않도록 한다.
    errors[:, 2:4] = np.arctan2(np.sin(errors[:, 2:4]), np.cos(errors[:, 2:4]))
    return errors


def evaluate_window(model, trajectory, start, stop, member, horizon, limits=None, include_trace=True):
    # stop 행도 평가 대상에 포함한다. valid 행을 압축해서 연결하지 않는다.
    times = trajectory.times[start:stop + 1]
    actual = trajectory.states[start:stop + 1]
    controls = trajectory.controls[start:stop + 1]
    ds = trajectory.ds[start:stop + 1]
    predicted, failure_reason = rollout(model, actual, controls, ds, times, limits)
    low_speed = bool(np.any(predicted[:, 4] <= 0.2))
    errors = state_errors(predicted, actual)

    record = {
        "member": member,
        "maneuver": next(p for p in member.split("/") if p.endswith("_30_hz")),
        "horizon_seconds": float(horizon),
        "actual_duration_seconds": float(times[-1] - times[0]),
        "start_index": start,
        "stop_index": stop,
        "start_time_seconds": float(times[0]),
        "steps": len(times) - 1,
        "failed": failure_reason is not None,
        "failure_reason": failure_reason,
        "diverged": failure_reason == "nonfinite",
        "out_of_bounds": failure_reason not in (None, "nonfinite"),
        "predicted_low_speed": low_speed,
        "endpoint_position_error_m": None,
        "trajectory_position_rmse_m": None,
    }
    # 초기 상태는 반드시 일치하므로 RMSE 계산에서 제외한다.
    for j, name in enumerate(STATE_NAMES):
        record[f"endpoint_{name}_error"] = None
        record[f"trajectory_{name}_rmse"] = None
    if failure_reason is None:
        position_error = np.linalg.norm(errors[1:, :2], axis=1)
        record["endpoint_position_error_m"] = float(position_error[-1])
        record["trajectory_position_rmse_m"] = float(np.sqrt(np.mean(position_error**2)))
        for j, name in enumerate(STATE_NAMES):
            record[f"endpoint_{name}_error"] = float(errors[-1, j])
            record[f"trajectory_{name}_rmse"] = float(np.sqrt(np.mean(errors[1:, j]**2)))

    if not include_trace:
        return record, None
    trace = pd.DataFrame({"elapsed_seconds": times - times[0]})
    for j, name in enumerate(STATE_NAMES):
        trace[f"actual_{name}"] = actual[:, j]
        trace[f"predicted_{name}"] = predicted[:, j]
        trace[f"error_{name}"] = errors[:, j]
    trace["steering"] = controls[:, 0]
    trace["throttle"] = controls[:, 1]
    trace["ds"] = ds
    return record, trace


def summarize(records):
    """성공 구간의 오차와 전체 시도의 실패율을 반드시 함께 보고한다."""
    summaries = []
    for horizon in sorted({row["horizon_seconds"] for row in records}):
        group = [r for r in records if r["horizon_seconds"] == horizon]
        successful = [r for r in group if not r["failed"]]
        summary = {
            "horizon_seconds": horizon,
            "windows": len(group),
            "successful_windows": len(successful),
            "failed_windows": sum(r["failed"] for r in group),
            "failure_rate": sum(r["failed"] for r in group) / len(group),
            "diverged_windows": sum(r["diverged"] for r in group),
            "out_of_bounds_windows": sum(r["out_of_bounds"] for r in group),
            "failure_reasons": {
                reason: sum(r["failure_reason"] == reason for r in group)
                for reason in ("nonfinite", "speed_limit", "yaw_rate_limit", "roll_limit")
            },
            "predicted_low_speed_windows": sum(r["predicted_low_speed"] for r in group),
            "endpoint_position_rmse_m": None,
            "endpoint_velocity_rmse_mps": None,
        }
        for name in STATE_NAMES[2:]:
            summary[f"endpoint_{name}_rmse"] = None
        if successful:
            distance_errors = np.array([r["endpoint_position_error_m"] for r in successful])
            summary["endpoint_position_rmse_m"] = float(np.sqrt(np.mean(distance_errors**2)))
            velocity_error_squared = [
                r["endpoint_vx_error"]**2 + r["endpoint_vy_error"]**2 for r in successful
            ]
            summary["endpoint_velocity_rmse_mps"] = float(np.sqrt(np.mean(velocity_error_squared)))
            for name in STATE_NAMES[2:]:
                values = np.array([r[f"endpoint_{name}_error"] for r in successful])
                summary[f"endpoint_{name}_rmse"] = float(np.sqrt(np.mean(values**2)))
        summaries.append(summary)
    return summaries


def save_example_plot(trace, record, path):
    # 파일로만 저장한다. 실행 중 그래프 창을 띄우지 않는다.
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    ax = axes[0, 0]
    ax.plot(trace.actual_px, trace.actual_py, label="Actual")
    ax.plot(trace.predicted_px, trace.predicted_py, "--", label="Predicted")
    ax.scatter(trace.actual_px.iloc[0], trace.actual_py.iloc[0], color="black", s=20)
    ax.set(xlabel="X [m]", ylabel="Y [m]", title="XY trajectory")
    ax.set_aspect("equal", adjustable="datalim")

    panels = (("yaw", "Yaw [rad]"), ("vx", "vx [m/s]"),
              ("vy", "vy [m/s]"), ("yaw_rate", "Yaw rate [rad/s]"),
              ("roll", "Roll [rad]"))
    for ax, (name, label) in zip(axes.flat[1:], panels):
        actual = trace[f"actual_{name}"]
        predicted = trace[f"predicted_{name}"]
        if name in ("yaw", "roll"):
            predicted = actual + trace[f"error_{name}"]
        ax.plot(trace.elapsed_seconds, actual, label="Actual")
        ax.plot(trace.elapsed_seconds, predicted, "--", label="Predicted")
        ax.set(xlabel="Elapsed time [s]", ylabel=label)

    for ax in axes.flat:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(
        f"{record['maneuver']} | {record['actual_duration_seconds']:.2f} s\n"
        f"Endpoint position error: {record['endpoint_position_error_m']:.3f} m\n"
        f"{record['member']} | start: {record['start_time_seconds']:.3f} s",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def clear_previous_results(output_dir):
    """Remove only files generated by this evaluator; preserve other contents."""
    output_dir = Path(output_dir).resolve()
    names = {"summary.json", "window_metrics.csv", "derivative_metrics.csv"}
    for maneuver in ("straight", "skidpad", "fishhook", "slalom"):
        for suffix in ("example.csv", "example.png",
                       "failure_example.csv", "failure_example.json"):
            names.add(f"{maneuver}_30_hz_{suffix}")

    removed = 0
    for name in sorted(names):
        path = output_dir / name
        if path.is_symlink() or path.is_file():
            path.unlink()
            removed += 1
    print(f"Cleared {removed} previous evaluation files from {output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT_DIR / "data" /
                        "AutoDRIVE-Hunter-SE-Dataset-on-road-dataset.zip")
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent / "trained_model")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "trajectory_results")
    parser.add_argument("--horizons", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 2.5])
    parser.add_argument("--start-interval", type=float, default=0.5,
                        help="평가 시작점 사이의 최소 시간 간격 [s]")
    parser.add_argument("--windows-per-run", type=int, default=0,
                        help="0=시간 간격으로 전체 평가, 양수=주행당 최대 구간 수")
    parser.add_argument("--max-speed", type=float, default=DEFAULT_LIMITS["speed_mps"])
    parser.add_argument("--max-yaw-rate", type=float, default=DEFAULT_LIMITS["yaw_rate_radps"])
    parser.add_argument("--max-roll", type=float, default=DEFAULT_LIMITS["roll_rad"])
    parser.add_argument("--no-plots", action="store_true")
   
    args = parser.parse_args()
    positive_values = args.horizons + [args.start_interval, args.max_speed, args.max_yaw_rate, args.max_roll]
    if args.windows_per_run < 0 or any(not np.isfinite(v) or v <= 0 for v in positive_values):
        parser.error("time intervals and prediction limits must be positive; windows-per-run must be >= 0")
    limits = {"speed_mps": args.max_speed, "yaw_rate_radps": args.max_yaw_rate, "roll_rad": args.max_roll}

    model = BayesianDynamics.load(args.model_dir / "dynamics_model.npz")
    manifest = json.loads((args.model_dir / "split.json").read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    examples = {}
    excluded = []
    derivative_batches = []
    failure_examples = {}

    with zipfile.ZipFile(args.data) as archive:
        clear_previous_results(args.out)
        test_members = manifest["splits"]["test"]
        for run_index, member in enumerate(test_members, 1):
            trajectory = load_trajectory(archive, member)
            ids = np.flatnonzero(trajectory.valid)
            maneuver = next(p for p in member.split("/") if p.endswith("_30_hz"))
            if len(ids):
                means, stds = model.predict(trajectory.states[ids], trajectory.controls[ids], trajectory.ds[ids], observed=True)
                derivative_batches.append((maneuver, means, trajectory.targets[ids], stds))
            for horizon in sorted(set(args.horizons)):
                windows = choose_windows(trajectory.times, trajectory.valid,
                                         horizon, args.windows_per_run, args.start_interval)
                if not windows:
                    excluded.append({"member": member, "horizon_seconds": horizon})
                for start, stop in windows:
                    record, trace = evaluate_window(
                        model, trajectory, start, stop, member, horizon, limits,
                        include_trace=horizon == max(args.horizons),
                    )
                    records.append(record)
                    if horizon == max(args.horizons):
                        if not record["failed"]:
                            examples.setdefault(maneuver, []).append((record, trace))
                        elif maneuver not in failure_examples:
                            failure_examples[maneuver] = (record, trace)
            print(f"Evaluated run {run_index}/{len(test_members)}: {member}", flush=True)

    if not records:
        raise ValueError("No continuous valid test windows found")
    pd.DataFrame(records).to_csv(args.out / "window_metrics.csv", index=False)
    derivative_results = {}
    derivative_rows = []
    for scope in ["overall"] + sorted({batch[0] for batch in derivative_batches}):
        batches = [batch for batch in derivative_batches if scope == "overall" or batch[0] == scope]
        predicted = np.concatenate([batch[1] for batch in batches])
        actual = np.concatenate([batch[2] for batch in batches])
        std = np.concatenate([batch[3] for batch in batches])
        metrics = derivative_metrics(predicted, actual, std)
        derivative_results[scope] = metrics
        derivative_rows.extend({"scope": scope, **row} for row in metrics)
    pd.DataFrame(derivative_rows).to_csv(args.out / "derivative_metrics.csv", index=False)
    summary = {
        "model_directory": str(args.model_dir.resolve()),
        "test_runs": len(manifest["splits"]["test"]),
        "windows_per_run_limit": args.windows_per_run,
        "start_interval_seconds": args.start_interval,
        "prediction_limits": limits,
        "overlapping_windows": True,
        "metrics_note": "Rollout RMSE excludes failed windows; always interpret with failure rate. "
                        "Overlapping windows are correlated. Coverage refers to preprocessed observed derivatives.",
        "evaluation": "Open-loop model prediction with recorded controls; actual initial state only.",
        "reference": "States preprocessed by load_trajectory; windows must remain valid throughout.",
        "excluded_run_horizons": excluded,
        "derivative_metrics": derivative_results,
        "results": summarize(records),
        "results_by_maneuver": {
            maneuver: summarize([r for r in records if r["maneuver"] == maneuver])
            for maneuver in sorted({r["maneuver"] for r in records})
        },
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )

    print("\nDerivative test (observed predictive intervals):")
    for row in derivative_results["overall"]:
        print(f"{row['target']:6s} RMSE={row['rmse']:.6f} MAE={row['mae']:.6f} "
              f"95% coverage={row['coverage_95']:.1%} NLL={row['mean_nll']:.5f}")
    print("\nOpen-loop trajectory test:")
    for row in summary["results"]:
        print(f"Horizon {row['horizon_seconds']:g} s | windows={row['windows']} "
              f"| failed={row['failed_windows']} ({row['failure_rate']:.1%}) "
              f"| nonfinite={row['diverged_windows']} bounds={row['out_of_bounds_windows']} "
              f"| low-speed={row['predicted_low_speed_windows']} "
              f"| endpoint position RMSE={row['endpoint_position_rmse_m']} m")

    # 최장 예측 길이에서 주행 종류별 끝점 오차가 중앙 순위인 예제를 저장한다.
    for maneuver, candidates in sorted(examples.items()):
        candidates.sort(key=lambda item: item[0]["endpoint_position_error_m"])
        record, trace = candidates[len(candidates) // 2]
        trace.to_csv(args.out / f"{maneuver}_example.csv", index=False)
        if not args.no_plots:
            save_example_plot(trace, record, args.out / f"{maneuver}_example.png")
    for maneuver, (record, trace) in sorted(failure_examples.items()):
        trace.to_csv(args.out / f"{maneuver}_failure_example.csv", index=False)
        (args.out / f"{maneuver}_failure_example.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
    print(f"Saved evaluation to {args.out.resolve()}")


if __name__ == "__main__":
    main()

"""원형 트랙 MPPI 실행 및 결과 저장. 프로젝트 루트에서 python -m mppi.run.

예측과 환경이 같은 학습 모델을 사용한다. 실제 차량 성능 검증 실험은 아니다.
각 실행은 results 아래 별도 폴더에 저장한다.
"""

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import Config, STATE_NAMES, CONTROL_NAMES
from .controller import MPPIController, NoValidTrajectoryError
from .dynamic_adaptor import DynamicsAdaptor
from .simulator import CircularTrack, Simulator


def save_plot(path: Path, data: dict, config: Config, track: CircularTrack,
              reason: str) -> None:
    """실행 궤적과 상태, 입력, controller 진단값을 한 그림에 저장한다."""
    states, times = data["states"], data["times"]
    action_times = times[:-1]
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    ax = axes[0, 0]
    angle = np.linspace(0, 2 * np.pi, 361)
    for radius, style, label in (
        (track.radius, "k--", "Centerline"),
        (track.radius - track.half_width, "k-", "Boundary"),
        (track.radius + track.half_width, "k-", None),
    ):
        ax.plot(track.center[0] + radius * np.cos(angle),
                track.center[1] + radius * np.sin(angle), style,
                linewidth=1, label=label)
    ax.plot(states[:, 0], states[:, 1], label="Executed trajectory")
    ax.scatter(states[0, 0], states[0, 1], color="green", label="Start")
    if np.isfinite(states[-1, :2]).all():
        ax.scatter(states[-1, 0], states[-1, 1], color="red", label="End")
    ax.set(xlabel="x (m)", ylabel="y (m)", title="Closed-loop trajectory")
    ax.set_aspect("equal", adjustable="box")

    ax = axes[0, 1]
    ax.plot(times, data["track_errors"], label="Normalized track error")
    ax.axhline(1, color="red", linestyle="--", label="Boundary")
    ax.axhline(-1, color="red", linestyle="--")
    ax.set(ylabel="h", xlabel="Time (s)")

    ax = axes[1, 0]
    ax.plot(times, np.hypot(states[:, 4], states[:, 5]), label="Speed")
    ax.axhline(config.target_speed, color="black", linestyle="--", label="Target")
    ax.set(ylabel="Speed (m/s)", xlabel="Time (s)")

    ax = axes[1, 1]
    ax.plot(times, states[:, 6], label="Yaw rate (rad/s)")
    ax.plot(times, states[:, 3], label="Roll (rad)")
    ax.set(xlabel="Time (s)")

    ax = axes[2, 0]
    ax.step(action_times, data["controls"][:, 0], where="post", label="Steering (rad)")
    ax.step(action_times, data["controls"][:, 1], where="post", label="Throttle")
    ax.plot(times, states[:, 7], linestyle="--", label="Applied steering (rad)")
    ax.plot(times, states[:, 8], linestyle="--", label="Applied throttle")
    ax.set(xlabel="Input application time (s)")

    ax = axes[2, 1]
    ax.plot(action_times, data["valid_fractions"], label="Valid candidate fraction")
    ax.scatter(action_times[data["fallbacks"]],
               data["valid_fractions"][data["fallbacks"]],
               marker="x", color="red", label="Sample fallback")
    ax.set(xlabel="Input application time (s)", ylim=(-0.05, 1.05))
    for ax in axes.flat:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Learned-model simulation | termination: {reason}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_diagnostics(output: Path, records: list[dict]) -> None:
    """시도한 제어의 반복별 요약 및 후보별 원본 값을 저장한다. 실패 시도도 포함한다."""
    if not records:
        return
    rows = [record["summary"] for record in records]
    with (output / "diagnostics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    arrays = {name: np.stack([record["candidates"][name] for record in records])
              for name in records[0]["candidates"]}
    for name in ("step", "application_time", "iteration", "input_applied"):
        arrays[name] = np.asarray([row[name] for row in rows])
    np.savez_compressed(output / "candidate_diagnostics.npz", **arrays)

    t = [row["application_time"] for row in rows]
    def column(name):
        return np.asarray([np.nan if row[name] is None else row[name] for row in rows])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for name in ("track", "speed", "control", "slip"):
        axes[0, 0].plot(t, column(f"{name}_cost_mean"), label=name)
    axes[0, 0].set_title("Mean component costs over valid candidates")
    for name in ("trajectory", "importance", "total"):
        axes[0, 1].plot(t, column(f"{name}_cost_mean"), label=name)
    for name in ("mean_shift", "variance_change"):
        if f"{name}_cost_mean" in rows[0]:
            axes[0, 1].plot(t, column(f"{name}_cost_mean"), linestyle=":", label=name)
    axes[0, 1].set_title("Mean cost incl. signed importance correction")
    axes[1, 0].plot(t, column("effective_sample_size"), label="ESS")
    axes[1, 0].plot(t, column("valid_count"), label="Valid candidate count")
    axes[1, 0].set_title("Weight concentration (log scale)")
    axes[1, 0].set_yscale("symlog", linthresh=1)
    for name in ("off_track", "low_speed", "nonfinite", "speed_limit",
                 "yaw_rate_limit", "roll_limit"):
        axes[1, 1].plot(t, column(f"reject_{name}_count"), label=name)
    axes[1, 1].set_title("Rejection counts (overlapping)")
    for ax in axes.flat:
        ax.set_xlabel("Input application time (s)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "diagnostics.png", dpi=160)
    plt.close(fig)


def run_simulation(config: Config, track: CircularTrack,
                   log_every: int = 30) -> tuple[Path, dict]:
    """주행을 실행한다. 무효 후보 또는 사용자 중단 시에도 완료된 스텝은 저장한다."""
    dynamics = DynamicsAdaptor(config)
    sim = Simulator(config, dynamics, track.track_error,
                    track.initial_state(config.target_speed))
    controller = MPPIController(config, dynamics, track.track_error)
    output = Path(config.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"config": asdict(config), "track": asdict(track),
                "state_names": STATE_NAMES, "control_semantics": "desired actuator targets",
                "derived": {"R_diagonal": config.control_cost_diag,
                            "base_command_variance": config.base_command_variance,
                            "exploration_variance_ratio": config.exploration_variance_ratio,
                            "base_actuator_diffusion_std": config.base_actuator_diffusion_std},
                "environment": "same learned mean dynamics as controller"}
    (output / "config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    states, times, errors = [sim.state], [sim.time], [sim.current_track_error]
    controls, valid_fractions, fallbacks, command_seconds = [], [], [], []
    applied_controls = []
    predictions = []
    diagnostic_records = []

    def record_diagnostics(step: int, time: float, applied: bool) -> None:
        for record in controller.last_iteration_diagnostics:
            # 기존 controller 통계는 복사하여 실행 시점과 입력 적용 여부를 추가한다.
            diagnostic_records.append({
                "summary": {"step": step, "application_time": time,
                            "input_applied": applied, **record["summary"]},
                "candidates": record["candidates"],
            })

    reason, message = "time_limit", None
    failed_command_seconds = None
    start = perf_counter()
    print(f"Output: {output}", flush=True)
    print(f"K={config.num_samples}, N={config.horizon_steps}, "
          f"horizon={config.horizon_seconds:.2f}s, dt={config.dt:.4f}s", flush=True)
    try:
        while not sim.done:
            command_start = perf_counter()
            try:
                control = controller.command(sim.state)
            except NoValidTrajectoryError as exc:
                failed_command_seconds = perf_counter() - command_start
                reason, message = "no_valid_trajectory", str(exc)
                record_diagnostics(sim.steps, sim.time, False)
                break
            elapsed = perf_counter() - command_start
            result = sim.step(control)
            record_diagnostics(sim.steps - 1, sim.time - config.dt, True)
            states.append(result.state)
            times.append(result.time)
            errors.append(result.track_error)
            controls.append(result.control)
            applied_controls.append(result.applied_control)
            valid_fractions.append(controller.last_valid_fraction)
            fallbacks.append(controller.last_used_sample_fallback)
            command_seconds.append(elapsed)
            predictions.append(controller.last_trajectory.copy())
            if sim.steps == 1 or sim.steps % log_every == 0 or sim.done:
                print(f"step={sim.steps}/{config.max_steps} t={sim.time:.2f}s "
                      f"h={result.track_error:.3f} "
                      f"speed={np.hypot(result.state[4], result.state[5]):.3f} "
                      f"valid={controller.last_valid_fraction:.1%} "
                      f"ESS={controller.last_iteration_diagnostics[-1]['summary']['effective_sample_size']:.1f} "
                      f"max_w={controller.last_iteration_diagnostics[-1]['summary']['max_weight']:.3f} "
                      f"command={elapsed:.3f}s", flush=True)
        else:
            reason = sim.reason
    except KeyboardInterrupt:
        reason, message = "interrupted", "Stopped by user"

    loop_wall_seconds = perf_counter() - start
    # S개 입력에 대해 상태/시간은 S+1개. 제어 t는 상태 t -> t+1에 적용된다.
    data = {
        "states": np.asarray(states), "times": np.asarray(times),
        "track_errors": np.asarray(errors),
        "controls": np.asarray(controls).reshape(-1, 2),
        "applied_controls": np.asarray(applied_controls).reshape(-1, 2),
        "valid_fractions": np.asarray(valid_fractions),
        "fallbacks": np.asarray(fallbacks, dtype=bool),
        "command_seconds": np.asarray(command_seconds),
        "predictions": np.asarray(predictions).reshape(-1, config.horizon_steps + 1, dynamics.state_dim),
    }
    np.savez_compressed(output / "trajectory.npz", **data)
    save_diagnostics(output, diagnostic_records)
    with (output / "states.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time", *STATE_NAMES, "track_error"])
        writer.writerows([t, *state, h] for t, state, h in zip(times, states, errors))
    with (output / "controls.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["application_time", *[f"desired_{name}" for name in CONTROL_NAMES],
                         *[f"applied_{name}" for name in CONTROL_NAMES], "valid_fraction",
                         "used_fallback", "command_seconds"])
        writer.writerows([t, *u, *a, v, f, c] for t, u, a, v, f, c in zip(
            times[:-1], controls, applied_controls, valid_fractions, fallbacks, command_seconds))

    # 발산을 평균에서 숨기지 않도록 비유한 값이 있으면 해당 지표는 null로 기록한다.
    def rmse(values: np.ndarray) -> float | None:
        if not np.isfinite(values).all():
            return None
        with np.errstate(over="ignore", invalid="ignore"):
            value = float(np.sqrt(np.mean(values**2)))
        return value if np.isfinite(value) else None

    speed = np.hypot(data["states"][:, 4], data["states"][:, 5])
    summary = {
        "termination_reason": reason, "message": message,
        "completed_requested_steps": len(controls) == config.max_steps and reason == "time_limit",
        "steps": len(controls), "simulation_seconds": times[-1],
        "loop_wall_seconds": loop_wall_seconds,
        "track_rmse_m": rmse(data["track_errors"] * track.half_width),
        "speed_rmse_mps": rmse(speed - config.target_speed),
        "nonfinite_state_rows": int(np.sum(~np.isfinite(data["states"]).all(axis=1))),
        "fallback_count": int(np.sum(data["fallbacks"])),
        "mean_valid_fraction": float(np.mean(valid_fractions)) if controls else None,
        "mean_command_seconds": float(np.mean(command_seconds)) if controls else None,
        "max_command_seconds": max(command_seconds) if controls else None,
        "failed_command_seconds": failed_command_seconds,
        "metric_scope": "Recorded states including initial and terminal states",
        "diagnostic_iterations": len(diagnostic_records),
        "last_diagnostics": diagnostic_records[-1]["summary"] if diagnostic_records else None,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    save_plot(output / "overview.png", data, config, track, reason)
    print(f"Finished: {reason}, steps={len(controls)}, simulation={times[-1]:.2f}s", flush=True)
    return output, summary


def main() -> int:
    defaults = Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=defaults.max_steps)
    parser.add_argument("--samples", type=int, default=defaults.num_samples)
    parser.add_argument("--horizon", type=int, default=defaults.horizon_steps)
    parser.add_argument("--target-speed", type=float, default=defaults.target_speed)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--half-width", type=float, default=0.75)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--model-path", type=Path, default=defaults.model_path)
    parser.add_argument("--log-every", type=int, default=30)
    parser.add_argument("--weight-temperature-mode", choices=("nominal_cost", "fixed", "theoretical"),
                        default=defaults.weight_temperature_mode)
    parser.add_argument("--temperature", type=float, default=defaults.temperature,
                        help="Weight temperature when mode is fixed")
    parser.add_argument("--cost-normalization-alpha", type=float, default=defaults.cost_normalization_alpha)
    parser.add_argument("--actuator-rate", type=float, default=defaults.actuator_rate)
    parser.add_argument("--disable-actuator", action="store_true", help="Actuator ablation only")
    args = parser.parse_args()
    if args.log_every < 1:
        parser.error("--log-every must be positive")
    config = replace(defaults, max_steps=args.steps, num_samples=args.samples,
                     horizon_steps=args.horizon, target_speed=args.target_speed,
                     seed=args.seed, output_dir=args.output_dir, model_path=args.model_path,
                     weight_temperature_mode=args.weight_temperature_mode,
                     temperature=args.temperature,
                     cost_normalization_alpha=args.cost_normalization_alpha,
                     actuator_rate=args.actuator_rate, actuator_enabled=not args.disable_actuator)
    _, summary = run_simulation(config, CircularTrack(args.radius, args.half_width), args.log_every)
    if summary["termination_reason"] == "interrupted":
        return 130
    return 0 if summary["completed_requested_steps"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

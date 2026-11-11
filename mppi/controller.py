"""
단일 차량 MPPI 제어기.

track_error(px, py)는 배열 입력을 받고 같은 크기의 정규화 오차 h를 반환한다.
h=0은 중심선, abs(h)=1은 트랙 경계다. 비용은 Config의 논문 참조 가중치를 쓴다.
이 구현은 트랙 이탈 후보를 무효 처리한다. 논문의 이탈 후 상태 고정과는 다르다.
기본 command 공분산 C0=lambda*(R*dt)^-1과 탐색 공분산 Ce를 구분한다.
보정은 unclipped latent command의 Gaussian 밀도 비율로 계산하고, 포화는 동역학에 둔다.
구동기는 상태의 마지막 두 성분이다. passive dynamics 불확실성 샘플링은 포함하지 않는다.
"""

from collections.abc import Callable

import numpy as np

from .config import Config
from .dynamic_adaptor import DynamicsAdaptor


class NoValidTrajectoryError(RuntimeError):
    """유효한 후보가 없음. 실제 차량의 정지/복구 동작은 run/simulator에서 결정한다."""


class MPPIController:
    def __init__(self, config: Config, dynamics: DynamicsAdaptor,
                 track_error: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        if config != dynamics.config:
            raise ValueError("Controller and dynamics must use the same Config")
        self.config = config
        self.dynamics = dynamics
        self.track_error = track_error
        self.rng = np.random.default_rng(config.seed)
        self.nominal_controls = np.empty((config.horizon_steps, 2), dtype=float)
        self.reset()

    def reset(self, control_sequence: np.ndarray | None = None) -> None:
        """첫 실행/시뮬레이션 재시작 때 nominal 시퀀스를 초기화한다."""
        cfg = self.config
        if control_sequence is None:
            self.nominal_controls[:] = (cfg.nominal_steering, cfg.nominal_throttle)
        else:
            sequence = self.dynamics.clip_controls(control_sequence)
            if sequence.shape != (cfg.horizon_steps, 2):
                raise ValueError("control_sequence must have shape (N, 2)")
            self.nominal_controls[:] = sequence
        self.last_costs = None
        self.last_weights = None
        self.last_control_sequence = None
        self.last_trajectory = None
        self.last_valid_fraction = 0.0
        self.last_used_sample_fallback = False
        self.last_iteration_diagnostics = []
        self.last_effective_temperature = None

    def sample_controls(self) -> tuple[np.ndarray, np.ndarray]:
        """Gaussian latent command와 원래 잡음을 반환한다. 포화는 동역학에서 처리한다."""
        cfg = self.config
        noise = self.rng.normal(size=(cfg.num_samples, cfg.horizon_steps, 2)) #(K, N, 2)
        noise *= np.asarray(cfg.control_noise_std)
        candidates = self.nominal_controls[None] + noise
        return candidates, noise

    def trajectory_costs(self, trajectories: np.ndarray,
                         controls: np.ndarray,
                         diagnostics: dict | None = None) -> np.ndarray:
        """(K,N+1,9) 궤적의 누적 상태 비용. 제어 에너지는 optimize에서 별도 계산한다."""
        cfg = self.config
        trajectories = np.asarray(trajectories, dtype=float)
        controls = np.asarray(controls, dtype=float)
        if controls.ndim != 3 or controls.shape[1:] != (cfg.horizon_steps, 2):
            raise ValueError("controls must have shape (K, N, 2)")
        if trajectories.shape != (len(controls), cfg.horizon_steps + 1, self.dynamics.state_dim):
            raise ValueError("trajectories must have shape (K, N+1, 9)")

        # 발산한 후보를 track callback에 전달하지 않는다.
        costs = np.full(len(controls), np.inf)
        with np.errstate(over="ignore", invalid="ignore"):
            valid = (np.isfinite(trajectories).all(axis=(1, 2))
                     & np.isfinite(controls).all(axis=(1, 2))
                     & (trajectories[..., 4] > cfg.min_prediction_vx).all(axis=1)
                     & (np.hypot(trajectories[..., 4], trajectories[..., 5])
                        <= cfg.max_prediction_speed).all(axis=1)
                     & (abs(trajectories[..., 6]) <= cfg.max_prediction_yaw_rate).all(axis=1)
                     & (abs(trajectories[..., 3]) <= cfg.max_prediction_roll).all(axis=1))
        if diagnostics is not None:
            # 종료 원인은 중복 집계한다. 예: 한 후보가 저속과 트랙 이탈을 모두 만족.
            finite = (np.isfinite(trajectories).all(axis=(1, 2))
                      & np.isfinite(controls).all(axis=(1, 2)))
            with np.errstate(over="ignore", invalid="ignore"):
                diagnostics.update({
                    "reject_nonfinite": ~finite,
                    "reject_low_speed": (trajectories[..., 4] <= cfg.min_prediction_vx).any(axis=1),
                    "reject_speed_limit": (np.hypot(trajectories[..., 4], trajectories[..., 5])
                                           > cfg.max_prediction_speed).any(axis=1),
                    "reject_yaw_rate_limit": (abs(trajectories[..., 6])
                                              > cfg.max_prediction_yaw_rate).any(axis=1),
                    "reject_roll_limit": (abs(trajectories[..., 3])
                                          > cfg.max_prediction_roll).any(axis=1),
                })
            h_all = np.full(trajectories.shape[:2], np.nan)
            if finite.any():
                finite_states = trajectories[finite]
                finite_h = np.asarray(self.track_error(finite_states[..., 0],
                                                       finite_states[..., 1]), dtype=float)
                if finite_h.shape != finite_states.shape[:2]:
                    raise ValueError("track_error must return the same shape as its position arrays")
                h_all[finite] = finite_h
            diagnostics["reject_off_track"] = (abs(h_all) > 1.0).any(axis=1)
            diagnostics["reject_track_nonfinite"] = finite & ~np.isfinite(h_all).all(axis=1)
            diagnostics["reject_nonfinite_cost"] = np.zeros(len(controls), dtype=bool)
            for name in ("track", "speed", "control", "slip"):
                diagnostics[f"{name}_cost"] = np.full(len(controls), np.nan)
        ids = np.flatnonzero(valid)
        if not len(ids):
            return costs
        states = trajectories[ids]
        h = (h_all[ids] if diagnostics is not None else
             np.asarray(self.track_error(states[..., 0], states[..., 1]), dtype=float))
        if h.shape != states.shape[:2]:
            raise ValueError("track_error must return the same shape as its position arrays")
        on_track = np.isfinite(h).all(axis=1) & (abs(h) <= 1.0).all(axis=1)

        # 각 입력 적용 후 상태의 running cost. 초기 상태는 유효성만 검사한다.
        future = states[:, 1:]
        with np.errstate(over="ignore", invalid="ignore"):
            speed = np.hypot(future[..., 4], future[..., 5])
            slip = np.arctan2(future[..., 5], future[..., 4])
            running = (cfg.w_track * h[:, 1:]**2
                       + cfg.w_speed * (speed - cfg.target_speed)**2
                       + cfg.w_slip * (abs(slip) > cfg.slip_angle_threshold))
            total = cfg.dt * np.sum(running, axis=1)
            if diagnostics is not None:
                # 실제 running cost와 같은 가중치 및 dt를 적용한 누적 항별 비용.
                terms = {
                    "track": cfg.w_track * h[:, 1:]**2,
                    "speed": cfg.w_speed * (speed - cfg.target_speed)**2,
                    "control": np.zeros_like(speed),
                    "slip": cfg.w_slip * (abs(slip) > cfg.slip_angle_threshold),
                }
                for name, values in terms.items():
                    diagnostics[f"{name}_cost"][ids] = cfg.dt * np.sum(values, axis=1)
                diagnostics["reject_nonfinite_cost"][ids] = ~np.isfinite(total)
        accepted = on_track & np.isfinite(total)
        costs[ids[accepted]] = total[accepted]
        return costs

    def importance_terms(self, noise: np.ndarray) -> dict[str, np.ndarray]:
        """-lambda*log(p/q)의 표본별 항. 로그 행렬식은 후보 공통 상수라 생략한다.

        p=N(0,C0), q=N(U,Ce), v=U+epsilon, Ce=diag(std**2).
        C0=lambda/(R*dt), nu=Ce/C0.
        전체 비율 비용 = .5*dt*sum(R*U**2) + mean_shift + variance_change.
        첫 항은 nominal 제어 에너지로 optimize에서 한 번만 더한다.
        """
        cfg = self.config
        r = np.asarray(cfg.control_cost_diag)
        nu = np.asarray(cfg.exploration_variance_ratio)
        return {
            "mean_shift_cost": cfg.dt * np.sum(r * self.nominal_controls[None] * noise,
                                                axis=(1, 2)),
            "variance_change_cost": 0.5 * cfg.dt * np.sum(
                r * (1.0 - 1.0 / nu) * noise**2, axis=(1, 2)),
        }

    def importance_costs(self, noise: np.ndarray) -> np.ndarray:
        terms = self.importance_terms(noise)
        return terms["mean_shift_cost"] + terms["variance_change_cost"]

    def weight_temperature(self, nominal_cost: float, candidate_costs: np.ndarray) -> float:
        """이론 lambda와 논문 실험의 비용 정규화 온도를 구분한다."""
        cfg = self.config
        if cfg.weight_temperature_mode == "theoretical":
            return cfg.path_integral_lambda
        if cfg.weight_temperature_mode == "fixed":
            return cfg.temperature
        if not np.isfinite(nominal_cost):
            finite = candidate_costs[np.isfinite(candidate_costs)]
            nominal_cost = float(finite.min()) if len(finite) else cfg.temperature
        return max(float(nominal_cost), 1.0e-8) / cfg.cost_normalization_alpha

    def compute_weights(self, costs: np.ndarray, temperature: float | None = None) -> np.ndarray:
        """유한한 비용만 사용하고, 최솟값을 뺀 softmax로 가중치를 구한다."""
        valid = np.isfinite(costs)
        if not valid.any():
            raise NoValidTrajectoryError("No valid candidate trajectory; no control was returned")
        weights = np.zeros_like(costs, dtype=float)
        shifted = costs[valid] - np.min(costs[valid])
        scale = self.config.temperature if temperature is None else temperature
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Weight temperature must be finite and positive")
        weights[valid] = np.exp(-shifted / scale)
        weights /= weights.sum()
        return weights

    def _record_diagnostics(self, iteration: int, values: dict,
                            costs: np.ndarray, weights: np.ndarray | None) -> dict:
        """후보별 값과, 유효 후보만 대상으로 계산한 통계를 보관한다."""
        valid = np.isfinite(costs)
        values["total_cost"] = costs.copy()
        values["valid"] = valid
        values["weights"] = (np.zeros_like(costs) if weights is None else weights.copy())
        count = int(valid.sum())
        summary = {"iteration": iteration, "num_candidates": len(costs),
                   "valid_count": count, "valid_fraction": count / len(costs),
                   "max_weight": float(weights.max()) if weights is not None else None,
                   "effective_sample_size": (float(1.0 / np.sum(weights**2))
                                             if weights is not None else None),
                   "best_candidate": int(np.argmin(costs)) if count else None,
                   "effective_temperature": self.last_effective_temperature,
                   "used_fallback": False}
        for name, value in values.items():
            if name.startswith("reject_"):
                summary[f"{name}_count"] = int(value.sum())
            elif name.endswith("_cost"):
                selected = value[valid]
                summary[f"{name}_mean"] = float(selected.mean()) if count else None
                summary[f"{name}_std"] = float(selected.std()) if count else None
                summary[f"{name}_min"] = float(selected.min()) if count else None
                summary[f"{name}_max"] = float(selected.max()) if count else None
                summary[f"{name}_best"] = float(value[summary["best_candidate"]]) if count else None
                summary[f"{name}_weighted_mean"] = (float(np.sum(selected * weights[valid]))
                                                      if weights is not None else None)
        self.last_iteration_diagnostics.append({"summary": summary, "candidates": values})
        return summary

    def optimize(self, state: np.ndarray) -> np.ndarray:
        """실제 구동기 상태를 포함한 관측 (9,)에서 N스텝 제어를 최적화한다."""
        state = np.asarray(state, dtype=float)
        if state.shape != (self.dynamics.state_dim,) or not np.isfinite(state).all():
            raise ValueError("state must be one finite 9-dimensional observation including actuators")
        self.last_iteration_diagnostics = []
        for iteration in range(self.config.num_iterations):
            candidates, noise = self.sample_controls()
            trajectories = self.dynamics.rollout(state, candidates)
            diagnostics = {}
            costs = self.trajectory_costs(trajectories, candidates, diagnostics)
            diagnostics["state_cost"] = costs.copy()
            control_energy = 0.5 * self.config.dt * np.sum(
                np.asarray(self.config.control_cost_diag) * self.nominal_controls**2)
            diagnostics["control_cost"] = np.full(len(costs), control_energy)
            costs += control_energy
            diagnostics["trajectory_cost"] = costs.copy()
            terms = self.importance_terms(noise)
            diagnostics.update(terms)
            importance = terms["mean_shift_cost"] + terms["variance_change_cost"]
            diagnostics["importance_cost"] = importance.copy()
            reference = self.dynamics.rollout(state, self.nominal_controls)
            nominal_cost = self.trajectory_costs(reference[None], self.nominal_controls[None])[0] + control_energy
            self.last_effective_temperature = self.weight_temperature(nominal_cost, costs)
            costs += importance
            diagnostics["reject_total_nonfinite"] = ~np.isfinite(costs)
            try:
                weights = self.compute_weights(costs, self.last_effective_temperature)
            except NoValidTrajectoryError:
                self._record_diagnostics(iteration, diagnostics, costs, None)
                self.last_costs = costs.copy()
                self.last_weights = np.zeros_like(costs)
                self.last_valid_fraction = 0.0
                raise
            diagnostic_summary = self._record_diagnostics(iteration, diagnostics, costs, weights)
            update = np.einsum("k,kni->ni", weights, noise)
            sequence = self.dynamics.clip_controls(self.nominal_controls + update)
            prediction = self.dynamics.rollout(state, sequence)
            # 유효 후보의 평균이 비선형 트랙 밖으로 나갈 수도 있으므로 재검사한다.
            used_fallback = not np.isfinite(self.trajectory_costs(
                prediction[None], sequence[None])[0])
            if used_fallback:
                best = int(np.argmin(costs))
                # latent 후보와 포화된 후보는 같은 물리 궤적을 만든다.
                sequence = self.dynamics.clip_controls(candidates[best])
                prediction = trajectories[best].copy()
            diagnostic_summary["used_fallback"] = used_fallback
            self.nominal_controls = sequence.copy()
            self.last_costs = costs.copy()
            self.last_weights = weights.copy()
            self.last_valid_fraction = float(np.mean(np.isfinite(costs)))
            self.last_used_sample_fallback = used_fallback
            self.last_control_sequence = sequence.copy()
            self.last_trajectory = prediction.copy()
        return self.nominal_controls.copy()

    def shift(self) -> None:
        """실행한 첫 입력을 제외하고 다음 최적화의 초기 시퀀스로 이동한다."""
        self.nominal_controls[:-1] = self.nominal_controls[1:].copy()
        self.nominal_controls[-1] = (self.config.nominal_steering, self.config.nominal_throttle)

    def command(self, state: np.ndarray) -> np.ndarray:
        """최적화 → 첫 [steering, throttle] 반환 → warm start 시퀀스 이동."""
        sequence = self.optimize(state)
        control = sequence[0].copy()
        self.shift()
        return control

"""25개 기저 동역학과 MPPI 사이의 연결부.

상태: 차량 상태 7개 + 실제 steering, throttle. 입력: 원하는 steering, throttle.
차량과 구동기를 같은 시점의 상태에서 함께 Euler 적분한다.
현재 구동기 상태로 차량을 진행시키고, 다음 구동기 상태는 k*(u_des-a)로 갱신한다.
"""

from pathlib import Path

import numpy as np

from basis25_nll.dynamics import BayesianDynamics
from .config import Config, CONTROL_NAMES, STATE_NAMES, VEHICLE_STATE_NAMES


class DynamicsAdaptor:
    state_dim = len(STATE_NAMES)
    vehicle_state_dim = len(VEHICLE_STATE_NAMES)
    control_dim = len(CONTROL_NAMES)

    def __init__(self, config: Config):
        self.config = config
        model_path = Path(config.model_path).resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"25-feature dynamics checkpoint not found: {model_path}")

        self.model = BayesianDynamics.load(model_path)

        if (self.model.weights.shape != (25, 4)
                or self.model.feature_mean.shape != (25,)
                or self.model.feature_scale.shape != (25,)):
            raise ValueError("DynamicsAdaptor requires a 25-feature checkpoint")
        if (not np.isfinite(self.model.weights).all()
                or not np.isfinite(self.model.feature_mean).all()
                or not np.isfinite(self.model.feature_scale).all()
                or np.any(self.model.feature_scale <= 0)):
            raise ValueError("Checkpoint contains invalid weights or feature scaling")

    def clip_controls(self, controls: np.ndarray) -> np.ndarray:
        """(..., 2) 입력을 학습 데이터의 명령 범위 안으로 제한한다."""
        controls = np.asarray(controls, dtype=float)
        if controls.ndim < 1 or controls.shape[-1] != self.control_dim:
            raise ValueError("controls must have shape (..., 2): [steering, throttle]")
        if not np.isfinite(controls).all():
            raise ValueError("controls must be finite")
        return np.clip(controls, self.config.control_min, self.config.control_max)

    def _states(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        if states.ndim < 1 or states.shape[-1] != self.state_dim:
            raise ValueError("states must have shape (..., 9), including actual actuator states")
        if not np.isfinite(states).all():
            raise ValueError("input states must be finite")
        return states

    def augment_state(self, vehicle_state: np.ndarray,
                      applied_control: np.ndarray | None = None) -> np.ndarray:
        """차량 상태 (...,7)에 실제 구동기 상태를 붙인다. 현재 입력을 알면 반드시 전달한다."""
        state = np.asarray(vehicle_state, dtype=float)
        if state.ndim < 1 or state.shape[-1] != self.vehicle_state_dim or not np.isfinite(state).all():
            raise ValueError("vehicle_state must have shape (..., 7) and be finite")
        if applied_control is None:
            applied_control = np.array([self.config.initial_applied_steering,
                                        self.config.initial_applied_throttle])
        applied_control = self.clip_controls(applied_control)
        batch = np.broadcast_shapes(state.shape[:-1], applied_control.shape[:-1])
        return np.concatenate((np.broadcast_to(state, batch + (7,)),
                               np.broadcast_to(applied_control, batch + (2,))), axis=-1)

    def _advance(self, states: np.ndarray, desired_controls: np.ndarray) -> np.ndarray:
        """검증/방송이 끝난 배열을 한 스텝 갱신한다. rollout 중 NaN/Inf는 보존한다."""
        cfg = self.config
        if cfg.actuator_enabled:
            applied = states[..., 7:9]
            next_applied = applied + cfg.actuator_rate * cfg.dt * (desired_controls - applied)
        else:
            applied = desired_controls
            next_applied = desired_controls
        vehicle = self.model.step(states[..., :7], applied, cfg.dt)
        return np.concatenate((vehicle, next_applied), axis=-1)

    def _broadcast(self, states, controls):
        states = self._states(states)
        controls = self.clip_controls(controls)
        batch_shape = np.broadcast_shapes(states.shape[:-1], controls.shape[:-1])
        return (np.broadcast_to(states, batch_shape + (self.state_dim,)),
                np.broadcast_to(controls, batch_shape + (self.control_dim,)))

    def step(self, states: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """1스텝 평균 예측. (9,) 또는 (K, 9) 등을 지원한다.

        하나의 상태와 K개 입력을 주면 K개 다음 상태를 반환한다.
        출력의 비정상 상태는 숨기지 않는다. controller에서 큰 비용을 부여해야 한다.
        """
        states, controls = self._broadcast(states, controls)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return self._advance(states, controls)

    def predict_derivatives(self, states: np.ndarray, controls: np.ndarray,
                            observed: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """[dvx, dvy, dr, droll]의 평균·표준편차를 반환한다.

        observed=True는 관측 잡음을 포함한다. 반환된 표준편차는 step/rollout에
        자동 적용되지 않으며, 미래 상태의 표준편차도 아니다.
        """
        states, controls = self._broadcast(states, controls)
        applied = states[..., 7:9] if self.config.actuator_enabled else controls
        return self.model.predict(states[..., :7], applied, observed=observed)

    def rollout(self, initial_states: np.ndarray,
                control_sequences: np.ndarray) -> np.ndarray:
        """후보 입력 (..., N, 2)을 예측 궤적 (..., N+1, 9)로 변환한다.

        initial_states=(9,), control_sequences=(K, N, 2)이면 (K, N+1, 9).
        궤적의 0번은 초기 상태다. 이후 실제 상태로 보정하지 않는다.
        발산한 후보에는 NaN/Inf가 남을 수 있으므로 controller에서 배제해야 한다.
        """
        initial_states = self._states(initial_states)
        controls = self.clip_controls(control_sequences)
        if controls.ndim < 2 or controls.shape[-2] < 1:
            raise ValueError("control_sequences must have shape (..., N, 2), N >= 1")
        batch_shape = np.broadcast_shapes(initial_states.shape[:-1], controls.shape[:-2])
        horizon = controls.shape[-2]
        controls = np.broadcast_to(controls, batch_shape + (horizon, self.control_dim)) #
        states = np.broadcast_to(initial_states, batch_shape + (self.state_dim,))
        trajectory = np.empty(batch_shape + (horizon + 1, self.state_dim), dtype=float)
        trajectory[..., 0, :] = states #궤적의 0번에 처음 상태 저장
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for t in range(horizon):
                states = self._advance(states, controls[..., t, :])
                trajectory[..., t + 1, :] = states
        return trajectory

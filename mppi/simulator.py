"""학습된 평균 동역학으로 실행하는 단일 차량 테스트 환경.

controller의 미래 예측과 달리, 선택된 입력 하나만 적용하고 현재 상태를 갱신한다.
같은 모델을 예측과 환경에 사용하므로 실제 차량에 대한 독립적인 성능 검증은 아니다.
트랙 이탈, 모델 사용 범위 초과, 시간 제한은 종료 원인으로 구분한다.
"""

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np

from .config import Config
from .dynamic_adaptor import DynamicsAdaptor


@dataclass(frozen=True)
class CircularTrack:
    """반시계 방향 원형 테스트 트랙. 길이 단위는 m.

    h = (중심으로부터 거리 - radius) / half_width
    중심선은 h=0, 안쪽/바깥쪽 경계는 h=-1/+1이다.
    """

    radius: float = 3.0
    half_width: float = 0.75
    center: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if not (np.isfinite(self.radius) and np.isfinite(self.half_width)
                and 0 < self.half_width < self.radius):
            raise ValueError("Track requires 0 < half_width < radius, both finite")
        center = np.asarray(self.center, dtype=float)
        if center.shape != (2,) or not np.isfinite(center).all():
            raise ValueError("center must contain two finite coordinates")

    def track_error(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """스칼라 또는 배열 위치에 대한 정규화 트랙 오차."""
        distance = np.hypot(np.asarray(px) - self.center[0],
                            np.asarray(py) - self.center[1])
        return (distance - self.radius) / self.half_width

    def initial_state(self, speed: float) -> np.ndarray:
        """원의 오른쪽 중심선에서 위쪽을 향해 주행하는 초기 상태."""
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError("speed must be finite and positive")
        return np.array([self.center[0] + self.radius, self.center[1],
                         np.pi / 2, 0.0, speed, 0.0, speed / self.radius])


@dataclass(frozen=True)
class SimulationStep:
    """입력 한 번 적용 후 결과. state/control은 환경 내부 배열의 복사본."""

    state: np.ndarray
    control: np.ndarray
    time: float
    track_error: float
    done: bool
    reason: str | None
    applied_control: np.ndarray  # 이번 차량 적분에 실제로 사용한 값 (스텝 시작의 a)


class Simulator:
    def __init__(self, config: Config, dynamics: DynamicsAdaptor,
                 track_error: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 initial_state: np.ndarray):
        if config != dynamics.config:
            raise ValueError("Simulator and dynamics must use the same Config")
        self.config = config
        self.dynamics = dynamics
        self.track_error = track_error
        self._initial_state = np.asarray(initial_state, dtype=float).copy()
        self.reset()

    @property
    def state(self) -> np.ndarray:
        """외부에서 배열을 수정해도 환경 상태가 바뀌지 않도록 복사해서 반환한다."""
        return self._state.copy()

    @property
    def time(self) -> float:
        return self.steps * self.config.dt

    def _status(self, state: np.ndarray) -> tuple[float, str | None]:
        """종료 원인을 판정한다. 트랙 이탈과 수치 발산은 구분한다."""
        cfg = self.config
        if not np.isfinite(state).all():
            return float("nan"), "nonfinite_state"
        h_array = np.asarray(self.track_error(state[0], state[1]), dtype=float)
        if h_array.shape != ():
            raise ValueError("track_error must return a scalar for scalar positions")
        h = float(h_array)
        if not np.isfinite(h):
            raise ValueError("track_error returned a nonfinite value for a finite state")
        if abs(h) > 1.0:
            return h, "off_track"
        if state[4] <= cfg.min_prediction_vx:
            return h, "low_speed"
        if np.hypot(state[4], state[5]) > cfg.max_prediction_speed:
            return h, "speed_limit"
        if abs(state[6]) > cfg.max_prediction_yaw_rate:
            return h, "yaw_rate_limit"
        if abs(state[3]) > cfg.max_prediction_roll:
            return h, "roll_limit"
        if (np.any(state[7:9] < np.asarray(cfg.control_min) - 1e-12)
                or np.any(state[7:9] > np.asarray(cfg.control_max) + 1e-12)):
            return h, "actuator_limit"
        return h, None

    def reset(self, initial_state: np.ndarray | None = None) -> np.ndarray:
        """지정 상태 또는 생성 시 초기 상태로 복원한다. controller.reset()은 별도 호출한다."""
        state = np.asarray(self._initial_state if initial_state is None
                           else initial_state, dtype=float)
        if state.shape == (7,):
            state = self.dynamics.augment_state(state)
        if state.shape != (9,):
            raise ValueError("initial_state must have shape (7,) or (9,)")
        h, reason = self._status(state)
        if reason is not None:
            raise ValueError(f"Invalid initial state: {reason}")
        self._state = state.copy()
        self.steps = 0
        self.done = False
        self.reason = None
        self.current_track_error = h
        return self.state

    def step(self, control: np.ndarray) -> SimulationStep:
        """[steering, throttle] 하나를 dt 동안 적용한다. 종료 후에는 reset이 필요하다."""
        if self.done:
            raise RuntimeError(f"Simulation already ended ({self.reason}); call reset()")
        control = np.asarray(control, dtype=float)
        if control.shape != (2,):
            raise ValueError("control must have shape (2,): [steering, throttle]")
        desired = self.dynamics.clip_controls(control)
        applied = self._state[7:9].copy() if self.config.actuator_enabled else desired.copy()
        next_state = np.asarray(self.dynamics.step(self._state, desired), dtype=float)
        if next_state.shape != (9,):
            raise ValueError("Dynamics must return one state with shape (9,)")
        h, reason = self._status(next_state)
        self._state = next_state.copy()
        self.steps += 1
        if reason is None and self.steps >= self.config.max_steps:
            reason = "time_limit"
        self.current_track_error = h
        self.reason = reason
        self.done = reason is not None
        return SimulationStep(self.state, desired.copy(), self.time,
                              h, self.done, self.reason, applied)

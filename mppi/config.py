"""단일 차량 MPPI의 초기 실험 설정.

상태: 차량 상태 7개 + 실제 적용 중인 steering, throttle.
입력: 원하는 steering, throttle. 구동기 응답을 거친 값이 차량 모델에 들어간다.

"""

from dataclasses import dataclass
from pathlib import Path
import math


PROJECT_DIR = Path(__file__).resolve().parent.parent
VEHICLE_STATE_NAMES = ("px", "py", "yaw", "roll", "vx", "vy", "yaw_rate")
STATE_NAMES = VEHICLE_STATE_NAMES + ("applied_steering", "applied_throttle")
CONTROL_NAMES = ("steering", "throttle")


@dataclass(frozen=True)
class Config:
    # 25개 기저 모델. 27개 모델은 ds 인터페이스가 달라 별도 연결이 필요하다.
    model_path: Path = PROJECT_DIR / "basis25_nll" / "trained_model" / "dynamics_model.npz"
    output_dir: Path = PROJECT_DIR / "mppi" / "results"

    dt: float = 1.0 / 30.0
    horizon_steps: int = 75  # 75 / 30 = 2.5초 예측
    num_samples: int = 1024
    num_iterations: int = 1
    temperature: float = 1.0  # fixed 정규화 모드에서만 사용하는 가중치 온도
    weight_temperature_mode: str = "nominal_cost"
    cost_normalization_alpha: float = 20.0  # S_nominal / alpha; 논문은 10, Hunter 조건 검증값은 20
    seed: int = 7

    # 학습 데이터에서 확인한 입력 범위.
    steering_min: float = -0.5235988
    steering_max: float = 0.5235988
    throttle_min: float = 0.0
    throttle_max: float = 1.0

    # 후보 입력에 더할 Gaussian 탐색 잡음의 표준편차.
    # Bayesian dynamics가 반환하는 예측 표준편차와는 별개.
    steering_noise_std: float = 0.08  # rad
    throttle_noise_std: float = 0.15
    nominal_steering: float = 0.0
    nominal_throttle: float = 0.3
    actuator_rate: float = 10.0  # da/dt = k * (u_des - a)
    actuator_enabled: bool = True  # False는 구동기 효과 분리 실험용
    initial_applied_steering: float = 0.0
    initial_applied_throttle: float = 0.3

    # 첫 주행 실험은 낮은 속도에서 시작한다. 0 -> 목표 속도 가속 실험은 나중에 수행.
    target_speed: float = 1.0  # m/s

    # 논문 비용식에서 참조한 가중치. 실제 비용 함수의 정의와 맞춰 검증해야 한다.
    # track_error는 논문처럼 정규화된 트랙 오차를 사용할 때의 계수다.
    w_track: float = 50.0
    w_speed: float = 75 #2.5
    w_control: float = 10.0
    w_slip: float = 40.0
    slip_angle_threshold: float = 0.25  # rad
    # R=2*w_control*I, R=lambda*k^2/(Bc^2). 아래 lambda이면 Bc=0.01*I.
    # 탐색 command 분산은 위 std로 지정하고, nu=탐색분산/기본분산으로 유도한다.
    # 가중치 정규화 온도와 이 물리/확률 모델의 lambda는 별개의 설정이다.
    path_integral_lambda: float = 2.0e-5

    # 후보 예측의 유효성 검사. 차량 하드웨어의 안전 한계를 의미하지 않는다.
    min_prediction_vx: float = 0.2  # 학습에서 사용한 전진 속도 범위의 하한
    max_prediction_speed: float = 20.0  # 아래 상한은 기존 동역학 평가와 동일
    max_prediction_yaw_rate: float = 10.0
    max_prediction_roll: float = 1.5

    # 구현 확인용 시뮬레이션 길이. 실제 실행 속도 보장을 뜻하지 않는다.
    max_steps: int = 900  # 시뮬레이션 시간 기준 30초

    @property
    def horizon_seconds(self) -> float:
        return self.dt * self.horizon_steps

    @property
    def control_min(self) -> tuple[float, float]:
        return self.steering_min, self.throttle_min

    @property
    def control_max(self) -> tuple[float, float]:
        return self.steering_max, self.throttle_max

    @property
    def control_noise_std(self) -> tuple[float, float]:
        return self.steering_noise_std, self.throttle_noise_std

    @property
    def control_cost_diag(self) -> tuple[float, float]:
        return 2.0 * self.w_control, 2.0 * self.w_control

    @property
    def base_command_variance(self) -> tuple[float, float]:
        return tuple(self.path_integral_lambda / (r * self.dt)
                     for r in self.control_cost_diag)

    @property
    def exploration_variance_ratio(self) -> tuple[float, float]:
        """nu는 분산의 비율이다. 잡음 진폭의 비율은 sqrt(nu)."""
        return tuple(std**2 / base for std, base in
                     zip(self.control_noise_std, self.base_command_variance))

    @property
    def base_actuator_diffusion_std(self) -> tuple[float, float]:
        return tuple(self.actuator_rate * math.sqrt(self.path_integral_lambda / r)
                     for r in self.control_cost_diag)

    def __post_init__(self) -> None:
        for name in ("dt", "temperature", "target_speed",
                     "steering_noise_std", "throttle_noise_std", "w_control",
                     "actuator_rate", "path_integral_lambda", "cost_normalization_alpha"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("horizon_steps", "num_samples", "num_iterations", "max_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not self.steering_min <= self.nominal_steering <= self.steering_max:
            raise ValueError("nominal_steering must be within steering limits")
        if not self.throttle_min <= self.nominal_throttle <= self.throttle_max:
            raise ValueError("nominal_throttle must be within throttle limits")
        if not self.steering_min <= self.initial_applied_steering <= self.steering_max:
            raise ValueError("initial_applied_steering must be within steering limits")
        if not self.throttle_min <= self.initial_applied_throttle <= self.throttle_max:
            raise ValueError("initial_applied_throttle must be within throttle limits")
        if self.weight_temperature_mode not in ("nominal_cost", "fixed", "theoretical"):
            raise ValueError("Unknown weight_temperature_mode")
        if not isinstance(self.actuator_enabled, bool):
            raise ValueError("actuator_enabled must be bool")
        if self.actuator_enabled and self.actuator_rate * self.dt > 1.0:
            raise ValueError("Euler actuator requires actuator_rate * dt <= 1")
        if any(nu < 1.0 for nu in self.exploration_variance_ratio):
            raise ValueError("Exploration variance must be at least base variance (nu >= 1)")

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import io
import zipfile

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

CSV_COLUMNS = [
    "timestamp", "throttle", "steering", "leftTicks", "rightTicks",
    "posX", "posY", "posZ", "roll", "pitch", "yaw", "speed",
    "angX", "angY", "angZ", "accX", "accY", "accZ",
]

TARGET_NAMES = ("dvx", "dvy", "dr", "droll")
BASIS_COUNT = 27
WHEELBASE = 0.55
LF = 0.22
LR = WHEELBASE - LF

@dataclass
class Trajectory:
    times: np.ndarray                # (N,), seconds from start
    states: np.ndarray               # (N, 7), [px, py, yaw, roll, vx, vy, yaw_rate]
    controls: np.ndarray             # (N, 2), [steering, throttle]
    targets: np.ndarray              # (N, 4), [dvx, dvy, dr, droll]
    valid: np.ndarray                # (N,), valid regression points
    ds: np.ndarray                   # (N,), steering derivative, optional

project_dir = Path(__file__).resolve().parent.parent
zip_path = project_dir / "data" / "AutoDRIVE-Hunter-SE-Dataset-on-road-dataset.zip"

def list_csv_members(zf: zipfile.ZipFile) -> list[str]:
    """Return only the four 30-Hz driving maneuver CSVs."""
    return sorted(
        name for name in zf.namelist()
        if name.endswith("V1 Log.csv")
        and any(f"/{maneuver}_30_hz/" in name
                for maneuver in ("straight", "skidpad", "fishhook", "slalom"))
    )

def load_trajectory(zf: zipfile.ZipFile, member: str, window: int = 11) -> Trajectory:
    #Read one independent run. Never join samples across runs.
    with zf.open(member) as f:
        df = pd.read_csv(f, header=None, names=CSV_COLUMNS)

    #repeated synthetic-looking first row elimination
    row = df.iloc[0]
    if (np.isclose(row["steering"], 0.1)
        and np.isclose(row["leftTicks"], 0.2)
        and np.isclose(row["rightTicks"], 0.3)
        and np.isclose(row["speed"], 0.4)):
        df = df.iloc[1:].copy()

    parsed = pd.to_datetime(df["timestamp"],
                                format="%Y_%m_%d_%H_%M_%S_%f", errors="coerce")
    df["parsed_time"] = parsed
    df = df.dropna(subset=["parsed_time"]).reset_index(drop=True)

    times = (df["parsed_time"] - df["parsed_time"].iloc[0]).dt.total_seconds().to_numpy()
    
    keep = np.r_[True, np.diff(times) > 0]
    df = df.loc[keep].reset_index(drop=True)
    times = times[keep]
    if len(times) < 25:
        raise ValueError(f"Run has too few samples: {member}")

    # Smooth measurements before differentiating; differentiation amplifies noise.
    window = min(window, len(times) if len(times) % 2 else len(times) - 1)
    window = max(window, 5)

    raw_positions = df[["posX", "posY"]].to_numpy(float)

    position_changes = np.diff(raw_positions, axis=0)
    distances = np.linalg.norm(position_changes, axis=1)
    times_gaps = np.diff(times)

    speed_limit = 6.0
    position_tolerance = 0.1

    allowed_distance = speed_limit * times_gaps + position_tolerance
    jump_indices = np.flatnonzero(distances > allowed_distance)

    def smooth(values: np.ndarray) -> np.ndarray:
        return savgol_filter(values, window_length=window,
                             polyorder=3, mode="interp")

    px = smooth(df["posX"].to_numpy(float))
    py = smooth(df["posY"].to_numpy(float))
    yaw = smooth(np.unwrap(df["yaw"].to_numpy(float)))
    roll = smooth(np.unwrap(df["roll"].to_numpy(float)))
    yaw_rate = smooth(df["angZ"].to_numpy(float))

    vx_world = np.gradient(px, times)
    vy_world = np.gradient(py, times)

    # Transform world-frame velocities to body-frame velocities.
    vx = smooth(vx_world * np.cos(yaw) + vy_world * np.sin(yaw))
    vy = smooth(-vx_world * np.sin(yaw) + vy_world * np.cos(yaw))

    steering = df["steering"].to_numpy(float)
    throttle = df["throttle"].to_numpy(float)

    ds = np.full_like(steering, np.nan)
    ds[1:] = np.diff(steering) / np.diff(times)

    states = np.column_stack([px, py, yaw, roll, vx, vy, yaw_rate])
    controls = np.column_stack([steering, throttle])
    # vx, vy, yaw_rate, roll derivatives as targets
    targets = np.column_stack([
        np.gradient(vx, times),
        np.gradient(vy, times),
        np.gradient(yaw_rate, times),
        np.gradient(roll, times),
    ])

    valid = np.all(np.isfinite(states), axis=1)
    valid &= np.all(np.isfinite(targets), axis=1)
    valid &= np.all(np.isfinite(controls), axis=1)
    valid &= np.isfinite(ds)
    valid &= (vx > 0.2)  # initial model: only forward movement; avoids slip singularity
    valid[:window] = False
    valid[-window:] = False

    margin = window + 1
    for i in jump_indices:
        start = max(0, i - margin)
        end = min(len(valid), i + margin + 1)
        valid[start:end] = False
    
    return Trajectory(times, states, controls, targets, valid, ds)
    
def basis_functions(states: np.ndarray, controls: np.ndarray, ds: np.ndarray) -> np.ndarray:
    """25 physics-motivated features"""
    x = np.asarray(states, dtype=float)
    u = np.asarray(controls, dtype=float)
    vx, vy, r, roll = x[..., 4], x[..., 5], x[..., 6], x[..., 3]
    steering, throttle = u[..., 0], u[..., 1]
    vx_safe = np.maximum(vx, 0.2)

    alpha_f = np.arctan2(vy + LF * r, vx_safe) - steering
    alpha_r = np.arctan2(vy - LR * r, vx_safe)

    # Safe for initial forward-driving fit. Validate steering/sign convention.
    gf = np.tan(np.clip(alpha_f, -1.2, 1.2))
    gr = np.tan(np.clip(alpha_r, -1.2, 1.2))
    sd = np.sin(steering)

    return np.stack([
        throttle, throttle**2, throttle**3,                  # 1-3 스로틀 입력의 비선형 특성
        vx, vx**2, vx**3,                                   # 4-6 비선형 속도 의존성
        vy, r, r * vy, r * vx, vy / vx_safe,                 # 7-11 횡속도, 회전 운동 결합항
        sd,                                                 # 12 조향에 비선형 영향
        gf, gf * np.abs(gf), gf**3,                         # 13-15 전륜 슬립각
        gr, gr * np.abs(gr), gr**3,                         # 16-18 후륜 슬립각
        sd * gf, sd * gf * np.abs(gf), sd * gf**3,          # 19-21 조향과 전륜 슬립각의 상호작용
        roll, roll * r, roll * vx, roll * vx * r,            # 22-25 롤과 회전 운동, 속도와의 상호작용
        ds, vx * ds / np.cos(steering)**2
    ], axis=-1)

class BayesianDynamics:
    """Four independent output regressions with Gaussian coefficient posteriors.

    Feature centering and a flat intercept prior yield:
        bias_mean = mean(Y)
        weight_cov = (Z.T @ Z / noise_var + I / prior_var)^(-1)
        weight_mean = weight_cov @ Z.T @ (Y - mean(Y)) / noise_var
    """

    def __init__(self):
        self.feature_mean = None
        self.feature_scale = None
        self.bias = None
        self.bias_variance = None
        self.weights = None
        self.covariances = None
        self.noise_stds = None
        self.prior_stds = None

    def fit(self, X: np.ndarray, Y: np.ndarray,
            noise_stds: np.ndarray, prior_stds: np.ndarray) -> "BayesianDynamics":
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        if X.ndim != 2 or X.shape[1] != BASIS_COUNT or Y.shape != (len(X), 4):
            raise ValueError("Expected X(N,27), Y(N,4)")
        self.noise_stds = np.asarray(noise_stds, float)
        self.prior_stds = np.asarray(prior_stds, float)
        if (self.noise_stds.shape != (4,) or self.prior_stds.shape != (4,)
            or np.any(self.noise_stds <= 0) or np.any(self.prior_stds <= 0)):
            raise ValueError("noise_stds and prior_stds must be positive arrays of length 4")

        self.feature_mean = X.mean(axis=0)
        self.feature_scale = X.std(axis=0)
        # Near-constant features should not be amplified.
        self.feature_scale = np.where(self.feature_scale < 1e-9, 1.0, self.feature_scale)
        Z = (X - self.feature_mean) / self.feature_scale

        self.bias = Y.mean(axis=0)
        self.bias_variance = self.noise_stds**2 / len(Y)
        self.weights = np.empty((BASIS_COUNT, 4))
        self.covariances = np.empty((4, BASIS_COUNT, BASIS_COUNT))

        gram = Z.T @ Z
        identity = np.eye(BASIS_COUNT)
        for j in range(4):
            sigma2 = self.noise_stds[j] ** 2
            tau2 = self.prior_stds[j] ** 2
            A = gram / sigma2 + identity / tau2
            rhs = Z.T @ (Y[:, j] - self.bias[j]) / sigma2
            self.weights[:, j] = np.linalg.solve(A, rhs)
            self.covariances[j] = np.linalg.solve(A, identity)
        return self

    def predict_mean(self, states: np.ndarray, controls: np.ndarray, ds: np.ndarray) -> np.ndarray:
        """불확실성이 필요 없는 궤적 계산에서는 평균만 계산한다."""
        X = basis_functions(states, controls, ds)
        Z = (X - self.feature_mean) / self.feature_scale
        return self.bias + Z @ self.weights

    def predict(self, states: np.ndarray, controls: np.ndarray, ds: np.ndarray,
                observed: bool = False) -> tuple[np.ndarray, np.ndarray]:
        X = basis_functions(states, controls, ds)
        Z = (X - self.feature_mean) / self.feature_scale
        mean = self.bias + Z @ self.weights
        variances = np.stack([
            np.einsum("...i,ij,...j->...", Z, self.covariances[j], Z)
            + self.bias_variance[j]
            + (self.noise_stds[j] ** 2 if observed else 0.0)
            for j in range(4)
        ], axis=-1)
        return mean, np.sqrt(np.maximum(variances, 0.0))


    def step(self, state: np.ndarray, control: np.ndarray, ds: np.ndarray, dt: float) -> np.ndarray:
        """Deterministic mean dynamics, forward Euler. Control treated as applied input."""
        state = np.asarray(state, float)
        mean = self.predict_mean(state, control, ds)
        px, py, yaw, roll, vx, vy, r = np.moveaxis(state, -1, 0) # ->(7,N)로 변함
        dvx, dvy, dr, droll = np.moveaxis(mean, -1, 0) # ->(4,N)로 변함

        delta = np.stack([
            vx * np.cos(yaw) - vy * np.sin(yaw),
            vx * np.sin(yaw) + vy * np.cos(yaw),
            r, droll, dvx, dvy, dr
        ], axis=-1)
        return state + float(dt) * delta

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, feature_mean=self.feature_mean,
            feature_scale=self.feature_scale, bias=self.bias,
            bias_variance=self.bias_variance, weights=self.weights,
            covariances=self.covariances, noise_stds=self.noise_stds,
            prior_stds=self.prior_stds)

    @classmethod
    def load(cls, path: str | Path) -> "BayesianDynamics":
        result = cls()
        with np.load(path) as data:
            for name in ("feature_mean", "feature_scale", "bias",
                         "bias_variance", "weights", "covariances",
                         "noise_stds", "prior_stds"):
                setattr(result, name, data[name].copy())
        return result

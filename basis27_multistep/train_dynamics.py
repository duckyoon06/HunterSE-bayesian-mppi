#Train and validate Hunter SE Bayesian dynamics from the original ZIP.

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import zipfile

import numpy as np
from dynamics import (
    BayesianDynamics, TARGET_NAMES, basis_functions,
    list_csv_members, load_trajectory,
)
from test_dynamics import choose_windows, evaluate_window

def split_runs(members: list[str], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    buckets = defaultdict(list)
    for member in members:
        parts = member.split("/")
        folder = next(p for p in parts if p.endswith("_30_hz"))
        subgroup = parts[parts.index(folder) + 1]
        buckets[(folder, subgroup)].append(member)
    """
    {
    ("fishhook_30_hz", "ccw"): [파일1, 파일2, ...],
    ("fishhook_30_hz", "cw"):  [파일3, 파일4, ...],
    ("slalom_30_hz", "ccw"):   [파일5, 파일6, ...],
    ...
    }
    """
    out = {"train": [], "validation": [], "test": []}
    for key, bucket in sorted(buckets.items()):
        bucket = sorted(bucket)
        rng.shuffle(bucket)
        n = len(bucket)
        if n < 3:
            raise ValueError(f"Need >= 3 runs per subgroup for split: {key}")
        n_val = max(1, round(n * 0.15))
        n_test = max(1, round(n * 0.15))
        if n - n_val - n_test < 1:
            raise ValueError(f"Not enough runs: {key}")
        out["test"].extend(bucket[:n_test])
        out["validation"].extend(bucket[n_test:n_test + n_val])
        out["train"].extend(bucket[n_test + n_val:])
    return out

def build_matrix(zf: zipfile.ZipFile, members: list[str], stride: int):
    #stride는 데이터 샘플링 간격을 의미. 예를 들어, stride=2이면 유효한 인덱스 중에서 2번째마다 샘플을 선택
    #stride는 기본적으로 1로 설정하고, 샘플 수를 줄이고 싶으면 늘리면 됨.
    all_X, all_Y = [], []
    for member in members:
        traj = load_trajectory(zf, member)
        valid_indices = np.flatnonzero(traj.valid)[::stride]
        if len(valid_indices) == 0:
            continue
        all_X.append(basis_functions(traj.states[valid_indices],
                                     traj.controls[valid_indices],
                                     traj.ds[valid_indices],
                                     ))
        all_Y.append(traj.targets[valid_indices])
    if not all_X:
        raise ValueError("No valid training samples")
    return np.concatenate(all_X), np.concatenate(all_Y)
    #각 주행을 행 방향으로 이어붙임. -> X: (Ntot,27), Y: (Ntot,4)

def evaluate_validation_rollout(model, validation_trajectories, horizons = (0.5, 1.0, 2.5), windows_per_run = 8):
    results = {}
    for horizon in horizons:
        records = []
        for member, trajectory in validation_trajectories:
            windows = choose_windows(
                trajectory.times,
                trajectory.valid,
                horizon,
                count=windows_per_run
            )

            for start, stop in windows:
                record, _ = evaluate_window(
                    model,
                    trajectory,
                    start,
                    stop,
                    member,
                    horizon,
                    include_trace=False
                )
                records.append(record)

        successful = [
            r for r in records if not r["failed"]
        ]

        failure_rate = (
            sum(r["failed"] for r in records) / len(records)
            if records else 1.0
        )

        if successful:
            position_rmse = np.sqrt(np.mean([
                r["endpoint_position_error_m"] ** 2
                for r in successful
            ]))

            yaw_rmse = np.sqrt(np.mean([
                r["endpoint_yaw_error"] ** 2
                for r in successful
            ]))
        else:
            position_rmse = float("inf")
            yaw_rmse = float("inf")

        results[horizon] = {
            "failure_rate": failure_rate,
            "position_rmse": position_rmse,
            "yaw_rmse": yaw_rmse,
            "windows": len(records)
        }

    return results


def select_hyperparameters(X_train, Y_train, X_val, Y_val, validation_trajectories):
    """Independent per-output empirical validation of prior/noise scale.

    Optimize Gaussian predictive negative log-likelihood, not test error.
    """
    x_mean = X_train.mean(axis=0)
    x_scale = np.where(X_train.std(axis=0) < 1e-9, 1.0,
                       X_train.std(axis=0))
    Z_train = (X_train - x_mean) / x_scale
    Z_val = (X_val - x_mean) / x_scale

    gram = Z_train.T @ Z_train
    identity = np.eye(X_train.shape[1])
    target_mean = Y_train.mean(axis=0)
    centered_targets = Y_train - target_mean

    noise_stds = np.zeros(4)
    prior_stds = np.zeros(4)
    for j, name in enumerate(TARGET_NAMES):
        spread = max(float(np.std(Y_train[:, j])), 1e-3)
        cross = Z_train.T @ centered_targets[:, j]
        best = (float("inf"), None, None, None, None)
        for noise_factor in (0.05, 0.15, 0.5, 1.0, 2.0):
            noise_std = max(1e-4, spread * noise_factor)
            for prior_factor in (0.005, 0.01, 0.02, 0.03,
                                 0.05, 0.1, 0.5, 2.0):
                prior_std = max(1e-4, spread * prior_factor)
                sigma2, tau2 = noise_std**2, prior_std**2
                A = gram / sigma2 + identity / tau2
                coef = np.linalg.solve(A, cross / sigma2)
                cov = np.linalg.solve(A, identity)
                mu = target_mean[j] + Z_val @ coef
                # Predict *observations*, therefore include observation noise.
                var = (sigma2 + sigma2 / len(Y_train)
                       + np.einsum("ni,ij,nj->n", Z_val, cov, Z_val))
                var = np.maximum(var, 1e-12)
                nll = float(np.mean(0.5 * (np.log(2 * np.pi * var)
                                       + (Y_val[:, j] - mu)**2 / var)))
                if nll < best[0]:
                    best = (nll, noise_std, prior_std, noise_factor, prior_factor)
        noise_stds[j], prior_stds[j] = best[1:3]
        print(f"  {name:6s} validation NLL={best[0]: .5f}"
              f" | noise_std={noise_stds[j]:.5g}"
              f" | prior_std={prior_stds[j]:.5g}"
              f" | noise_factor={best[3]:.5g}"
              f" | prior_factor={best[4]:.5g}"
            )
    def selection_key(model):
        rollout_results = evaluate_validation_rollout(
            model,
            validation_trajectories,
            windows_per_run=2
        )
        #the badest failure rate
        failure_rate = max(result["failure_rate"] for result in rollout_results.values())
        # 위치와 방향 오차를 기준 단위로 나누어 합산
        # 아래 0.1m, 0.1rad는 예시 기준이며 모든 후보에 동일하게 적용
        rollout_error = float(np.mean([
            result["position_rmse"] / 0.1
            + result["yaw_rmse"] / 0.1
            for result in rollout_results.values()
        ]))

        Z = (X_val - model.feature_mean) / model.feature_scale
        mean = model.bias + Z @ model.weights

        variance = np.stack([
            np.einsum("ni,ij,nj->n", Z, model.covariances[k], Z)
            + model.bias_variance[k]
            + model.noise_stds[k] ** 2
            for k in range(4)
        ], axis=-1)

        variance = np.maximum(variance, 1e-12)
        nll = float(np.mean(
            0.5 * (
                np.log(2 * np.pi * variance)
                + (Y_val - mean) ** 2 / variance
            )
        ))

        return failure_rate, rollout_error, nll

    def ranking(key):
        failure_rate, rollout_error, nll = key

        if failure_rate <= 0.05:
            return (0, rollout_error, nll)
        else:
            return (1, failure_rate, rollout_error, nll)
    
    # NLL 기준 초기 모델도 비교 대상에 포함
    initial_model = BayesianDynamics().fit(
        X_train, Y_train,
        noise_stds=noise_stds,
        prior_stds=prior_stds,
    )
    current_key = selection_key(initial_model)

    candidates = [
        (nf, pf)
        for nf in (0.05, 0.15, 0.5, 1.0, 2.0)
        for pf in (0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.5, 2.0)
    ]

    # 한 출력의 설정만 변경하고 나머지 세 출력은 고정
    for j, name in enumerate(TARGET_NAMES):
        best_key = current_key
        best_noise = noise_stds.copy()
        best_prior = prior_stds.copy()

        spread = max(float(np.std(Y_train[:, j])), 1e-3)

        for noise_factor, prior_factor in candidates:
            candidate_noise = noise_stds.copy()
            candidate_prior = prior_stds.copy()

            candidate_noise[j] = max(1e-4, spread * noise_factor)
            candidate_prior[j] = max(1e-4, spread * prior_factor)

            candidate_model = BayesianDynamics().fit(
                X_train, Y_train,
                noise_stds=candidate_noise,
                prior_stds=candidate_prior,
            )
            key = selection_key(candidate_model)

            if ranking(key) < ranking(best_key):
                best_key = key
                best_noise = candidate_noise.copy()
                best_prior = candidate_prior.copy()

        # 이 출력의 후보 비교가 끝난 뒤에만 설정 갱신
        noise_stds = best_noise
        prior_stds = best_prior
        current_key = best_key

        print(
            f"{name}: failure={best_key[0]:.2%}, "
            f"rollout_error={best_key[1]:.5f}, "
            f"NLL={best_key[2]:.5f}",
            flush=True,
        )

    return noise_stds, prior_stds

        

def main():
    project_dir = Path(__file__).resolve().parent.parent
    data_path = (
        project_dir
        / "data"
        / "AutoDRIVE-Hunter-SE-Dataset-on-road-dataset.zip"
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=data_path)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "trained_model")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--stride", type=int, default=1)

    args = ap.parse_args()
    if args.stride < 1:
        ap.error("--stride must be >=1")
    args.out.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(args.data) as zf:
        members = list_csv_members(zf)
        print(f"Found {len(members)} driving CSV runs in ZIP")
        split = split_runs(members, args.seed)
        for part, names in split.items():
            print(f"  {part:10s} {len(names)} independent runs")
        X_tr, Y_tr = build_matrix(zf, split["train"], args.stride)
        X_va, Y_va = build_matrix(zf, split["validation"], args.stride)
        validation_trajectories = [
                    (member, load_trajectory(zf, member))
                    for member in split["validation"]
                ]

        print("Train samples:", len(X_tr), "Validation samples:", len(X_va))
        noise, prior = select_hyperparameters(X_tr, Y_tr, X_va, Y_va, validation_trajectories)

        # Refit with train+validation after hyperparameters are fixed.
        final_model = BayesianDynamics().fit(
            np.concatenate([X_tr, X_va]),
            np.concatenate([Y_tr, Y_va]),
            noise_stds=noise,
            prior_stds=prior
        )
        final_model.save(args.out / "dynamics_model.npz")
        manifest = {
            "source_zip_filename": args.data.name,
            "seed": args.seed,
            "stride": args.stride,
            "splits": split,
            "targets": list(TARGET_NAMES),
            "note": "Test runs were never used for hyperparameter selection or fitting.",
        }
        (args.out / "split.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print(f"Saved model and split into {args.out.resolve()}")


if __name__ == "__main__":
    main()


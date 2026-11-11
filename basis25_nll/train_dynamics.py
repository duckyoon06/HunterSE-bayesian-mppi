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
                                     traj.controls[valid_indices]))
        all_Y.append(traj.targets[valid_indices])
    if not all_X:
        raise ValueError("No valid training samples")
    return np.concatenate(all_X), np.concatenate(all_Y)
    #각 주행을 행 방향으로 이어붙임. -> X: (Ntot,25), Y: (Ntot,4)

def select_hyperparameters(X_train, Y_train, X_val, Y_val):
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

        print("Train samples:", len(X_tr), "Validation samples:", len(X_va))
        noise, prior = select_hyperparameters(X_tr, Y_tr, X_va, Y_va)

        # Refit with train+validation after hyperparameters are fixed.
        final_model = BayesianDynamics().fit(
            np.concatenate([X_tr, X_va]),
            np.concatenate([Y_tr, Y_va]),
            noise_stds=noise, prior_stds=prior
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


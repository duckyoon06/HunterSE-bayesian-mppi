# Cost/noise and actuator alignment results

## Implemented

- Archived pre-change source in baseline_source.json. No learned dynamics/training/model checkpoint was changed.
- Augmented the state from 7 to 9 with actual steering and throttle.
- Predictor and simulator share simultaneous Euler vehicle/actuator updates with k=10 /s.
- Replaced arbitrary importance_coefficient with the derived mean-shift and variance-change terms.
- R=20 I, path-integral lambda=2e-5, Bc=.01 I; baseline command variance=.00003 per channel.
- Exploration command std remains (.08,.15), giving variance ratios (213.333...,750).
- Sampling, likelihood correction and update use unclipped latent Gaussian noise; saturation is in the transition.
- Removed the extra noisy-candidate quadratic input penalty. Nominal control energy is included once.
- Separated fixed-lambda theoretical weighting from nominal-cost or fixed-temperature practical normalization.
- The paper reports alpha=10. It was poor in the initial 10 s Hunter screen (speed RMSE .486 m/s, track RMSE .442 m). Default alpha is adjusted to 20.
- Saved desired and applied commands separately and fixed diagnostic costs to represent actual applied cost terms.

## Validation: seeds 7, 8, 9; 30 seconds per run

| Version | Completion | Mean speed (m/s) | Speed RMSE (m/s) | Track RMSE (m) | Median ESS |
|---|---|---|---|---|---|
| baseline | 3/3 | 0.9545 to 0.9549 | 0.0476 to 0.0482 | 0.0176 to 0.0196 | 214.8587 to 216.5169 |
| adaptive20_validation | 3/3 | 0.9850 to 0.9868 | 0.0219 to 0.0251 | 0.0769 to 0.0783 | 41.9992 to 43.7319 |
| fixed03_validation | 3/3 | 0.9843 to 0.9855 | 0.0185 to 0.0213 | 0.0796 to 0.0809 | 94.4505 to 94.8492 |

## Interpretation

The mean applied step-to-step steering change falls from 0.03590 to 0.02392 rad; throttle change falls from 0.03141 to 0.02401.
Desired commands can fluctuate more than before. The actuator reduces changes in the values actually fed to the vehicle.
Speed tracking improves, but track RMSE increases. Initial yaw-rate overshoot remains and is larger in some runs. This is not an across-the-board performance improvement.
The updated structure clarifies the stochastic-control formulation and actuator timing. It is not proof of exact 2016 reproduction.
The plant changes when the actuator is enabled. Old/new RMSE is a combined algorithm-and-plant comparison, not an isolated algorithm ranking.
The correction_only screen disables the actuator with the new cost/normalization. The actuator_only screen uses the archived controller cost and normalization with a 9-state interface. Both run 10 s, seed 7 only.

## Tests

9 unittest checks passed: Gaussian log-density identity including saturation cases, R/noise relation, actuator Euler response, simulator/rollout agreement, normalization, invalid-candidate handling, cost decomposition, command limits and fallback limits.
Runner integration checks passed for 9-state CSV/NPZ serialization, desired/applied inputs, and prediction vs execution timing.

## Reproduce

    python -m mppi.run
    python -m mppi.run --weight-temperature-mode fixed --temperature 0.3
    python -m unittest mppi.test_mppi -v

## Scope and assumptions

Same learned model for plant and prediction; no passive Bayesian noise. Track exits still get infinite cost rather than frozen finite-cost rollouts. Hunter track, target speed, basis and compute differ from AutoRally.
The latent Gaussian density identity is exact. Saturated discrete-command dynamics is an implementation adaptation of the continuous diffusion formulation.
Adaptive/fixed practical temperature is explicitly different from the theoretical lambda. It must not be described as the exact fixed-lambda free-energy weights.
The learned model input is assumed to be applied actuation; additional lag could double count response if the dataset input already includes that lag. Independent actuator calibration remains needed.
Initial applied input and nominal remain (0,.3), while the initial vehicle yaw rate is 1/3. This mismatch has not been removed or hidden.
Baseline files are referenced in baseline_metrics.json. Full theory and timing are documented in ../../FORMULATION.md.
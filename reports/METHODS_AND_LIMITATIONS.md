# Methods and limitations

## Vehicle model
- Original data: AutoDRIVE Hunter SE on-road driving logs, 30 Hz, four maneuvers (straight, skidpad, fishhook, slalom).
- Train/validation/test split is run-wise and stratified by maneuver subgroups; individual samples in different runs are never joined into a single continuous rollout.
- Nonlinear, physics-motivated features are input to independent Bayesian linear regressions for longitudinal/lateral acceleration, yaw acceleration, and roll rate.
- The original ICRA 2016 paper says 24 basis functions but does not publish their exact list. The 25-feature model uses a different, documented selection inspired in part by the 2019 doctoral appendix. A strict original-paper reproduction has not been demonstrated.
- 25-feature training selects per-output prior and observation noise by validation predictive negative log-likelihood; 27-feature model also uses rollout-based hyperparameter criteria. Therefore this comparison is NOT an isolated causal ablation of feature count.
- Evaluation: on held-out runs, using the actual initial state plus recorded controls. Overlapping test windows are correlated. Failed windows are omitted from endpoint RMSE and reported separately.

## Controller
- The MPPI implementation uses a learned 25-feature deterministic mean dynamics model in both the controller and the simulated vehicle. This is a closed-loop controller test against the same model, not independent plant validation.
- The 9-state formulation augments vehicle dynamics with actuator states; the model's command-to-applied-actuator mapping may need independent calibration.
- Cost includes track deviation, speed deviation, sideslip penalty, and control; Gaussian exploration uses the correction terms described in mppi/FORMULATION.md.
- Hard rejection of invalid trajectories, omitted process-uncertainty propagation, Euler integration, a Hunter-size circular track, and practical temperature choices differ from Williams et al. ICRA 2016.
- A completed 30-second run means that the requested simulation time was reached; it is not proof of robustness, real-time operation, physical deployment, or completion of an exact lap count.
- The 30-second MPPI validation seed 7 used 1024 sampled trajectories per step, horizon 75 at dt=1/30 s, target speed 1 m/s, circular track radius 3 m / half width 0.75 m.
- The observed per-command wall time can exceed 1/30 s on CPU. No real-time guarantee is claimed.

## Versioning
- Results in `mppi/results/importance_experiment` (not included here) were produced with an *older* controller formulation using an importance coefficient. They should not be presented as outcomes of the current code.
- The bundled `mppi/results/paper_alignment` report documents later updates; the selected small JSON/CSV files are evidence of the specific runs, not independent retests performed during repo packaging.

## Provenance and distribution
- The dataset is attributed to Tinker-Twins/AutoDRIVE-Hunter-SE-Dataset (on-road-dataset branch, BSD-2-Clause); the dataset is not redistributed here.
- The PDF of the 2016 IEEE paper is not redistributed. Cite the DOI instead.
- No license for the author's *new source code* has been chosen automatically. Add a LICENSE file after deciding the intended software license.

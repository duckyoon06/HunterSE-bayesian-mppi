# Cost, exploration noise and actuator model

This implementation follows the structure of Williams et al., ICRA 2016,
equations (6), (29)-(34), and the actuator model in section IV.
Source: https://doi.org/10.1109/ICRA.2016.7487277

It keeps the learned basis25 vehicle model and the Hunter-scale track/cost settings.
It is not an exact reproduction of all AutoRally experiments.

## State and actuator timing

The state is [px, py, yaw, roll, vx, vy, yaw_rate, applied_steering, applied_throttle].
The optimized command is the desired actuator target, not the applied actuator state.
For each Euler step, with a = actual actuator and g = command saturation:

    vehicle_next = learned_vehicle_step(vehicle, a, dt)
    a_next = a + k * dt * (g(desired_command) - a)

Both predictions and the simulator use DynamicsAdaptor._advance. The vehicle uses
the actuator at the beginning of the step, as required by simultaneous explicit Euler.
k=10 /s follows the paper. We require k*dt <= 1 so bounded targets preserve bounds.
Setting actuator_enabled=False is an ablation, not the default physical model.

The learned vehicle model is assumed to take applied actuation. If the dataset input
actually already includes an actuator lag through command-to-response identification,
this additional actuator requires calibration against independent data.

## Gaussian measures and R

Use latent (unclipped) commands v = U + epsilon:

    p(v) = N(0, C0)
    q(v) = N(U, Ce)
    Ce = diag(exploration_std**2)
    R = 2 * w_control * I
    C0 = lambda * inverse(R * dt)
    nu_j = Ce_jj / C0_jj >= 1

Here nu is a VARIANCE ratio; its square root is the noise amplitude ratio.
The corresponding actuator diffusion satisfies:

    Bc_j = k * sqrt(lambda / R_j)
    R_j = lambda * k**2 / Bc_j**2

For lambda=2e-5, k=10, w_control=10 and dt=1/30:

    R = diag(20, 20)
    Bc = diag(.01, .01)
    C0 = diag(.00003, .00003)
    Ce = diag(.0064, .0225)
    nu = (213.333..., 750)

Thus the small baseline diffusion is distinct from the larger exploration noise.
No free importance_coefficient multiplies the result.

## Likelihood-ratio cost

Up to a determinant term constant across candidates, -lambda*log(p(v)/q(v)) is:

    control_energy = .5 * dt * sum_t,j R_j * U_tj**2
    mean_shift = dt * sum_t,j R_j * U_tj * epsilon_tj
    variance_change = .5 * dt * sum_t,j R_j * (1 - 1/nu_j) * epsilon_tj**2

The total candidate cost is:

    state_cost + control_energy + mean_shift + variance_change

state_cost includes track, speed and sideslip penalties. The control energy is
included exactly once. The old extra penalty on every noisy candidate command
is removed; it would specify a different objective in addition to the Gaussian prior.
control_energy is common to candidates within one iteration, so it does not change
normalized weights, but it is retained for reporting and nominal-cost normalization.

Sampling, correction and the weighted update use the ORIGINAL Gaussian epsilon.
Saturation is a deterministic part of the transition, not a change to the latent
Gaussian density. This avoids evaluating a Gaussian density at clipped samples.
The latent-input density identity is exact; equivalence to the continuous diffusion
path measure is limited by Euler discretization and saturation.

## Weight normalization is separate from lambda

The theoretical weight is exp(-(cost-min_cost)/lambda). The paper's experiments
instead report normalization by nominal trajectory cost with alpha=10 (section V-A).
We expose the choices explicitly:

- theoretical: effective temperature = path_integral_lambda.
- fixed: effective temperature = temperature.
- nominal_cost: effective temperature = nominal_cost / cost_normalization_alpha.

The nominal reference includes state cost and control energy, with zero perturbation.
If the nominal is invalid, use the lowest finite sampled state-plus-control cost as
reference, since the paper does not specify our hard-rejection fallback.
The denominator has a small positive floor. Adaptive normalization is a practical
choice; it is not the exact fixed-lambda free-energy weighting at the same time.

The normalization alpha is vehicle-specific and is evaluated in paper_alignment
experiments; using the paper's numerical value is not assumed to ensure performance.

## Remaining differences

Passive Bayesian model-noise sampling is off. Track-exit candidates are rejected,
rather than freezing their dynamics with continuing finite cost as in the paper.
The plant is the same learned mean model as the predictor. Track, speed scale,
vehicle geometry, basis functions and computing platform also differ.

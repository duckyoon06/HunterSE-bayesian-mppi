"""보정식의 밀도 비율 및 예측/실행 구동기 일치 검증: python -m unittest mppi.test_mppi"""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import norm

from .config import Config
from .controller import MPPIController, NoValidTrajectoryError
from .dynamic_adaptor import DynamicsAdaptor
from .simulator import CircularTrack, Simulator


class MPPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = Config(num_samples=32, horizon_steps=6, max_steps=2,
                            cost_normalization_alpha=10.)
        cls.dynamics = DynamicsAdaptor(cls.config)
        cls.track = CircularTrack()

    def controller(self):
        return MPPIController(self.config, self.dynamics, self.track.track_error)

    def test_cost_noise_diffusion_relation(self):
        cfg = self.config
        r = np.asarray(cfg.control_cost_diag)
        bc = np.asarray(cfg.base_actuator_diffusion_std)
        np.testing.assert_allclose(r, cfg.path_integral_lambda * cfg.actuator_rate**2 / bc**2)
        np.testing.assert_allclose(bc, [.01, .01])
        self.assertTrue(np.all(np.asarray(cfg.exploration_variance_ratio) >= 1))

    def test_correction_matches_gaussian_log_density_ratio(self):
        ctrl = self.controller()
        ctrl.nominal_controls[:] = [.2, .8]
        latent, noise = ctrl.sample_controls()
        cfg = self.config
        # 포화를 발생시키는 latent 표본에서도 unclipped 밀도 비율이어야 한다.
        self.assertTrue(np.any(latent[..., 1] > cfg.throttle_max))
        log_p = np.sum(norm.logpdf(latent, scale=np.sqrt(cfg.base_command_variance)), axis=(1, 2))
        log_q = np.sum(norm.logpdf(latent, loc=ctrl.nominal_controls,
                                  scale=cfg.control_noise_std), axis=(1, 2))
        density_cost = -cfg.path_integral_lambda * (log_p - log_q)
        energy = .5 * cfg.dt * np.sum(np.asarray(cfg.control_cost_diag) * ctrl.nominal_controls**2)
        implemented = energy + ctrl.importance_costs(noise)
        # 생략한 로그 행렬식은 후보 사이의 차를 구하면 사라진다.
        np.testing.assert_allclose(implemented - implemented[0], density_cost - density_cost[0],
                                   atol=1e-12)

    def test_actuator_euler_and_vehicle_timing(self):
        dyn, cfg = self.dynamics, self.config
        state = dyn.augment_state(self.track.initial_state(1), np.array([0., .3]))
        desired = np.array([.4, .7])
        nxt = dyn.step(state, desired)
        np.testing.assert_allclose(nxt[:7], dyn.model.step(state[:7], state[7:], cfg.dt))
        np.testing.assert_allclose(nxt[7:], state[7:] + cfg.actuator_rate * cfg.dt * (desired-state[7:]))
        initial = state[7:].copy()
        for _ in range(6):
            state = dyn.step(state, desired)
        np.testing.assert_allclose(state[7:], desired + (initial-desired)*(1-cfg.actuator_rate*cfg.dt)**6)

    def test_rollout_matches_simulator_and_batch(self):
        sim = Simulator(self.config, self.dynamics, self.track.track_error,
                        self.track.initial_state(1))
        controls = np.tile([.2, .35], (6, 1))
        trace = self.dynamics.rollout(sim.state, controls)
        batch = self.dynamics.rollout(sim.state, np.stack([controls, controls]))
        np.testing.assert_allclose(batch[0], trace, rtol=1e-12, atol=1e-12)
        for t in range(2):
            old_actuator = sim.state[7:9]
            result = sim.step(controls[t])
            np.testing.assert_array_equal(result.state, trace[t+1])
            np.testing.assert_array_equal(result.applied_control, old_actuator)
        self.assertEqual(result.reason, "time_limit")

    def test_saturation_in_dynamics_does_not_change_latent_noise(self):
        state = self.dynamics.augment_state(self.track.initial_state(1))
        np.testing.assert_array_equal(self.dynamics.step(state, [10., -10.]),
                                      self.dynamics.step(state, [self.config.steering_max, 0.]))

    def test_invalid_noise_or_actuator_discretization(self):
        with self.assertRaises(ValueError):
            replace(self.config, dt=.2)
        with self.assertRaises(ValueError):
            replace(self.config, steering_noise_std=1e-5)

    def test_normalization_and_no_valid_candidate(self):
        ctrl = self.controller()
        self.assertAlmostEqual(ctrl.weight_temperature(20., np.array([10.])), 2.)
        self.assertAlmostEqual(ctrl.weight_temperature(np.inf, np.array([np.inf, 10.])), 1.)
        weights = ctrl.compute_weights(np.array([1e12, 1e12+1, np.inf]), 1.)
        self.assertAlmostEqual(weights.sum(), 1.)
        self.assertEqual(weights[-1], 0.)
        with self.assertRaises(NoValidTrajectoryError):
            ctrl.compute_weights(np.array([np.inf, np.inf]))
        state = self.dynamics.augment_state(self.track.initial_state(1))
        state[4] = .1
        with self.assertRaises(NoValidTrajectoryError):
            ctrl.command(state)
        self.assertEqual(ctrl.last_iteration_diagnostics[-1]["summary"]["valid_count"], 0)

    def test_command_bounds_and_cost_decomposition(self):
        ctrl = self.controller()
        state = self.dynamics.augment_state(self.track.initial_state(1))
        command = ctrl.command(state)
        self.assertTrue(np.all(command >= self.config.control_min))
        self.assertTrue(np.all(command <= self.config.control_max))
        values = ctrl.last_iteration_diagnostics[-1]["candidates"]
        mask = values["valid"]
        np.testing.assert_allclose(values["total_cost"][mask],
                                   values["state_cost"][mask] + values["control_cost"][mask]
                                   + values["mean_shift_cost"][mask] + values["variance_change_cost"][mask])

    def test_fallback_bounds_unbounded_latent_samples(self):
        ctrl = self.controller()
        state = self.dynamics.augment_state(self.track.initial_state(1))
        candidates = np.full((self.config.num_samples, self.config.horizon_steps, 2), 2.)
        noise = candidates - ctrl.nominal_controls[None]
        costs = np.full(self.config.num_samples, np.inf)
        costs[0] = 0.
        with patch.object(ctrl, "sample_controls", return_value=(candidates, noise)), \
                patch.object(ctrl, "trajectory_costs", side_effect=[costs, np.array([0.]), np.array([np.inf])]):
            command = ctrl.command(state)
        self.assertTrue(ctrl.last_used_sample_fallback)
        np.testing.assert_array_equal(command, self.config.control_max)
        self.assertTrue(np.all(ctrl.last_control_sequence <= self.config.control_max))


if __name__ == "__main__":
    unittest.main()

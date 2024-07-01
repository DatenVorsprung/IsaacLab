"""
cartpole_decoupled_eval.py

Cartpole environment with decoupled cart & pole for evaluation with sb3
"""

"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import sys
from typing import Optional, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.integrate import odeint


class CartpoleDecoupledEvalEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    Environment simulation for Beckhoff Cartpole system.
    In this environment, we assume the pole and cart to be decoupled, because of the active
    control of the cart in the XTS Linear system.
    """

    def __init__(self):

        # system parameters
        self.gravity = 9.8               # m/s^2
        self.xacc_mag = 2                # m/s^2
        self.tau = 0.01                  # time step in s
        self.max_jerk = 23               # m/s^3
        self.pole_friction = 2.1e-3      # kg m² / s²
        self.pole_length = 0.41           # half pole length in m
        self.pole_mass = 0.08            # kg
        self.cart_mass = 0.46            # kg
        self.momentum_inertia = 1.05e-2     # kg m^2

        self.x_threshold = 0.4           # maximum position m
        self.x_dot_dot = 0               # acceleration m/s^2
        self.max_velocity = 1.1          # m/s
        self.start_theta = np.pi

        high = np.array([self.x_threshold, 1.1, 1., 1., 5.], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(), dtype=float)
        self.state = None

        self.steps_beyond_terminated = None

    def dynamics(self, state, t, *args):
        x, x_dot, theta, theta_dot = state

        xacc = args[0].item()

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        thetaacc = ((self.pole_mass * self.pole_length * (
                    self.gravity * sintheta - xacc * costheta) - self.pole_friction * theta_dot) /
                    self.momentum_inertia)

        return np.array([x_dot, xacc, theta_dot, thetaacc])

    def integrate(self, acceleration):
        """ Integrate one step with RK method"""

        new_state = odeint(self.dynamics, self.state, np.array([0., self.tau]),
                           (acceleration,))
        return new_state[-1]

    def step(self, action):

        acceleration = action * self.xacc_mag
        new_state = self.integrate(acceleration)

        self.state = new_state
        terminated = self.is_terminated(self.state)
        reward = self.get_reward(self.state, action)

        obs = self.get_obs()
        return obs, reward, terminated, False, {}

    def get_obs(self):
        return np.array([self.state[0], self.state[1], np.sin(self.state[2]), np.cos(self.state[2]), self.state[3]],
                        dtype=np.float32)

    def get_reward(self, state, action):
        x, _, theta, theta_dot = state
        reward = np.sum([
            1 + np.cos(theta),
            -0.01 * np.abs(theta_dot),
            -0.01 * np.abs(x),
        ])
        return reward

    def is_terminated(self, state):
        return state[0] < -self.x_threshold or state[0] > self.x_threshold

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        start_theta = self.np_random.uniform(low=-0.2, high=0.2)
        start_x = self.np_random.uniform(low=-0.1, high=0.1)
        start_x_dot = self.np_random.uniform(low=-0.05, high=0.05)
        start_theta_dot = self.np_random.uniform(low=-0.05, high=0.05)
        self.state = np.array([start_x, start_x_dot, self.start_theta, start_theta_dot],
                              dtype=np.float32)
        self.steps_beyond_terminated = None
        obs = self.get_obs()

        return obs, {}

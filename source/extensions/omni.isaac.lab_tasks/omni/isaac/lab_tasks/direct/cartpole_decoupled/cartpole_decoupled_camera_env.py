# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import math
import numpy as np
import torch
import torchvision.transforms
from torchdiffeq import odeint
from collections.abc import Sequence
from collections import deque

from omni.isaac.lab_assets.cartpole_decoupled import CARTPOLE_DECOUPLED_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg, save_images_to_file, Camera, CameraCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform

from omni.isaac.lab_tasks.direct.cartpole_decoupled.randomizer_cartpole_decoupled import CartPoleDecoupledRandomizer


@configclass
class CartpoleDecoupledRGBCameraSimConfig(SimulationCfg):

    dt = 0.01  # 10 ms
    gravity = [0.0, 0.0, -9.81]
    render_interval = 1
    use_fabric = True
    enable_scene_query_support = False
    disable_contact_processing = False
    use_gpu_pipeline = True


@configclass
class CartpoleDecoupledRGBCameraEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0 # [s]
    max_accel = 2  # [N]
    decimation = 1  # number of control steps / time step
    num_actions = 1
    num_observations = 5
    num_states = 0
    num_channels = 3
    frame_stack = 1

    # reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005


    # simulation
    sim: SimulationCfg = CartpoleDecoupledRGBCameraSimConfig()

    # robot
    robot_cfg: ArticulationCfg = CARTPOLE_DECOUPLED_CFG.replace(prim_path="/World/envs/env_.*/Cartpole")
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"

    # camera
    tiled_camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=CameraCfg.OffsetCfg(pos=(2.0, 0.5, 2.5), rot=( 0.5, 0.5, 0.5, 0.5), convention="opengl"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1e5)
        ),
        width=300,
        height=200,
    )
    num_observations = num_channels * tiled_camera.height * tiled_camera.width
    write_image_to_file = False

    # change viewer settings
    viewer = ViewerCfg(eye=(20.0, 20.0, 20.0))

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=20.0, replicate_physics=True)

    # reset
    # reset
    max_cart_pos = 0.35  # the cart is reset if it exceeds that position [m]
    max_pole_angle = 0.4
    initial_pole_angle_range = [-0.2, 0.2]  # the range (in multiples of pi) in which the pole angle is sampled from on reset [rad]


class CartpoleDecoupledCameraEnv(DirectRLEnv):

    cfg: CartpoleDecoupledRGBCameraEnvCfg

    def __init__(
        self, cfg: CartpoleDecoupledRGBCameraEnvCfg, render_mode: str | None = None, randomize: bool = False, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self._cart_dof_idx, _ = self.cartpole.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole_dof_name)
        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

        self.state_buf = None
        self.cart_mass = 0.46 * torch.ones(self.num_envs, device=self.device)
        self.pole_len = 0.41 * torch.ones(self.num_envs, device=self.device)
        self.pole_mass = 0.08 * torch.ones(self.num_envs, device=self.device)
        self.pole_friction = 2.1e-3 * torch.ones(self.num_envs, device=self.device)
        self.moment_of_inertia = 0.0105 * torch.ones(self.num_envs, device=self.device)
        self.gravity = 9.81 * torch.ones(self.num_envs, device=self.device)
        self.max_accel = self.cfg.max_accel
        self.time_steps = torch.zeros(self.num_envs, device=self.device)
        self.obs_buf = {'policy': deque(maxlen=self.cfg.frame_stack)}

        self._custom_randomizer = CartPoleDecoupledRandomizer(active=randomize)

        if self._custom_randomizer.attribute_randomize:
            self._custom_randomizer.attribute_randomizer(self)

        if len(self.cfg.tiled_camera.data_types) != 1:
            raise ValueError(
                "The CartpoleDecoupled camera environment only supports one image type at a time but the following were"
                f" provided: {self.cfg.tiled_camera.data_types}"
            )

    def close(self):
        """Cleanup for the environment."""
        super().close()

    def cartpole_decoupled_dynamics(self, t, state, action):
        """ Dynamics equations of decoupled cartpole system """

        accel = self.max_accel * action.squeeze(1)

        x_dot = state[:, 1]
        x_dot_dot = accel
        theta_dot = state[:, 3]
        sin_theta = torch.sin(state[:, 2])
        cos_theta = torch.cos(state[:, 2])
        theta_acc = ((self.pole_mass * self.pole_len * (self.gravity * sin_theta - accel * cos_theta) - self.pole_friction * state[:, 3]) /
                      self.moment_of_inertia)
        return torch.hstack([x_dot.unsqueeze(1), x_dot_dot.unsqueeze(1), theta_dot.unsqueeze(1), theta_acc.unsqueeze(1)])

    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""
        # observation space (unbounded since we don't impose any limits)
        self.num_actions = self.cfg.num_actions
        self.num_observations = self.cfg.num_observations
        self.num_states = self.cfg.num_states

        # set up spaces
        self.single_observation_space = gym.spaces.Dict()
        self.single_observation_space["policy"] = gym.spaces.Box(
            low=0.,
            high=1.,
            shape=(self.cfg.frame_stack * 64,),
        )
        if self.num_states > 0:
            self.single_observation_space["critic"] = gym.spaces.Box(
                low=0.,
                high=1.,
                shape=(self.cfg.frame_stack * 64,),
            )
        self.single_action_space = gym.spaces.Box(low=-1, high=1, shape=(self.num_actions,))

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        # RL specifics
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.sim.device)

    def _setup_scene(self):
        """Setup the scene with the cartpole and camera."""
        self.cartpole = Articulation(self.cfg.robot_cfg)
        self._tiled_camera = Camera(self.cfg.tiled_camera)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(size=(500, 500)))
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        # add articultion and sensors to scene
        self.scene.articulations["cartpole"] = self.cartpole
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        # get the current state and store it in state buffer
        states = torch.cat(
            (
                self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        self.state_buf = states

    def _get_guide_model_obs(self):
        state = torch.cat(
            (
                self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                torch.sin(self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1)),
                torch.cos(self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1)),
                self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        return state

    def step(self, action: torch.Tensor):
        """ Overridden step method; used for applying decoupled physics """

        # add action noise
        if self._custom_randomizer.action_randomize:
            action = self._custom_randomizer.action_randomizer(self, action.clone())

        # store actions and current state in buffer
        self._pre_physics_step(action)

        # calculate the new states with the decoupled dynamics
        # create tensor with timesteps
        ts = torch.tensor([0., self.physics_dt], dtype=torch.float32, device=self.device)
        # integrate the dynamics
        sol = odeint(lambda t, state: self.cartpole_decoupled_dynamics(t, state, action), self.state_buf, ts)
        # get next state from solution
        new_states = sol[-1, :, :]

        # set the new states
        env_ids = self.cartpole._ALL_INDICES
        joint_pos = new_states[:, [0, 2]]
        joint_vel = new_states[:, [1, 3]]

        # step the simulation
        # this is needed in order for the simulation to work at all, the correct states are overwritten later on
        self._sim_step_counter += 1
        self.sim.step(render=True)

        # get the next state according to the ODE
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
        self.cartpole.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # render without stepping to update the simulation with the correct state
        self.sim.render()
        self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # get observation and put it into obs_buf
        obs_buf = self._get_observations()

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        # the noise will only be added to the observation that entered the obs_buf last
        if self._custom_randomizer.observation_randomize:
            obs = obs_buf['policy'][-1]
            obs = self._custom_randomizer.observation_randomizer(self, obs)
            obs_buf['policy'][-1] = obs

        self.time_steps += 1

        # return observations, rewards, resets and extras
        return obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _apply_action(self) -> None:
        self.cartpole.set_joint_effort_target(self.actions, joint_ids=self._cart_dof_idx)

    def _get_observations(self) -> dict:
        img = self._tiled_camera.data.output['rgb'].clone()
        img = img[:, :, :, :3].permute(0, 3, 1, 2)
        img = torchvision.transforms.Resize((80, 80), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)(img) / 255.
        self.obs_buf['policy'].append(torch.rand(self.num_envs, 64))
        return {'policy': torch.cat(list(self.obs_buf['policy']), dim=1)}

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pole_pos,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_pole_vel,
            self.joint_pos[:, self._pole_dof_idx[0]],
            self.joint_vel[:, self._pole_dof_idx[0]],
            self.joint_pos[:, self._cart_dof_idx[0]],
            self.joint_vel[:, self._cart_dof_idx[0]],
            self.reset_terminated,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > self.cfg.max_pole_angle, dim=1)
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.cartpole._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.cartpole.data.default_joint_pos[env_ids]
        # set cart to [-0.1, 0.1]
        joint_pos[:, self._cart_dof_idx] += sample_uniform(-0.1, 0.1, joint_pos[:, self._cart_dof_idx].shape, joint_pos.device)
        # set angle to [-pi, pi]
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0],
            self.cfg.initial_pole_angle_range[1],
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self.cartpole.data.default_joint_vel[env_ids]
        # set cart and pole velocity to some small random value
        joint_vel[:, self._cart_dof_idx] += sample_uniform(-0.05, 0.05, joint_vel[:, self._cart_dof_idx].shape, joint_vel.device)
        joint_vel[:, self._pole_dof_idx] += sample_uniform(-0.05, 0.05, joint_vel[:, self._cart_dof_idx].shape, joint_vel.device)

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
        self.cartpole.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.state_buf = torch.cat(
            (
                self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
            ),
            dim=-1,
        )

        reset_time_steps = torch.ones(self.num_envs, device=self.device)
        reset_time_steps[env_ids] = 0.
        # fill obs_buf
        for _ in range(self.cfg.frame_stack):
            self._get_observations()
        self.time_steps = self.time_steps * reset_time_steps


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    return total_reward

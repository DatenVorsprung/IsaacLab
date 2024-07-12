# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Cartpole balancing environment with decoupled pole and cart
"""

import gymnasium as gym

from . import agents
from .cartpole_decoupled_camera_env import CartpoleDecoupledCameraEnv, CartpoleDecoupledRGBCameraEnvCfg
from .cartpole_decoupled_env import CartpoleDecoupledEnv, CartpoleDecoupledEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-CartpoleDecoupled-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.cartpole_decoupled:CartpoleDecoupledEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CartpoleDecoupledEnvCfg,
        "rl_games_ppo_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rl_games_sac_cfg_entry_point": f"{agents.__name__}:rl_games_sac_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.CartpoleDecoupledPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_ppo_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sb3_sac_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
    },
)

gym.register(
    id="Isaac-CartpoleDecoupled-RGB-Camera-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.cartpole_decoupled:CartpoleDecoupledCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CartpoleDecoupledRGBCameraEnvCfg,
        "rl_games_ppo_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
        "rl_games_sac_cfg_entry_point": f"{agents.__name__}:rl_games_camera_sac_cfg.yaml",
        "sb3_sac_cfg_entry_point": f"{agents.__name__}:sb3_sac_video_cfg.yaml",
        "sb3_ppo_cfg_entry_point": f"{agents.__name__}:sb3_ppo_video_cfg.yaml",
    },
)

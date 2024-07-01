"""
cartpole_decoupled_eval

Evaluation environment for decoupled cartpole system
"""

import gymnasium as gym
from .cartpole_decoupled_eval_env import CartpoleDecoupledEvalEnv


gym.register('CartpoleDecoupledEval-v0',
             entry_point="omni.isaac.lab_tasks.direct.cartpole_decoupled_eval:CartpoleDecoupledEvalEnv",
)

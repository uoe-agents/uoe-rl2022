"""
Those are tests that will be shared with students
They should test that the code structure/return values
are of correct type/shape
"""

import pytest
import gym
import os.path
import numpy as np

def test_imports():
    from rl2022.exercise4 import DDPG
    from rl2022.exercise4.train_ddpg import BIPEDAL_CONFIG as CONFIG

def test_config():
    from rl2022.exercise4.train_ddpg import BIPEDAL_CONFIG
    assert "episode_length" in BIPEDAL_CONFIG
    assert "max_timesteps" in BIPEDAL_CONFIG
    assert "eval_freq" in BIPEDAL_CONFIG
    assert "eval_episodes" in BIPEDAL_CONFIG
    assert "policy_learning_rate" in BIPEDAL_CONFIG
    assert "critic_learning_rate" in BIPEDAL_CONFIG
    assert "policy_hidden_size" in BIPEDAL_CONFIG
    assert "critic_hidden_size" in BIPEDAL_CONFIG
    assert "tau" in BIPEDAL_CONFIG
    assert "batch_size" in BIPEDAL_CONFIG
    assert "gamma" in BIPEDAL_CONFIG
    assert "buffer_capacity" in BIPEDAL_CONFIG
    assert "save_filename" in BIPEDAL_CONFIG

def test_restore_file():
    from rl2022.exercise4 import DDPG
    from rl2022.exercise4.train_ddpg import BIPEDAL_CONFIG
    env = gym.make("BipedalWalker-v3")
    agent = DDPG(
        action_space=env.action_space,
        observation_space=env.observation_space,
        **BIPEDAL_CONFIG
    )
    agent.restore("bipedal_latest.pt")

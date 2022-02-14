EX1_CONSTANTS = {
    "gamma": 0.9,
}

EX2_CONSTANTS = {
    "env": "Taxi-v3",
    "gamma": 0.99,
    "eps_max_steps": 50,
    "eval_eps_max_steps": 100,
}

EX2_MC_CONSTANTS = EX2_CONSTANTS.copy()
EX2_MC_CONSTANTS["total_eps"] = 100000

EX2_QL_CONSTANTS = EX2_CONSTANTS.copy()
EX2_QL_CONSTANTS["total_eps"] = 10000

EX3_CARTPOLE_CONSTANTS = {
    "env": "CartPole-v1",
    "gamma": 0.99,
    "episode_length": 200,
    "max_time": 30 * 60,
    "save_filename": None,
}

EX3_DQN_CARTPOLE_CONSTANTS = EX3_CARTPOLE_CONSTANTS.copy()
EX3_DQN_CARTPOLE_CONSTANTS["max_timesteps"] = 20000

EX3_REINFORCE_CARTPOLE_CONSTANTS = EX3_CARTPOLE_CONSTANTS.copy()
EX3_REINFORCE_CARTPOLE_CONSTANTS["max_timesteps"] = 200000

EX3_LUNARLANDER_CONSTANTS = {
    "env": "LunarLander-v2",
    "gamma": 0.99,
    "max_timesteps": 300000,
    "episode_length": 500,
    "max_time": 120 * 60,
    "save_filename": "dqn_lunarlander_latest.pt",
}

EX4_PENDULUM_CONSTANTS = {
    "env": "Pendulum-v1",
    "target_return": -300.0,
    "episode_length": 200,
    "max_timesteps": 400000,
    "max_time": 120 * 60,
    "gamma": 0.99,
    "save_filename": "pendulum_latest.pt",
}

EX4_BIPEDAL_CONSTANTS = {
    "env": "BipedalWalker-v3",
    "target_return": 300.0,
    "episode_length": 1600,
    "max_timesteps": 400000,
    "max_time": 120 * 60,
    "gamma": 0.99,
    "save_filename": "bipedal_latest.pt",

}

EX5_PENALTY_CONSTANTS = {
    "env": "penalty",
    "env_args": (-15, 5),
    "total_eps": 20000,
    "gamma": 0.99,
}

EX5_CLIMBING_CONSTANTS = {
    "env": "climbing",
    "env_args": (5,),
    "total_eps": 10000,
    "gamma": 0.99,
}

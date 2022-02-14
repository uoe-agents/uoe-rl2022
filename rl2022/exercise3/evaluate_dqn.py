import gym
from typing import List, Tuple

from rl2022.exercise3.agents import DQN
from rl2022.exercise3.train_dqn import LUNARLANDER_CONFIG, play_episode


RENDER = True
CONFIG = LUNARLANDER_CONFIG


def evaluate(env: gym.Env, config, output: bool = True) -> Tuple[List[float], List[float]]:
    """
    Execute training of DDPG on given environment using the provided configuration

    :param env (gym.Env): environment to train on
    :param config: configuration dictionary mapping configuration keys to values
    :param output (bool): flag whether evaluation results should be printed
    :return (Tuple[List[float], List[float]]): eval returns during training, times of evaluation
    """
    timesteps_elapsed = 0

    agent = DQN(
        action_space=env.action_space, observation_space=env.observation_space, **config
    )
    # try:
    agent.restore(config['save_filename'])
    # except:
    #     raise ValueError(f"Could not find model to load at {config['save_filename']}")

    eval_returns_all = []
    eval_times_all = []


    eval_returns = 0
    for _ in range(config["eval_episodes"]):
        _, episode_return, _ = play_episode(
            env,
            agent,
            0,
            train=False,
            explore=False,
            render=RENDER,
            max_steps=config["episode_length"],
            batch_size=config["batch_size"],
        )
        eval_returns += episode_return / config["eval_episodes"]

    return eval_returns


if __name__ == "__main__":
    env = gym.make(CONFIG["env"])
    returns = evaluate(env, CONFIG)
    print(returns)
    env.close()

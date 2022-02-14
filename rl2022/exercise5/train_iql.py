import copy

import gym

from rl2022.constants import EX5_PENALTY_CONSTANTS as PENALTY_CONSTANTS
from rl2022.constants import EX5_CLIMBING_CONSTANTS as CLIMBING_CONSTANTS
from rl2022.exercise5.agents import IndependentQLearningAgents
from rl2022.exercise5.utils import visualise_q_table, evaluate, visualise_q_convergence
from rl2022.exercise5.matrix_game import create_penalty_game, create_climbing_game


PEN_CONFIG = {
    "eval_freq": 100,
    "lr": 0.05,
    "epsilon": 0.9,
}
PEN_CONFIG.update(PENALTY_CONSTANTS)

CLIMBING_CONFIG = {
    "eval_freq": 100,
    "lr": 0.05,
    "epsilon": 0.9,
}
CLIMBING_CONFIG.update(CLIMBING_CONSTANTS)

CONFIG = PEN_CONFIG
# CONFIG = CLIMBING_CONFIG


def iql_eval(env, config, q_tables, max_steps=10, eval_episodes=500, render=False, output=True):
    """
    Evaluate configuration of independent Q-learning on given environment when initialised with given Q-table

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :param q_tables (List[Dict[Act, float]]): Q-tables mapping actions to Q-values for each agent
    :param max_steps (int): number of steps per evaluation episode
    :param eval_episodes (int): number of evaluation episodes
    :param render (bool): flag whether evaluation runs should be rendered
    :param output (bool): flag whether mean evaluation performance should be printed
    :return (float, float): mean and standard deviation of returns received over episodes
    """
    eval_agents = IndependentQLearningAgents(
            num_agents=env.n_agents,
            action_spaces=env.action_space,
            gamma=config["gamma"],
            learning_rate=config["lr"],
            epsilon=0.0,
        )
    eval_agents.q_tables = q_tables
    return evaluate(env, eval_agents, max_steps, eval_episodes, render, output)


def train(env, config, output=True):
    """
    Train and evaluate independent Q-learning on given environment with provided hyperparameters

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :param output (bool): flag if mean evaluation results should be printed
    :return (float, List[float], List[float], Dict[Act, float]):
        total reward over all episodes, list of means and standard deviations of evaluation
        returns, final Q-table
    """
    agents = IndependentQLearningAgents(
            num_agents=env.n_agents,
            action_spaces=env.action_space,
            gamma=config["gamma"],
            learning_rate=config["lr"],
            epsilon=config["epsilon"],
        )

    step_counter = 0
    max_steps = config["total_eps"] * env.ep_length

    total_reward = 0
    evaluation_return_means = []
    evaluation_return_stds = []
    evaluation_q_tables = []

    for eps_num in range(config["total_eps"]):
        env.reset()
        episodic_return = 0
        t = 0

        while t < env.ep_length:
            agents.schedule_hyperparameters(step_counter, max_steps)
            acts = agents.act()
            _, rewards, dones, _ = env.step(acts)
            agents.learn(acts, rewards, dones)

            t += 1
            step_counter += 1
            # fully cooperative tasks --> only track single reward / all rewards are identical
            episodic_return += rewards[0]

            if all(dones):
                break

        total_reward += episodic_return

        if eps_num > 0 and eps_num % config["eval_freq"] == 0:
            mean_return, std_return = iql_eval(
                env, config, agents.q_tables, output=output
            )
            evaluation_return_means.append(mean_return)
            evaluation_return_stds.append(std_return)
            evaluation_q_tables.append(copy.deepcopy(agents.q_tables))

    return total_reward, evaluation_return_means, evaluation_return_stds, evaluation_q_tables, agents.q_tables


if __name__ == "__main__":
    if CONFIG["env"] == "penalty":
        env = create_penalty_game(*CONFIG["env_args"])
    elif CONFIG["env"] == "climbing":
        env = create_climbing_game(*CONFIG["env_args"])
    else:
        raise ValueError(f"Unsupported environment {CONFIG['env']}!")

    total_reward, _, _, evaluation_q_tables, q_tables = train(env, CONFIG)
    print()
    print("Q-table:")
    # print(q_tables)
    for i, q_table in enumerate(q_tables):
        visualise_q_table(env, q_table, i)

    for i in range(env.n_agents):
        eval_q_tables = [q_tabs[i] for q_tabs in evaluation_q_tables]
        visualise_q_convergence(i, eval_q_tables, env)

import random

import numpy as np

from rl2022.exercise5.agents import IndependentQLearningAgents, JointActionLearning
from rl2022.exercise5.utils import visualise_both_q_convergence
from rl2022.exercise5.matrix_game import create_penalty_game, create_climbing_game
from rl2022.exercise5.train_iql import PEN_CONFIG as IQL_PEN_CONFIG
from rl2022.exercise5.train_iql import CLIMBING_CONFIG as IQL_CLIMBING_CONFIG
from rl2022.exercise5.train_iql import train as iql_train
from rl2022.exercise5.train_jal import PEN_CONFIG as JAL_PEN_CONFIG
from rl2022.exercise5.train_jal import CLIMBING_CONFIG as JAL_CLIMBING_CONFIG
from rl2022.exercise5.train_jal import train as jal_train


GAME = "penalty" # "climbing" or "penalty"
SEEDS = 5
SAVEFIG = None # give string to save file with generated plots under given name


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    if GAME == "penalty":
        IQL_CONFIG = IQL_PEN_CONFIG
        JAL_CONFIG = JAL_PEN_CONFIG
        num_actions = 3
        # use environment arguments but fix episode length at 1 for stable plots
        env_args = list(IQL_CONFIG["env_args"])
        env_args[1] = 1
        env = create_penalty_game(*env_args)
    elif GAME == "climbing":
        IQL_CONFIG = IQL_CLIMBING_CONFIG
        JAL_CONFIG = JAL_CLIMBING_CONFIG
        num_actions = 3
        # use environment arguments but fix episode length at 1 for stable plots
        env_args = list(IQL_CONFIG["env_args"])
        env_args[0] = 1
        env = create_climbing_game(*env_args)
    else:
        raise ValueError(f"Unsupported environment {GAME}!")

    print("Train IQL ...")
    iql_total_eval_q_tables = []
    for seed in range(SEEDS):
        print(f"\tStart training with seed={seed}")
        set_seed(seed)
        _, _, _, all_eval_q_tables, _ = iql_train(env, IQL_CONFIG, output=False)
        # Individual Q-table shape: num_agents x num_eval x num_actions
        all_eval_q_tables = np.array([[[q_table[i] for i in range(num_actions)] for q_table in eval_q_tables] for eval_q_tables in all_eval_q_tables]).swapaxes(0, 1)

        iql_total_eval_q_tables.append(all_eval_q_tables)
    iql_total_eval_q_tables = np.array(iql_total_eval_q_tables)

    print()
    print("Train JAL ...")
    jal_total_eval_q_tables = []
    for seed in range(SEEDS):
        print(f"\tStart training with seed={seed}")
        set_seed(seed)
        _, _, all_eval_q_tables, _ = jal_train(env, JAL_CONFIG, output=False)
        # Joint Q-table shape: num_agents x num_eval x num_actions x num_actions
        all_eval_q_tables = np.array([[[[joint_q_table[(a1, a2)] for a2 in range(num_actions)] for a1 in range(num_actions)]for joint_q_table in eval_q_tables] for eval_q_tables in all_eval_q_tables]).swapaxes(0, 1)

        jal_total_eval_q_tables.append(all_eval_q_tables)
    jal_total_eval_q_tables = np.array(jal_total_eval_q_tables)

    visualise_both_q_convergence(iql_total_eval_q_tables, jal_total_eval_q_tables, env)

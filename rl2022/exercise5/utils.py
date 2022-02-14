import time

import matplotlib.pyplot as plt
import numpy as np

FIG_WIDTH=5
FIG_HEIGHT=2
FIG_ALPHA=0.2
FIG_WSPACE=0.3
FIG_HSPACE=0.2


def visualise_q_table(env, q_table, i):
    assert hasattr(env, "payoff")
    assert hasattr(env, "ep_length")
    ep_length = env.ep_length
    for a1 in range(3):
        q_value = q_table[a1]
        actual_value = 0
        if i == 0:
            print(env.payoff[a1] * ep_length)
        else:
             print(list(np.array(env.payoff).T[a1] * ep_length))
        for a2 in range(3):
            if i == 0:
                actual_value += env.payoff[a1][a2] * ep_length
            else:
                 actual_value += env.payoff[a2][a1] * ep_length      
        # expectation over all three values
        actual_value /= 3
        print(f"Q({a1 + 1}) = {q_value:.2f}\t\tActual Value: {actual_value}")
    print()


def visualise_joint_q_table(env, q_table, i):
    assert hasattr(env, "payoff")
    assert hasattr(env, "ep_length")
    for a1 in range(3):
        for a2 in range(3):
            q_value = q_table[(a1, a2)]
            actual_value = env.payoff[a1][a2] * env.ep_length
            print(f"Q(a_{a1}, b_{a2}) = {q_value:.2f}\t\tActual Value: {actual_value}")
    print()


def evaluate(env, agents, max_steps, eval_episodes, render, output=True):
    """
    Evaluate configuration on given environment when initialised with given Q-table

    :param env (gym.Env): environment to execute evaluation on
    :param agents (MultiAgent): agent to act in environment
    :param max_steps (int): max number of steps per evaluation episode
    :param eval_episodes (int): number of evaluation episodes
    :param render (bool): flag whether evaluation runs should be rendered
    :param output (bool): flag whether mean evaluation results should be printed
    :return (float, float): mean and standard deviation of returns received over episodes
    """
    episodic_returns = []
    for eps_num in range(eval_episodes):
        env.reset()
        if (eps_num == eval_episodes - 1) and render:
            env.render()
            time.sleep(0.5)
        episodic_return = 0
        dones = [False] * agents.num_agents
        steps = 0

        while not all(dones) and steps < max_steps:
            acts = agents.act()
            _, rewards, dones, _ = env.step(acts)
            if (eps_num == eval_episodes - 1) and render:
                env.render()
                time.sleep(0.5)

            episodic_return += rewards[0]
            steps += 1

        episodic_returns.append(episodic_return)

    mean_return = np.mean(episodic_returns)
    std_return = np.std(episodic_returns)

    if output:
        # print(f"EVALUATION: EPISODIC RETURNS: {episodic_returns}")
        print(f"EVALUATION: MEAN RETURN OF {mean_return}")
    return mean_return, std_return


def visualise_q_convergence(player_index, q_tables, env, title=None, savefig=None):
    """
    Plot q_table convergence
    :param player_index (int): player index (either 0 or 1)
    :param q_tables (List[Dict[Act, float]]): q_tables for each evaluation
    :param env (gym.Env): gym matrix environment with `payoff` attribute
    :param title (str): title for figure
    """
    assert hasattr(env, "payoff")
    assert hasattr(env, "ep_length")
    payoff = np.array(env.payoff)
    ep_length = env.ep_length
    num_actions = len(q_tables[0].keys())
    q_tables = np.array([[q_table[i] for i in range(num_actions)] for q_table in q_tables])

    fig, ax = plt.subplots(nrows=1, ncols=num_actions, figsize=(num_actions * FIG_WIDTH, FIG_HEIGHT))

    max_payoff = payoff.max() * ep_length
    min_payoff = payoff.min() * ep_length
    
    for act in range(num_actions):
        # plot max Q-values
        if player_index == 0:
            max_q = payoff[act, :].max() * ep_length
            max_label = rf"$max_b Q(a, b)$"
            q_label = rf"$Q(a_{act}, \cdot)$"
        else:
            max_q = payoff[:, act].max() * ep_length
            max_label = rf"$max_a Q(a, b_{act})$"
            q_label = rf"$Q(\cdot, b_{act})$"
        ax[act].axhline(max_q, ls='--', color='r', alpha=0.5, label=max_label)

        # plot respective Q-values
        q_values = q_tables[:, act]
        ax[act].plot(q_values, label=q_label)

        # axes labels and limits
        ax[act].set_ylim([min_payoff - ep_length * 0.5, max_payoff + ep_length * 0.5])
        ax[act].set_xlabel(f"Evaluations")
        if player_index == 0:
            ax[act].set_ylabel(fr"$Q(a_{act})$")
        else:
            ax[act].set_ylabel(fr"$Q(b_{act})$")

        ax[act].legend(loc="lower center")

    fig.subplots_adjust(wspace=FIG_WSPACE)

    if title is not None:
        fig.title(title)

    if savefig is not None:
        plt.savefig(f"{savefig}_{player_index+1}.pdf", format="pdf")

    plt.show()


def visualise_joint_q_convergence(q_tables, env, title=None, savefig=None):
    """
    Plot joint q_table convergence
    :param q_tables (List[Dict[Act, float]]): q_tables for each evaluation
    :param env (gym.Env): gym matrix environment with `payoff` attribute
    :param title (str): title for figure
    :aram savefig (str): name of figure to save under (or None if not saved)
    """
    assert hasattr(env, "payoff")
    assert hasattr(env, "ep_length")
    payoff = np.array(env.payoff)
    ep_length = env.ep_length
    n_agents = env.n_agents
    num_actions = payoff.shape[0]
    assert n_agents == len(payoff.shape) == 2
    assert payoff.shape[0] == payoff.shape[1] == num_actions

    max_payoff = payoff.max() * ep_length
    min_payoff = payoff.min() * ep_length

    fig, ax = plt.subplots(nrows=num_actions, ncols=num_actions, figsize=(num_actions * FIG_WIDTH, num_actions * FIG_HEIGHT))
    for act0 in range(num_actions):
        for act1 in range(num_actions):
            # plot respective Q-values
            q_values = [q_table[(act0, act1)] for q_table in q_tables]
            q_label = rf"$Q(a, b)$"
            ax[act0, act1].plot(q_values, label=q_label)

            true_q = payoff[act0, act1] * ep_length
            true_label = rf"True $Q(a, b)$"
            ax[act0, act1].axhline(true_q, ls='--', color='k', alpha=0.5, label=true_label)

            ax[act0, act1].set_ylim([min_payoff - ep_length * 0.5, max_payoff + ep_length * 0.5])

            if act0 == num_actions - 1:
                ax[act0, act1].set_xlabel(f"Evaluations")
            ax[act0, act1].set_ylabel(fr"$Q(a_{act0}, b_{act1})$")

    fig.subplots_adjust(wspace=FIG_WSPACE, hspace=FIG_HSPACE)
    # global legend
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=14)

    if title is not None:
        fig.title(title)

    if savefig is not None:
        plt.savefig(f"{savefig}.pdf", format="pdf")

    plt.show()


def visualise_both_q_convergence(iql_q_tables, jal_q_tables, env, savefig=None):
    """
    Plot independent and joint q_table convergence
    :param iql_q_tables (np.array of shape num_seeds x num_agents x num_eval x num_actions): independent
        q_tables for each evaluation
    :param jal_q_tables (np.array of shape num_seeds x num_agents x num_eval x num_actions x num_actions):
        joint q tables for each evaluation
    :param env (gym.Env): gym matrix environment with `payoff` attribute
    :aram savefig (str): name of figure to save under (or None if not saved)
    """
    assert hasattr(env, "payoff")
    assert hasattr(env, "ep_length")
    payoff = np.array(env.payoff)
    ep_length = env.ep_length
    n_agents = env.n_agents
    num_actions = payoff.shape[0]
    assert n_agents == len(payoff.shape) == 2
    assert payoff.shape[0] == payoff.shape[1] == num_actions
    num_evals = jal_q_tables.shape[2]
    assert num_evals == iql_q_tables.shape[2]
    
    x = np.arange(num_evals)

    max_payoff = payoff.max() * ep_length
    min_payoff = payoff.min() * ep_length

    for agent in range(n_agents):
        fig, ax = plt.subplots(nrows=num_actions, ncols=num_actions, figsize=(num_actions * FIG_WIDTH, num_actions * FIG_HEIGHT))
        for act0 in range(num_actions):
            for act1 in range(num_actions):
                # plot JAL Q-value
                jal_agent_q_tables = jal_q_tables[:, agent, :, act0, act1]
                jal_agent_q_tables_mean = jal_agent_q_tables.mean(axis=0)
                jal_agent_q_tables_std = jal_agent_q_tables.std(axis=0)
                ax[act0, act1].plot(jal_agent_q_tables_mean, label="JAL")
                ax[act0, act1].fill_between(
                    x,
                    jal_agent_q_tables_mean - jal_agent_q_tables_std,
                    jal_agent_q_tables_mean + jal_agent_q_tables_std,
                    alpha=FIG_ALPHA,
                )

                # plot IQL Q-value
                if agent == 0:
                    iql_agent_q_tables = iql_q_tables[:, agent, :, act0]
                else:
                    iql_agent_q_tables = iql_q_tables[:, agent, :, act1]
                iql_agent_q_tables_mean = iql_agent_q_tables.mean(axis=0)
                iql_agent_q_tables_std = iql_agent_q_tables.std(axis=0)
                ax[act0, act1].plot(iql_agent_q_tables_mean, label="IQL")
                ax[act0, act1].fill_between(
                    x,
                    iql_agent_q_tables_mean - iql_agent_q_tables_std,
                    iql_agent_q_tables_mean + iql_agent_q_tables_std,
                    alpha=FIG_ALPHA,
                )

                # plot true joint Q-value
                true_q = payoff[act0, act1] * ep_length
                ax[act0, act1].axhline(true_q, ls='--', color='k', alpha=0.5, label="True Q")

                ax[act0, act1].set_ylim([min_payoff - ep_length * 0.5, max_payoff + ep_length * 0.5])
                if act0 == num_actions - 1:
                    # x-label only in last row
                    ax[act0, act1].set_xlabel(f"Evaluations", fontsize=14)
                ax[act0, act1].set_ylabel(fr"$Q(a_{act0}, b_{act1})$", fontsize=14)

        fig.subplots_adjust(wspace=FIG_WSPACE, hspace=FIG_HSPACE)
        # global legend
        handles, labels = ax[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=14)

        plt.suptitle(f"Q-values for Agent {agent + 1}", fontsize=16, y=0.94)
        if savefig is not None:
            plt.savefig(f"{savefig}_{agent}.pdf", format="pdf")

        plt.show()

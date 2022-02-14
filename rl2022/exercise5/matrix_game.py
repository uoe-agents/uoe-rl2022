from collections.abc import Iterable
import numpy as np
import gym


def matrix_shape(matrix, shape=[]):
    """
    Get shape of matrix
    :param matrix: list of lists
    :return: shape of matrix
    """
    if not isinstance(matrix, Iterable):
        return shape
    else:
        shape.append(len(matrix))
        return matrix_shape(matrix[0], shape)


def actions_to_onehot(num_actions, actions):
    """
    Transfer actions to onehot representation
    :param num_actions: list of number of actions of each agent
    :param actions: list of actions (int) for each agent
    :return: onehot representation of actions
    """
    onehot = [[0] * num_action for num_action in num_actions]
    for ag, act in enumerate(actions):
        onehot[ag][act] = 1
    return onehot


class MatrixGame(gym.Env):
    def __init__(self, payoff_matrix, ep_length):
        """
        Create matrix game
        :param payoff_matrix: list of lists or numpy array for payoff matrix of all agents
        :param ep_length: length of episode (before done is True)
        """
        self.payoff = payoff_matrix
        self.num_actions = matrix_shape(payoff_matrix, [])
        self.n_agents = len(self.num_actions)
        self.ep_length = ep_length

        self.t = 0

        self.action_space = gym.spaces.Tuple([gym.spaces.Discrete(num_action) for num_action in self.num_actions])

    def reset(self):
        self.t = 0
        return None

    def step(self, action):
        self.t += 1
        self.last_actions = action
        reward = self.payoff
        for a in action:
            reward = reward[a]

        if self.t >= self.ep_length:
            done = True
        else:
            done = False

        return None, [reward] * self.n_agents, [done] * self.n_agents, {}

    def render(self):
        print(f"Step {self.t} - actions: {self.last_actions}")

# penalty game
def create_penalty_game(penalty, ep_length):
    assert penalty <= 0
    payoff = [
        [penalty, 0, 10],
        [0, 2, 0],
        [10, 0, penalty],
    ]
    game = MatrixGame(payoff, ep_length)
    return game

# climbing game
def create_climbing_game(ep_length):
    payoff = [
        [0, 6, 5],
        [-30, 7, 0],
        [11, -30, 0],
    ]
    game = MatrixGame(payoff, ep_length)
    return game

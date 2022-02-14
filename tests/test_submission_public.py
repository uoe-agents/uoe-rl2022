"""
Those are tests that will be shared with students
They should test that the code structure/return values
are of correct type/shape
"""

import os
import pytest
from pathlib import Path

@pytest.fixture(scope="session")
def rl2022_dir():
    path_base = os.path.dirname(os.path.dirname(__file__))
    rl2022_path = os.path.join(path_base, "rl2022")
    return rl2022_path

def test_exercise1(rl2022_dir):
    ex1_path = os.path.join(rl2022_dir, "exercise1")
    init_path = os.path.join(ex1_path, "__init__.py")
    assert os.path.isfile(init_path)
    mdp_solver_path = os.path.join(ex1_path, "mdp_solver.py")
    assert os.path.isfile(mdp_solver_path)

def test_exercise2(rl2022_dir):
    ex2_path = os.path.join(rl2022_dir, "exercise2")
    init_path = os.path.join(ex2_path, "__init__.py")
    assert os.path.isfile(init_path)
    agents_path = os.path.join(ex2_path, "agents.py")
    assert os.path.isfile(agents_path)
    train_mc_path = os.path.join(ex2_path, "train_monte_carlo.py")
    assert os.path.isfile(train_mc_path)
    train_q_path = os.path.join(ex2_path, "train_q_learning.py")
    assert os.path.isfile(train_q_path)

def test_exercise3(rl2022_dir):
    ex3_path = os.path.join(rl2022_dir, "exercise3")
    init_path = os.path.join(ex3_path, "__init__.py")
    assert os.path.isfile(init_path)
    agents_path = os.path.join(ex3_path, "agents.py")
    assert os.path.isfile(agents_path)
    train_dqn_path = os.path.join(ex3_path, "train_dqn.py")
    assert os.path.isfile(train_dqn_path)
    train_reinforce_path = os.path.join(ex3_path, "train_reinforce.py")
    assert os.path.isfile(train_reinforce_path)
    dqn_ll_params_path = os.path.join(ex3_path, "dqn_lunarlander_latest.pt")
    assert os.path.isfile(dqn_ll_params_path)
    loss_path = os.path.join(ex3_path, "loss.pdf")
    assert os.path.isfile(loss_path)

def test_exercise4(rl2022_dir):
    ex4_path = os.path.join(rl2022_dir, "exercise4")
    init_path = os.path.join(ex4_path, "__init__.py")
    assert os.path.isfile(init_path)
    agents_path = os.path.join(ex4_path, "agents.py")
    assert os.path.isfile(agents_path)
    train_ddpg_path = os.path.join(ex4_path, "train_ddpg.py")
    assert os.path.isfile(train_ddpg_path)
    pendulum_params_path = os.path.join(ex4_path, "bipedal_latest.pt")
    assert os.path.isfile(pendulum_params_path)

def test_exercise5(rl2022_dir):
    ex5_path = os.path.join(rl2022_dir, "exercise5")
    init_path = os.path.join(ex5_path, "__init__.py")
    assert os.path.isfile(init_path)
    agents_path = os.path.join(ex5_path, "agents.py")
    assert os.path.isfile(agents_path)
    train_iql_path = os.path.join(ex5_path, "train_iql.py")
    assert os.path.isfile(train_iql_path)
    train_jal_path = os.path.join(ex5_path, "train_jal.py")
    assert os.path.isfile(train_jal_path)

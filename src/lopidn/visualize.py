import sys
import os
from pathlib import Path

project_dir = Path(os.getcwd()).parent.parent
os.chdir(project_dir)
print(os.getcwd())

sys.path.append(str(project_dir))
sys.path.append('../../')
sys.path.append('../src/lopidn')

import gymnasium as gym
import random
import torch as th
import numpy as np
from tqdm import tqdm
import yaml
from config import RunConfig
from data import generate_dataset
from optimization import optimize_controller, optimize_LMN
import utils
from models import LocalModelNetwork, CriticLMN

def plot_phase_diagram(router, env, n_episodes):

    STATES = []
    ACTIONS = []
    MEMBERSCHIP = []

    for i in range(n_episodes):
        states, actions, membership = [], [], []
        router.reset()
        obs, _ = env.reset(seed=i)
        terminated, truncated = False, False

        while not terminated or truncated:
            # Inference
            action, weights = router._route(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            states.append(obs)
            actions.append(action)
            membership.append(weights)

        if terminated or truncated:
            STATES.append(np.array(states))
            ACTIONS.append(np.array(actions))
            MEMBERSCHIP.append(np.array(membership))
            break
        else:
            obs = next_obs

    print()










if __name__ == '__main__':

    # Load config
    args = None
    with open(f'{project_dir}/config/Pendulum.yaml', "r") as f:
        args = yaml.safe_load(f)
    args = RunConfig(**args)
    # Load model modules
    args.load_modules()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    # Create environment
    env = gym.make(args.env_id)



    router = CriticLMN(n_inputs=env.observation_space.shape[0],
                       n_outputs=env.action_space.shape[0],
                       n_controllers=args.n_controllers,
                       controller_class=args.controller_class,
                       controller_target=args.target,
                       controller_bound=(env.action_space.low, env.action_space.high),
                       critic=dataset.critic)

    # Load trained router
    router.load('pendulum_router.th')

    plot_phase_diagram(router, env, n_episodes=1)
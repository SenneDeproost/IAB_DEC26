import importlib
from collections import defaultdict
import numpy as np
from dataclasses import dataclass
from dataclasses import dataclass, asdict
from typing import List, Union, Any
import pickle
import gymnasium as gym
from tqdm import tqdm
import sys

sys.path.append('../../')

import models

class TrajectoryDataset:
    def __init__(self, states=None, actions=None,
                 n_episodes: Union[int, None] = None,
                 mean_return: Union[float, None] = None,
                 critic: Union[models.Critic, None] = None):
        self.states = states
        self.actions = actions
        self.n_episodes = n_episodes
        self.mean_return = mean_return
        self.critic = critic

    def __str__(self) -> str:
        return f'Dataset with size {len(self)}'

    def __repr__(self) -> dict[str, Any]:
        return {
                'states': self.states,
                'actions': self.actions
                }

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (self.states[idx], self.actions[idx])

    def __iter__(self):
        return zip(self.states, self.actions)

    def save(self, path):
        with open(path, "wb+") as f:
            data = {
                'states': self.states,
                'actions': self.actions,
                'mean_return': self.mean_return,
                'n_episodes': self.n_episodes,
                'critic': self.critic
            }

            pickle.dump(data, f)
            f.close()

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
            f.close()
            self.states = data['states']
            self.actions = data['actions']
            self.mean_return = data['mean_return']
            self.n_episodes = data['n_episodes']
            self.critic = data['critic']


def generate_dataset(env_id, agent, algorithm, n_episodes, save_path=None, load_path=None):
    env = gym.make(env_id)
    agent = algorithm.load(agent, env=env)

    # Load dataset if path is given
    if load_path:
        dataset = TrajectoryDataset()
        dataset.load(load_path)
        print('Mean return of training dataset: ', dataset.mean_return)
        return dataset

    # Retrieve critic in a correct manner
    alg = algorithm.__name__
    match alg:
        case 'TD3':
            critic = agent.policy.critic.qf0  # ToDo: Use multiple critics
        case _:
            raise NotImplementedError(f"{alg} algorithm not supported")
    critic = models.Critic(critic)

    # Generate data
    returns = []
    all_states, all_actions = [], []

    for i in tqdm(range(n_episodes)):
        obs, _ = env.reset(seed=i)
        r = 0
        states, actions = [], []
        terminated, truncated = False, False
        while not terminated or truncated:
            action, _ = agent.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            #value = critic.predict_q(obs, action)
            r += reward
            states.append(obs)
            actions.append(action)

            if terminated or truncated:
                obs, _ = env.reset()
                returns.append(r)
                all_states.append(states)
                all_actions.append(actions)
                break
            else:
                obs = next_obs

    mean_return = np.mean(returns)
    print(f"Mean return of training dataset: {mean_return}")

    dataset = TrajectoryDataset(all_states, all_actions, n_episodes, mean_return, critic)

    if save_path:
        dataset.save(save_path)

    return dataset


if __name__ == '__main__':
    env_id = 'MountainCarContinuous-v0'
    agent_alg = 'TD3'
    agent_path = f'trained_rl/agents/{agent_alg}/{env_id}.zip'
    module = importlib.import_module(f'stable_baselines3.{agent_alg.lower()}.{agent_alg.lower()}')
    agent_alg = getattr(module, agent_alg)

    # Test generating
    dataset_path = f'trained_rl/datasets/{agent_alg}/{env_id}.pkl'
    dataset = generate_dataset(env_id, agent_path, agent_alg, n_episodes=100, save_path=dataset_path)
    print(dataset[0][0])

    # Test loading
    dataset = generate_dataset(env_id, agent_path, agent_alg, n_episodes=100, load_path=dataset_path)
    print(dataset[0][0])

import gymnasium as gym
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def validate(controller, env_id, n_episodes, render_env=False):
    returns = []
    if render_env:
        env = gym.make(env_id, render_mode='human')
    else:
        env = gym.make(env_id)

    for i in range(n_episodes):
        obs, _ = env.reset(seed=i)
        r = 0
        controller.reset()
        terminated, truncated = False, False
        while not terminated or truncated:
            action = controller.predict(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            if render_env:
                env.render()
            r += reward

            if terminated or truncated:
                obs, _ = env.reset()
                returns.append(r)
                break
            else:
                obs = next_obs
    return returns

def plot_routing(model, env_id, seed=0):
    env = gym.make(env_id)
    obs, _ = env.reset(seed=seed)
    r = 0
    w = []
    model.reset()
    terminated, truncated = False, False
    while not terminated or truncated:
        action, weights = model._route(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        r += reward
        w.append(weights)

        if terminated or truncated:
            obs, _ = env.reset()
            break
        else:
            obs = next_obs

    w = np.array(w)
    for c in range(model.n_controllers):
        plt.plot(w[:,c])
    plt.show()
    #print(w[:-10])
    print(model.get_gains())

    return r
import gymnasium as gym
from tqdm import tqdm
import random
import torch
import numpy as np
from matplotlib import pyplot as plt

import imageio
from IPython.display import Image, display

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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

def create_gif(controller, args, path):
    img_env = gym.make(args.env_id, render_mode='rgb_array')
    obs, info = img_env.reset(seed=0)
    done = False
    truncated = False

    frames = []
    # Capture initial frame
    frames.append(img_env.render())

    while not (done or truncated):
        action, _ = controller.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = img_env.step(action)
        frames.append(img_env.render())

    img_env.close()

    # Save the captured frames as a GIF that repeats (loop=0)
    gif_path = path
    imageio.mimsave(gif_path, frames, fps=30, loop=0)

    # Render GIF inline in the notebook
    display(Image(filename=gif_path))
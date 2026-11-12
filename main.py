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
from tqdm import tqdm
import yaml
from config import RunConfig
from data import generate_dataset
from optimization import optimize_controller, optimize_LMN
import utils
from models import LocalModelNetwork, CriticLMN


import numpy as np




def main(args):
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    # Create environment
    env = gym.make(args.env_id)

    # Data
    dataset = generate_dataset(env_id=args.env_id,
                                     agent=f'{project_dir}/{args.rl_agent}',
                                     save_path=f'{project_dir}/trained_rl/datasets/{args.algorithm}/{args.env_id}.pkl',
                                     algorithm=args.algorithm_class,
                                     n_episodes=args.data_n_episodes)

    router = CriticLMN(n_inputs=env.observation_space.shape[0],
                               n_outputs=env.action_space.shape[0],
                               n_controllers=args.n_controllers,
                               controller_class=args.controller_class,
                               controller_target=args.target,
                               controller_bound = (env.action_space.low, env.action_space.high),
                               critic=dataset.critic)


    #r = router.forward(th.tensor(dataset[0][0][0]))
    optimize_LMN(model=router,
                 data=dataset,
                 args=args)

    val_ret = utils.validate(controller=router,
                   env_id=args.env_id,
                   render_env=False,
                   n_episodes=args.validation_n_episodes)

    utils.plot_routing(router, args.env_id, seed=0)

    router.save('router.th')

    print(f'Mean return: {np.mean(val_ret)} ± {np.std(val_ret)}')



if __name__ == '__main__':

    # Load config
    args = None
    with open(f'{project_dir}/config/Pendulum.yaml', "r") as f:
        args = yaml.safe_load(f)
    args = RunConfig(**args)
    # Load model modules
    args.load_modules()

    main(args)
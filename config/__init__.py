from dataclasses import dataclass
import importlib
import sys
from pathlib import Path
from typing import Union

sys.path.append('../../')

@dataclass
class RunConfig:
    # Run settings
    algorithm: str = ""
    rl_agent: str = ""
    env_id: str = ""
    seed: int = 0
    n_iterations: int = 0
    project: str = ""
    save_dir: str = ""

    # Target
    target: tuple[float] = tuple([0.0])

    # Local Model Network
    n_controllers: int = 0
    controller_class: str = ""
    controller_init: Union[str, None] = None

    # Optimizing controller
    learning_rate: float = 0.0
    batch_size: int = 0
    n_epochs_training: int = 0
    es_patience: int = 0
    es_min_delta: float = 0.0

    # Episodes for data
    data_n_episodes: int = 0

    # Validating subpolicies
    validation_n_episodes: int = 0

    # Config file
    config_file: str = ""

    def load_modules(self):
        # SB3 agent
        module = importlib.import_module(f'stable_baselines3.{self.algorithm.lower()}.{self.algorithm.lower()}')
        self.algorithm_class = getattr(module, self.algorithm)
        # Controller model
        module = importlib.__import__(f'controllers')
        self.controller_class = getattr(module, self.controller_class)
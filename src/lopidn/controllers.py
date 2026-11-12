import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym


class MIMOPID(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, Kp=None, Ki=None, Kd=None, dt=1.0):
        super().__init__()
        nn.Module.__init__(self)

        # I/O
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.dt = dt

        # Bounds, unbounded by default
        self.min_bound = -th.ones(n_inputs) * th.inf
        self.max_bound = th.ones(n_inputs) * th.inf

        # Target setpoint
        self.target = None

        # PID gains
        if Kp is None:
            Kp = th.zeros(n_outputs, n_inputs)
        else:
            Kp = th.tensor(Kp, dtype=th.float32)

        if Ki is None:
            Ki = th.zeros(n_outputs, n_inputs)
        else:
            Ki = th.tensor(Ki, dtype=th.float32)

        if Kd is None:
            Kd = th.zeros(n_outputs, n_inputs)
        else:
            Kd = th.tensor(Kd, dtype=th.float32)

        # Learnable parameters
        self.Kp = nn.Parameter(Kp.float())
        self.Ki = nn.Parameter(Ki.float())
        self.Kd = nn.Parameter(Kd.float())

        self.integral_error = None
        self.previous_error = None

    def get_gains(self):
        r = {
            'Kp': self.Kp.detach().numpy().tolist(),
            'Ki': self.Ki.detach().numpy().tolist(),
            'Kd': self.Kp.detach().numpy().tolist(),
        }
        return r

    def set_target(self, target):
        self.target = th.tensor(target)

    def set_bounds(self, min_bound, max_bound):
        self.min_bound = th.tensor(min_bound)
        self.max_bound = th.tensor(max_bound)

    def reset(self):
        self.integral_error = None
        self.previous_error = None

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = th.tensor(obs)
        res = self.forward(obs)
        return res.detach().numpy()

    def predict(self, obs):
        return self.__call__(obs)

    def forward(self, obs: th.Tensor) -> th.Tensor:

        assert self.target is not None, "No target provided."
        assert self.target.shape == obs.shape, "Different shape between target and obs."

        current_error = self.target - obs

        # Initialize integral and prev error if it is the first step taken
        if self.integral_error is None:
            self.integral_error = th.zeros_like(current_error)
            self.previous_error = th.zeros_like(current_error)

        # Calculate terms
        self.integral_error = (self.integral_error + current_error * self.dt)
        derivative_error = (current_error - self.previous_error) / self.dt

        # MIMO
        P = current_error @ self.Kp.T
        I = self.integral_error @ self.Ki.T
        D = derivative_error @ self.Kd.T

        self.previous_error = current_error

        # Produce bounded action
        action = th.clamp(P + I + D, self.min_bound, self.max_bound)
        return action

class MIMOPID_WIND(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, Kp=None, Ki=None, Kd=None, dt=1.0):
        super().__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.dt = dt

        # Bounds live in action space (n_outputs), unbounded by default
        self.min_bound = -th.ones(n_outputs) * th.inf
        self.max_bound = th.ones(n_outputs) * th.inf

        self.target = None

        def _gain(g):
            return th.zeros(n_outputs, n_inputs) if g is None else th.tensor(g, dtype=th.float)

        self.Kp = nn.Parameter(_gain(Kp))
        self.Ki = nn.Parameter(_gain(Ki))
        self.Kd = nn.Parameter(_gain(Kd))

        self.integral_term = None   # integrator state, in output space (n_outputs)
        self.previous_error = None

    def get_gains(self):
        return {
            'Kp': self.Kp.detach().numpy().tolist(),
            'Ki': self.Ki.detach().numpy().tolist(),
            'Kd': self.Kd.detach().numpy().tolist(),  # was returning Kp
        }

    def set_target(self, target):
        self.target = th.tensor(target, dtype=th.float)

    def set_bounds(self, min_bound, max_bound):
        self.min_bound = th.as_tensor(min_bound, dtype=th.float)
        self.max_bound = th.as_tensor(max_bound, dtype=th.float)

    def reset(self):
        self.integral_term = None
        self.previous_error = None

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = th.tensor(obs, dtype=th.float)
        return self.forward(obs).detach().numpy()

    def predict(self, obs):
        return self.__call__(obs)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        assert self.target is not None, "No target provided."
        assert self.target.shape == obs.shape, "Different shape between target and obs."

        current_error = self.target - obs

        if self.integral_term is None:
            self.integral_term = th.zeros(*current_error.shape[:-1], self.n_outputs)
            self.previous_error = th.zeros_like(current_error)

        derivative_error = (current_error - self.previous_error) / self.dt
        self.previous_error = current_error

        P = current_error @ self.Kp.T
        D = derivative_error @ self.Kd.T

        # Candidate integral update, in output space
        I_increment = (current_error * self.dt) @ self.Ki.T
        I_candidate = self.integral_term + I_increment

        # Anti-windup (conditional integration): freeze an output's integrator
        # if it is saturated and the increment would drive it further into saturation
        u_unsat = P + I_candidate + D
        blocked = ((u_unsat > self.max_bound) & (I_increment > 0)) | \
                  ((u_unsat < self.min_bound) & (I_increment < 0))
        self.integral_term = th.where(blocked, self.integral_term, I_candidate)

        action = th.clamp(P + self.integral_term + D, self.min_bound, self.max_bound)
        return action


if __name__ == '__main__':
    import utils

    env_id = 'MountainCarContinuous-v0'
    env = gym.make(env_id)
    agent = MIMOPID(env.observation_space.shape[0], env.action_space.shape[0],
                    Kp=[-2.759854, -4.2243414], Ki=[0.1481911, -0.09588563], Kd=[-2.8392627, -6.9371767])
    agent.set_target([0.45, 0.0])
    agent.set_bounds(env.action_space.low, env.action_space.high)

    val_ret = utils.validate(agent, env_id, 100)
    print(print(f'Mean return: {np.mean(val_ret)} ± {np.std(val_ret)}'))

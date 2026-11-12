from abc import abstractclassmethod
from collections import defaultdict

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import utils

import controllers


class Critic:
    def __init__(self, model):
        self.model = model

    def predict_q(self, state, action):
        state = th.tensor(np.array(state), dtype=th.float)
        action = th.tensor(np.array(action), dtype=th.float)
        data = th.concat((state, action), dim=0)
        with th.no_grad():
            value = self.model(data)
        value = value.detach().numpy()
        return value

    def forward(self, state, action):
        data = th.concat((state, action), dim=0)
        with th.no_grad():
            return self.model(data)


    def predict_ratio(self, state, og_action, action):
        og_q = self.predict_q(state, og_action)
        q = self.predict_q(state, action)
        nom = q - og_q
        denom = np.abs(og_q) - np.abs(q)
        return nom / denom


class GaussianBasis(th.nn.Module):
    def __init__(self, n_inputs, n_models):
        super().__init__()
        nn.Module.__init__(self)
        self.centers = nn.Parameter(th.randn(n_models, n_inputs))
        self.log_sigma = nn.Parameter(th.zeros(n_models, n_inputs))
        self.weights = nn.Parameter(th.randn(n_models))
        #self.bias = nn.Parameter(th.zeros(1))

    def forward(self, obs:th.Tensor):
        diff = obs.unsqueeze(-2) - self.centers
        # Ensure positive sigma
        sigma = F.softplus(self.log_sigma) + 1e-6
        gaussians = th.exp(-0.5 * diff * diff / sigma)
        # Take the product across dimensions
        gaussians = gaussians.prod(dim=-1)
        # Normalize the weights
        weights = F.softmax(self.weights, dim=0)
        output = gaussians * weights
        output = output / (output.sum(dim=-1, keepdim=True) + 1e-8)
        return output



class LocalModelNetwork(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_controllers,
                 controller_class, controller_target, controller_bound=None):
        super().__init__()
        nn.Module.__init__(self)
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_controllers = n_controllers
        self.controllers = nn.ModuleList()
        for _ in range(n_controllers):
            controller = controller_class(n_inputs,
                                          n_outputs,
                                          Kp=np.random.randn(n_outputs, n_inputs) * 1,
                                          Ki=np.random.randn(n_outputs, n_inputs) * 1,
                                          Kd=np.random.randn(n_outputs, n_inputs) * 1)
            controller.set_target(controller_target)
            controller.set_bounds(*controller_bound) if controller_bound is not None else None
            self.controllers.append(controller)
        self.weights = [[] for _ in range(n_controllers)]
        self.predictions = [[] for _ in range(n_controllers)]

    def reset(self):
        self.weights = [[] for _ in range(self.n_controllers)]
        self.predictions = [[] for _ in range(self.n_controllers)]
        self.reset_controllers()

    def reset_controllers(self):
        for controller in self.controllers:
            controller.reset()

    def get_gains(self):
        dict = {}
        for i, controller in enumerate(self.controllers):
            dict[i] = controller.get_gains()
        return dict

    def calculate_loss(self, og_actions):
        w = th.stack([th.stack(c) for c in self.weights])
        pred_actions = th.stack([th.stack(c) for c in self.predictions])
        # Linear combination
        weighted_pred = w.unsqueeze(-1) * pred_actions
        weighted_action = weighted_pred.sum(dim=0)
        diff = og_actions - weighted_action
        return th.mean((diff).pow(2))


    @abstractclassmethod
    def predict(self, state):
        pass


class CriticLMN(LocalModelNetwork):
    def __init__(self, n_inputs, n_outputs, n_controllers, controller_class, controller_target, critic, controller_bound=None):
        super().__init__(n_inputs, n_outputs, n_controllers, controller_class, controller_target, controller_bound)
        self.softmax = nn.Softmax(dim=-1)
        self.critic = critic


    def _forward_controllers(self, obs:th.Tensor):
        res = []
        for i, controller in enumerate(self.controllers):
            action = controller.forward(obs)
            res.append(action)
            self.predictions[i].append(action)
        return res

    def forward(self, obs):
        # List of actions per controller
        pred_actions = self._forward_controllers(obs)
        # Get values per controller
        with th.no_grad():
            q = th.tensor([self.critic.forward(obs, a) for a in pred_actions])
            norm_q = self.softmax(q)
            [self.weights[i].append(norm_q[i]) for i in range(self.n_controllers)]
        # Calculate results
        res = 0
        for i in range(self.n_controllers):
            #res += norm_q[i] * pred_actions[i]
            res += 1 * pred_actions[i]
        return res

    def _route(self, obs):
        control_output = np.array([ctrl.predict(obs) for ctrl in self.controllers])
        critic_output = np.array([self.critic.predict_q(obs, act) for act in control_output])
        critic_weights = utils.softmax(critic_output)
        w = [float(c[0]) for c in critic_weights]
        res = control_output*critic_weights
        res = np.sum(res, axis=0)
        return res, w

    def predict(self, obs):
        return self._route(obs)[0]

    def save(self, save_path):
        th.save({'state_dict': self.state_dict(), 'critic': self.critic.model.state_dict()}, save_path)

    def load(self, load_path):
        module = th.load(load_path)
        self.load_state_dict(module['state_dict'])
        self.critic.model.load_state_dict(module['critic'])





if __name__ == "__main__":
    gaussians = GaussianBasis(2, 5)
    x = th.randn(32, 2)
    y = gaussians(x)
    print()

#    router = LocalModelNetwork(n_inputs=2, n_outputs=2, n_controllers=2, controller_class=controllers.MIMOPID)
#    print()

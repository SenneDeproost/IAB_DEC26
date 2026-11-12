import torch as th
from tqdm import tqdm
import numpy as np
import utils


def optimize_controller(controller, data, args, weights=None):

    optimizer = th.optim.Adam(controller.parameters(), lr=args.learning_rate)

    for epoch in tqdm(range(args.n_epochs_training)):

        total_loss = 0.0

        # Go over each trajectory
        for states, rl_actions in data:

            states = th.tensor(np.array(states), requires_grad=True)
            rl_actions = th.tensor(np.array(rl_actions), requires_grad=True)

            controller.reset()
            ctrl_actions = []

            for t in range(len(states)):

                current_state = states[t]
                ctrl_action = controller.forward(current_state)
                ctrl_actions.append(ctrl_action.reshape(1, -1))

            pred_actions = th.cat(ctrl_actions, dim=0)

            # Calculate MSE loss between RL actions and the ones produced by the controller
            ctrl_loss = th.mean(pred_actions - rl_actions).pow(2)
            total_loss += ctrl_loss

        # Average the total loss over all trajectories
        total_loss = total_loss / len(data)

        # Optimize
        optimizer.zero_grad()
        total_loss.backward()
        th.nn.utils.clip_grad_norm_(
            controller.parameters(),
            max_norm=10.0
        )
        optimizer.step()

        # Logging
        if epoch % 25 == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"Loss = {total_loss.item():.8f}"
            )
            #print(controller.get_gains())
    return True

def optimize_LMN(model, data, args, weights=None):

    optimizer = th.optim.Adam(model.controllers.parameters(), lr=args.learning_rate)

    for epoch in tqdm(range(args.n_epochs_training)):

        total_loss = 0.0

        # Go over each trajectory
        for states, rl_actions in data:

            states = th.tensor(np.array(states), requires_grad=True)
            rl_actions = th.tensor(np.array(rl_actions), requires_grad=True)
            model.reset() #!!!!
            ctrl_actions = []

            for t in range(len(states)):

                current_state = states[t]
                ctrl_action = model.forward(current_state)
                ctrl_actions.append(ctrl_action.reshape(1, -1))

            #pred_actions = th.cat(ctrl_actions, dim=0)

            # Calculate MSE loss between RL actions and the ones produced by the controller
            # During loss calculation, the loss of each controller is weighted by the router

            loss = model.calculate_loss(rl_actions)
            total_loss += loss

        # Average the total loss over all trajectories
        total_loss = total_loss / len(data)

        # Optimize
        optimizer.zero_grad()
        total_loss.backward()
        th.nn.utils.clip_grad_norm_(
            model.controllers.parameters(),
            max_norm=10.0
            )
        optimizer.step()

        # Logging
        if epoch % 5 == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"Loss = {total_loss.item():.8f}"
            )
            #val_ret = utils.validate(controller=model,
            #                         env_id=args.env_id,
            #                         render_env=False,
            #                         n_episodes=args.validation_n_episodes)
            #print(model.get_gains())
            #print(f'Mean return: {np.mean(val_ret)} ± {np.std(val_ret)}')
    return True


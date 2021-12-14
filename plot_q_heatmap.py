import os
import shutil
import numpy as np
from mlagents.torch_utils import torch
from mlagents.trainers.torch.networks import ValueNetwork
from mlagents.trainers.settings import NetworkSettings
from mlagents.trainers.tests.dummy_config import create_observation_specs_with_shapes
from mlagents.trainers.torch.utils import ModelUtils
import matplotlib.pyplot as plt
np.set_printoptions(threshold=10_000)

OBS_SIZE = 2
ACT_SIZE = 2
DIV_SIZE = 4
HIDDEN_SIZE = 64
LAYERS = 2

class QNet(torch.nn.Module):
    def __init__(self, obs_spec, net_settings):
        super().__init__()
        self.q1_network = ValueNetwork(["extrinsic"], obs_spec, net_settings, ACT_SIZE, 1) 
        self.q2_network = ValueNetwork(["extrinsic"], obs_spec, net_settings, ACT_SIZE, 1) 

    def forward(self, inp, acts):
        return self.q1_network(inp, acts)[0]

def load_and_plot():
    run_id = "diayn-clean"
    model_path = "results/"+ run_id + "/BasicDiverse/checkpoint.pt"
    saved_state_dict = torch.load(model_path)
    obs_shapes = []
    if DIV_SIZE > 1:
        obs_shapes.append((DIV_SIZE,))
    obs_shapes.append((OBS_SIZE,))
    obs_spec = create_observation_specs_with_shapes(obs_shapes, [0])
    net_settings = NetworkSettings(num_layers=LAYERS, hidden_units=HIDDEN_SIZE, normalize=True, goal_conditioning_type=None)
    q_net = QNet(obs_spec, net_settings) 
    q_net.load_state_dict(saved_state_dict["Optimizer:q_network"])
    
    #states = [[0, 0], [0, 1], [1, 1], [1, 0], [-1,1], [0, -1], [1,-1], [-1, 0], [-1,-1]]
    states = [[0, 0], [0, 1], [1, 1]] #[1, 0], [-1,1], [0, -1], [1,-1], [-1, 0], [-1,-1]]
    X, Y = np.mgrid[-1:1:21j, -1:1:21j]
    actions = np.vstack((-1 * X.flatten(), Y.flatten())).T
    actions_t = torch.as_tensor(np.asanyarray([actions]), dtype=torch.float32).squeeze()
    fig, axs = plt.subplots(DIV_SIZE, len(states))
    for i, (ax, state) in enumerate(zip(axs, states)):
        if DIV_SIZE > 1:
            for div in range(DIV_SIZE):
                state = np.array(state)
                div_one_hot = np.zeros(DIV_SIZE)
                div_one_hot[div] = 1
                state_t = torch.as_tensor(np.asanyarray([state]), dtype=torch.float32).repeat(441, 1)
                div_t = torch.as_tensor(np.asanyarray([div_one_hot]), dtype=torch.float32).repeat(441, 1)
                #vals_t = q_net([state_t, div_t], actions_t)
                vals_t = q_net([div_t, state_t], actions_t)
                #print(np.array(sorted(list(torch.cat([vals_t["extrinsic"].unsqueeze(-1),actions_t], dim=1).detach().cpu().numpy()), key=lambda x: x[0])))

                vals = vals_t["extrinsic"].unsqueeze(-1).reshape(21, 21).detach().cpu().numpy()
                axs[div, i].imshow(vals)
                #if div == 0:
                #    ax.set_title(str(state))

        else:
            state = np.array(state)
            state_t = torch.as_tensor(np.asanyarray([state]), dtype=torch.float32).repeat(441, 1)
            vals_t = q_net([state_t], actions_t)
            #print(np.array(sorted(list(torch.cat([vals_t["extrinsic"].unsqueeze(-1),actions_t], dim=1).detach().cpu().numpy()), key=lambda x: x[0])))

            vals = vals_t["extrinsic"].unsqueeze(-1).reshape(21, 21).detach().cpu().numpy()
            ax.imshow(vals)
            ax.set_title(str(state))
    plt.savefig(run_id + ".png")
    plt.close()

if __name__ == "__main__":
    load_and_plot()

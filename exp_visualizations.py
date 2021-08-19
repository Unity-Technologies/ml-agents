import os
import numpy as np
import csv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math


def print_metrics(path, baseline=None, solved=0.8):
    with open(os.path.join(path, "reward_bins.npy"), "rb") as f:
        rewards = np.load(f)

    print(path.split("/")[-2])
    print("Avg Reward: {}".format(np.mean(rewards)))
    print("Max Reward: {}".format(np.amax(rewards)))
    if baseline is not None:
        print("% Solved: {}".format(np.count_nonzero(rewards > baseline * solved) / np.prod(rewards.shape)))
    print()


def plot_lasermaze(path, name=None):
    class Step:
        def __init__(self, x, y, z, r):
            self.__dict__.update(
                dict(x=x, y=y, z=z, r=r)
            )

    class Traj:
        def __init__(self):
            self.traj = []
            self.reward = 0
        
        def add_step(self, s):
            self.traj.append(s)

        def set_reward(self, r):
            self.reward = r

        def get_trajectory(self):
            return self.traj

        def get_reward(self):
            return self.reward

    class BehaviorData:
        def __init__(self):
            self.data = []

        def add_traj(self, t):
            self.data.append(t)

        def get_latest(self):
            return self.data[-1]

        def get_reward(self, k=1):
            rewards = []
            for traj in self.data:
                rewards.append(traj.get_reward())
            return np.mean(rewards)

    with open(path, "r") as f:
        reader = csv.DictReader(f)
    
        data = dict()
        for row in reader:
            t = int(row["timestep"])
            b = row["behavior"]
            x = float(row["x"])
            z = float(row["z"])
            c = row["crouch"] == "True"
            j = row["jump"] == "True"
            y = float(row["rotation"])
            r = float(row["reward"])

            if b not in data:
                data[b] = BehaviorData()

            if t == 0:
                data[b].add_traj(Traj())

            data[b].get_latest().add_step(Step(
                x,
                "m" if j else "b" if c else "k",
                z,
                math.radians(y)
            ))
            data[b].get_latest().set_reward(r)

    rewards = [d.get_reward() for d in data.values()]
    data = dict(reversed([d for _, d in sorted(zip(rewards, data.items()), key=lambda pair: pair[0])]))

    font = {
        "family" : "serif",
        "size"   : 12
    }
    matplotlib.rc("font", **font)

    fig, axs = plt.subplots(len(data), figsize=(5, 1.5*len(data)))
    axs = axs.flatten()
    for i, (behavior, behavior_data) in enumerate(data.items()):
        print("behavior {} had {} trajectories".format(behavior, len(behavior_data.data)))

        env = [
            patches.Rectangle((-5, -1.5), 10, 3, fill=True, color="darkgray"),
            patches.Rectangle((-5, -1.5), .5, 3, fill=True, color="dimgray"),
            patches.Rectangle((4.5, -1.5), .5, 3, fill=True, color="g"),
            patches.Rectangle((-3.3, -2), .1, 4, angle=0, fill=True, color="r"),
            patches.Rectangle((-1.6, -2), .1, 4, angle=-7, fill=True, color="r"),
            patches.Rectangle((-0.2, -2), .1, 4, angle=-5, fill=True, color="r"),
            patches.Rectangle((2.5, -2), .1, 4, angle=22, fill=True, color="r"),
            patches.Rectangle((3.6, -2), .1, 4, angle=0, fill=True, color="r"),
        ]
        for patch in env:
            axs[i].add_patch(patch)
    
        for traj in behavior_data.data[:100]:
            for step in traj.get_trajectory():
                axs[i].add_patch(patches.Arrow(
                    step.x, step.z, 
                    math.sin(step.r)/4, math.cos(step.r)/8,
                    width=.2,
                    fill=True, 
                    color=step.y, 
                    alpha=.1
                ))
                # axs[i].scatter(step.x, step.z, alpha=.5, c=step.y, marker=step.r, zorder=2)

        axs[i].title.set_text("return: {:.2f}".format(behavior_data.get_reward()))
        axs[i].set_xlim(-5, 5)
        axs[i].set_ylim(-1.5, 1.5)
        axs[i].tick_params(left=False,
                            bottom=False,
                            labelleft=False,
                            labelbottom=False)
    
    fig.tight_layout()
    if name is None:
        plt.show()
    else:
        plt.savefig("/Users/kolby.nottingham/Documents/laser_{}.pdf".format(name))



def plot_probs(path):

    idx = {
        0: ("4walls", "MEDE 2"),
        1: ("4walls", "MEDE 4"),
        2: ("4walls", "MEDE 8"),
        3: ("4walls", "DIAYN 2"),
        4: ("4walls", "DIAYN 4"),
        5: ("4walls", "DIAYN 8"),
        6: ("laser", "MEDE 2"),
        7: ("laser", "MEDE 4"),
        8: ("laser", "MEDE 8"),
        9: ("laser", "DIAYN 2"),
        10: ("laser", "DIAYN 4"),
        11: ("laser", "DIAYN 8"),
        12: ("worm", "MEDE 2"),
        13: ("worm", "MEDE 4"),
        14: ("worm", "MEDE 8"),
        15: ("worm", "DIAYN 2"),
        16: ("worm", "DIAYN 4"),
        17: ("worm", "DIAYN 8"),
        18: ("crawl", "MEDE 2"),
        19: ("crawl", "MEDE 4"),
        20: ("crawl", "MEDE 8"),
        21: ("crawl", "DIAYN 2"),
        22: ("crawl", "DIAYN 4"),
        23: ("crawl", "DIAYN 8")
    }
    mede_reward = {
        "4walls": {"DIAYN 2": [], "MEDE 2": [], "DIAYN 4": [], "MEDE 4": [], "DIAYN 8": [], "MEDE 8": []},
        "laser": {"DIAYN 2": [], "MEDE 2": [], "DIAYN 4": [], "MEDE 4": [], "DIAYN 8": [], "MEDE 8": []},
        "worm": {"DIAYN 2": [], "MEDE 2": [], "DIAYN 4": [], "MEDE 4": [], "DIAYN 8": [], "MEDE 8": []},
        "crawl": {"DIAYN 2": [], "MEDE 2": [], "DIAYN 4": [], "MEDE 4": [], "DIAYN 8": [], "MEDE 8": []},
    }
    diayn_reward = {
        "4walls": {"DIAYN 2": [], "MEDE 2": [], "DIAYN 4": [], "MEDE 4": [], "DIAYN 8": [], "MEDE 8": []},
        "laser": {"DIAYN 2": [], "MEDE 2": [], "DIAYN 4": [], "MEDE 4": [], "DIAYN 8": [], "MEDE 8": []},
        "worm": {"DIAYN 2": [], "MEDE 2": [], "DIAYN 4": [], "MEDE 4": [], "DIAYN 8": [], "MEDE 8": []},
        "crawl": {"DIAYN 2": [], "MEDE 2": [], "DIAYN 4": [], "MEDE 4": [], "DIAYN 8": [], "MEDE 8": []},
    }

    with open(path, "r") as f:
        reader = csv.reader(f)
        for i, (m, d) in enumerate(reader):
            j = i % 24
            experiment = idx[j]
            mede_reward[experiment[0]][experiment[1]].append(np.exp(float(m)))
            diayn_reward[experiment[0]][experiment[1]].append(np.exp(float(d)))
    
    print(sum(len(y) for x in mede_reward.values() for y in x.values()))
    font = {
        "family" : "serif",
        "size"   : 12
    }
    matplotlib.rc("font", **font)

    for env in mede_reward.keys():
        plt.clf()
        height = []
        err = []
        color = []
        tick_labels = []
        for algo in mede_reward[env].keys():
            height.append(np.mean(diayn_reward[env][algo]))
            height.append(np.mean(mede_reward[env][algo]))
            err.append(np.std(diayn_reward[env][algo]))
            err.append(np.std(mede_reward[env][algo]))
            color += ["r", "b"]
            tick_labels += ["", algo]
        x = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16]
        plt.bar(x, height, color=color, yerr=err, tick_label=tick_labels, align="edge")
        plt.ylim(0, 1)
        plt.tick_params(
            axis="x",
            which="both",
            bottom=False, 
            top=False
        )
        colors = {"p(z|s)":"red", "p(z|s,a)":"blue"}         
        labels = list(colors.keys())
        handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
        plt.legend(handles, labels)
        plt.savefig("/Users/kolby.nottingham/Documents/{}_probs.pdf".format(env))


def grid_visualization(config, weights):
    import torch
    import yaml
    from yaml import Loader
    from mlagents.trainers.policy.torch_policy import TorchPolicy
    from mlagents_envs.base_env import BehaviorSpec
    from mlagents_envs.base_env import ObservationSpec, ActionSpec
    from mlagents.trainers.settings import RunOptions

    state2pos = {
        0: (0, 0),
        1: (1, 0),
        2: (1, 1),
        3: (0, 1),
    }
    state_action = {
        (0, 1):(.2, 0),
        (0, 2):(.2, .2),
        (0, 3):(0, .2),
        (1, 0):(-.2, 0),
        (1, 2):(0, .2),
        (1, 3):(-.2, .2),
        (2, 0):(-.2, -.2),
        (2, 1):(0, -.2),
        (2, 3):(-.2, 0),
        (3, 0):(0, -.2),
        (3, 1):(.2, -.2),
        (3, 2):(.2, 0),
    }
    
    spec = BehaviorSpec(
        [
            ObservationSpec((4,), (0,), 0, ""), 
            ObservationSpec((10,), (0,), 1, "")
        ], 
        ActionSpec(0, (4,))
    )
    settings = RunOptions.from_dict(yaml.load(open(config), Loader=Loader)).behaviors["GridDiverse"]
    policy = TorchPolicy(0, spec, settings)

    policy.load_weights(torch.load(weights)["Policy"])
    
    fig, axs = plt.subplots(2, 5, figsize=(10, 5))
    axs = axs.flatten()

    for b in range(10):
        for s in range(4):
            masks = torch.ones((1, 4))            
            obs = [torch.zeros((1, 10)), torch.zeros((1, 4))]
            obs[0][0, b] = 1
            obs[1][0, s] = 1

            _, logprobs, _, _ = policy.sample_actions(obs, masks)
            a = torch.argmax(logprobs.all_discrete_list[0]).item()

            if s != a:
                axs[b].add_patch(patches.Arrow(
                    state2pos[s][0], state2pos[s][1], 
                    state_action[(s, a)][0], state_action[(s, a)][1],
                    width=.2,
                    fill=True, 
                    color="r"
                ))
            else:
                axs[b].add_patch(patches.Circle(
                    (state2pos[s][0], state2pos[s][1]),
                    .1,
                    fill=True,
                    color="r"
                ))

            axs[b].tick_params(left=False,
                                bottom=False,
                                labelleft=False,
                                labelbottom=False)
            axs[b].set_xlim(-.5, 1.5)
            axs[b].set_ylim(-.5, 1.5)

    fig.tight_layout()
    plt.savefig("/Users/kolby.nottingham/Documents/grid_{}.pdf".format("diayn" if "diayn" in weights else "mede"))


if __name__ == "__main__":
    grid_visualization(
        "/Users/kolby.nottingham/ml-agents/config/sac/GridDiverse.yaml", 
        "/Users/kolby.nottingham/ml-agents/results/grid-disc10-8-16/GridDiverse/GridDiverse-200512.pt", 
    )
    grid_visualization(
        "/Users/kolby.nottingham/ml-agents/config/sac/GridDiverse_diayn.yaml", 
        "/Users/kolby.nottingham/ml-agents/results/grid-diayn10-8-16/GridDiverse/GridDiverse-200512.pt", 
    )

    # plot_probs("/Users/kolby.nottingham/ml-agents/eval_results.txt")

    # plot_lasermaze("/Users/kolby.nottingham/ml-agents/Project/LaserMaze.txt")
    # plot_lasermaze("/Users/kolby.nottingham/ml-agents/Project/LaserMaze_diayn2_8_9.txt", "diayn2")
    # plot_lasermaze("/Users/kolby.nottingham/ml-agents/Project/LaserMaze_disc2_8_9.txt", "disc2")
    # plot_lasermaze("/Users/kolby.nottingham/ml-agents/Project/LaserMaze_diayn4_8_9.txt", "diayn4")
    # plot_lasermaze("/Users/kolby.nottingham/ml-agents/Project/LaserMaze_disc4_8_9.txt", "disc4")
    # plot_lasermaze("/Users/kolby.nottingham/ml-agents/Project/LaserMaze_diayn8_8_9.txt", "diayn8")
    # plot_lasermaze("/Users/kolby.nottingham/ml-agents/Project/LaserMaze_disc8_8_9.txt", "disc8")

    # print_metrics("/Users/kolby.nottingham/ml-agents/results/basic-disc4-inf/BasicDiverse", .95)
    # print_metrics("/Users/kolby.nottingham/ml-agents/results/basic-disc8-inf/BasicDiverse", .95)
    # print_metrics("/Users/kolby.nottingham/ml-agents/results/basic-cont1-inf/BasicDiverse", .95)
    # print_metrics("/Users/kolby.nottingham/ml-agents/results/basic-cont2-inf/BasicDiverse", .95)
    # print_metrics("/Users/kolby.nottingham/ml-agents/results/basic-diayn4-inf/BasicDiverse", .95)
    # print_metrics("/Users/kolby.nottingham/ml-agents/results/basic-diayn8-inf/BasicDiverse", .95)

    # print_metrics("/Users/kolby.nottingham/ml-agents/results/laser-disc4-inf/LaserMazeDiverse", .8)
    # print_metrics("/Users/kolby.nottingham/ml-agents/results/laser-disc8-inf/LaserMazeDiverse", .8)
    # print_metrics("/Users/kolby.nottingham/ml-agents/results/laser-cont1-inf/LaserMazeDiverse", .8)
    # print_metrics("/Users/kolby.nottingham/ml-agents/results/laser-cont2-inf/LaserMazeDiverse", .8)
    # print_metrics("/Users/kolby.nottingham/ml-agents/results/laser-diayn4-inf/LaserMazeDiverse", .8)
    # print_metrics("/Users/kolby.nottingham/ml-agents/results/laser-diayn8-inf/LaserMazeDiverse", .8)

    # print_metrics("/Users/kolby.nottingham/ml-agents/results/worm-disc4-inf/WormDiverse", 433)
    # print_metrics("/Users/kolby.nottingham/ml-agents/results/worm-disc8-inf/WormDiverse", 433)
    # print_metrics("/Users/kolby.nottingham/ml-agents/results/worm-cont1-inf/WormDiverse", 433)
    # print_metrics("/Users/kolby.nottingham/ml-agents/results/worm-cont2-inf/WormDiverse", 433)
    # print_metrics("/Users/kolby.nottingham/ml-agents/results/worm-diayn4-inf/WormDiverse", 433)
    # print_metrics("/Users/kolby.nottingham/ml-agents/results/worm-diayn8-inf/WormDiverse", 433)

    # print_metrics("/Users/kolby.nottingham/ml-agents/results/crawl-disc4-inf/CrawlerDiverse", 338)
    # print_metrics("/Users/kolby.nottingham/ml-agents/results/crawl-disc8-inf/CrawlerDiverse", 338)
    # print_metrics("/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-crawl-cont1-steps10m-005to05-8-3/CrawlerDiverse", 338)
    # print_metrics("/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-crawl-cont2-steps10m-005to05-8-3/CrawlerDiverse", 338)
    # print_metrics("/Users/kolby.nottingham/ml-agents/results/crawl-diayn4-inf/CrawlerDiverse", 338)
    # print_metrics("/Users/kolby.nottingham/ml-agents/results/crawl-diayn8-inf/CrawlerDiverse", 338)
    pass

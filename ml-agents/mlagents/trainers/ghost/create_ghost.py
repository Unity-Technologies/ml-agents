# # Unity ML-Agents Toolkit
# ## ML-Agent Learning (Ghost Trainer)
# Contains an implementation of a 'ghost trainer' which loads
# snapshots of the agent's past selves to train against in adversarial settings.

import logging
from collections import defaultdict

# from typing import Dict

# import numpy as np

from mlagents.envs.brain import BrainInfo, AllBrainInfo

# from mlagents.trainers.rl_trainer import RLTrainer, AllRewardsOutput
from mlagents.envs.action_info import ActionInfoOutputs
from mlagents.envs.action_info import ActionInfo

logger = logging.getLogger("mlagents.trainers")

""" Hacking together a ghost trainer"""


def safe_collect_from_list(l, indices):
    if len(indices) <= len(l):
        return [l[x] for x in indices]
    else:
        return []


# splits brain_info into separate brain_infos per team
def split_brain_info_by_team(brain_info, teams):
    indices = defaultdict(list)
    brain_infos = []
    print(brain_info.agents)
    for i, team_id in enumerate(brain_info.team_ids):
        indices[team_id].append(i)
    for team_id in range(teams):
        brain_infos.append(
            BrainInfo(
                visual_observation=safe_collect_from_list(
                    brain_info.visual_observations, indices[team_id]
                ),
                vector_observation=safe_collect_from_list(
                    brain_info.vector_observations, indices[team_id]
                ),
                text_observations=safe_collect_from_list(
                    brain_info.text_observations, indices[team_id]
                ),
                memory=safe_collect_from_list(brain_info.memories, indices[team_id]),
                reward=safe_collect_from_list(brain_info.rewards, indices[team_id]),
                agents=safe_collect_from_list(brain_info.agents, indices[team_id]),
                local_done=safe_collect_from_list(
                    brain_info.local_done, indices[team_id]
                ),
                vector_action=safe_collect_from_list(
                    brain_info.previous_vector_actions, indices[team_id]
                ),
                text_action=safe_collect_from_list(
                    brain_info.previous_text_actions, indices[team_id]
                ),
                max_reached=safe_collect_from_list(
                    brain_info.max_reached, indices[team_id]
                ),
                custom_observations=safe_collect_from_list(
                    brain_info.custom_observations, indices[team_id]
                ),
                team_ids=safe_collect_from_list(brain_info.team_ids, indices[team_id]),
                action_mask=safe_collect_from_list(
                    brain_info.action_masks, indices[team_id]
                ),
            )
        )
    return brain_infos


def create_ghost_trainer(
    trainer,
    num_ghosts,
    brain,
    reward_buff_cap,
    trainer_parameters,
    training,
    load,
    seed,
    *args,
):

    # Get trainer and policy classes
    trainer_type = trainer.__class__
    policy_type = trainer.policy.__class__

    class GhostPolicy(policy_type):
        def __init__(self, ghosts):

            super(GhostPolicy, self).__init__(
                seed, brain, trainer_parameters, training, load
            )

            self.ghosts = ghosts
            self.num_ghosts = len(ghosts)

        def get_action(self, brain_info: BrainInfo) -> ActionInfo:
            """
            Decides actions given observations information, and takes them in environment.
            :param brain_info: A dictionary of brain names and BrainInfo from environment.
            :return: an ActionInfo containing action, memories, values and an object
            to be passed to add experiences
            """
            if len(brain_info.agents) == 0:
                return ActionInfo([], [], [], None, None)

            # plus 1 for number of teams since num_ghosts = teams -1
            brain_infos = split_brain_info_by_team(brain_info, self.num_ghosts + 1)
            run_out = self.evaluate(brain_info)

            # current policy
            run_out = self.evaluate(brain_infos[0])
            # ghost actions
            for ghost in range(self.num_ghosts):
                ghost_run_out = self.ghosts[ghost].evaluate(brain_infos[ghost])
                run_out["action"] += ghost_run_out["action"]
                # run_out["memory_out"] += ghost_run_out["memory_out"]
                run_out["value"] += ghost_run_out["value"]

            return ActionInfo(
                action=run_out.get("action"),
                memory=run_out.get("memory_out"),
                text=None,
                value=run_out.get("value"),
                outputs=run_out,
            )

    # ghost trainer now wraps base trainer class
    class GhostTrainer(trainer_type):
        def __init__(self):
            super(GhostTrainer, self).__init__(
                brain, reward_buff_cap, trainer_parameters, training, load, seed, *args
            )

            # number of ghosts to spawn
            self.num_ghosts = num_ghosts

            # instantiate ghosts
            self.ghosts = []
            for _ in range(self.num_ghosts):
                self.ghosts.append(
                    policy_type(seed, brain, trainer_parameters, False, load)
                )
            self.policy = GhostPolicy(self.ghosts)

            self.policy_snapshots = []

        def process_experiences(
            self, current_info: AllBrainInfo, new_info: AllBrainInfo
        ) -> None:

            super(GhostTrainer, self).process_experiences(current_info, new_info)

        def add_experiences(
            self,
            curr_all_info: AllBrainInfo,
            next_all_info: AllBrainInfo,
            take_action_outputs: ActionInfoOutputs,
        ) -> None:

            super(GhostTrainer, self).add_experiences(
                curr_all_info, next_all_info, take_action_outputs
            )

        def save_snapshot(self):
            pass

        def swap_snapshot(self):
            pass

    return GhostTrainer()

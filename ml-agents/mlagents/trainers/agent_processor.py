from typing import List, Dict
from collections import defaultdict, Counter
import numpy as np

from mlagents.trainers.trainer import Trainer
from mlagents.trainers.trajectory import Trajectory, AgentExperience
from mlagents.trainers.brain import BrainInfo
from mlagents.trainers.tf_policy import TFPolicy
from mlagents.trainers.action_info import ActionInfoOutputs


class AgentProcessor:
    """
    AgentProcessor contains a dictionary per-agent trajectory buffers. The buffers are indexed by agent_id.
    Buffer also contains an update_buffer that corresponds to the buffer used when updating the model.
    One AgentProcessor should be created per agent group.
    """

    def __init__(self, trainer: Trainer, policy: TFPolicy, max_trajectory_length: int):
        """
        Create an AgentProcessor.
        :param trainer: Trainer instance connected to this AgentProcessor. Trainer is given trajectory
        when it is finished.
        :param policy: Policy instance associated with this AgentProcessor.
        :param max_trajectory_length: Maximum length of a trajectory before it is added to the trainer.
        """
        self.experience_buffers: Dict[str, List[AgentExperience]] = defaultdict(list)
        self.last_brain_info: Dict[str, BrainInfo] = {}
        self.last_take_action_outputs: Dict[str, ActionInfoOutputs] = defaultdict(
            ActionInfoOutputs
        )
        self.stats: Dict[str, List[float]] = defaultdict(list)
        # Note: this is needed until we switch to AgentExperiences as the data input type.
        # We still need some info from the policy (memories, previous actions)
        # that really should be gathered by the env-manager.
        self.policy = policy
        self.episode_steps: Counter = Counter()
        self.episode_rewards: Dict[str, float] = defaultdict(lambda: 0.0)
        if max_trajectory_length:
            self.max_trajectory_length = max_trajectory_length
            self.ignore_max_length = False
        else:
            self.max_trajectory_length = 0
            self.ignore_max_length = True
        self.trainer = trainer

    def add_experiences(
        self,
        curr_info: BrainInfo,
        next_info: BrainInfo,
        take_action_outputs: ActionInfoOutputs,
    ) -> None:
        """
        Adds experiences to each agent's experience history.
        :param curr_info: current BrainInfo.
        :param next_info: next BrainInfo.
        :param take_action_outputs: The outputs of the Policy's get_action method.
        """
        if take_action_outputs:
            self.stats["Policy/Entropy"].append(take_action_outputs["entropy"].mean())
            self.stats["Policy/Learning Rate"].append(
                take_action_outputs["learning_rate"]
            )
            for name, values in take_action_outputs["value_heads"].items():
                self.stats[name].append(np.mean(values))

        for agent_id in curr_info.agents:
            self.last_brain_info[agent_id] = curr_info
            self.last_take_action_outputs[agent_id] = take_action_outputs

        # Store the environment reward
        tmp_environment_reward = np.array(next_info.rewards, dtype=np.float32)

        for agent_id in next_info.agents:
            stored_info = self.last_brain_info.get(agent_id, None)
            stored_take_action_outputs = self.last_take_action_outputs.get(
                agent_id, None
            )
            if stored_info is not None:
                idx = stored_info.agents.index(agent_id)
                next_idx = next_info.agents.index(agent_id)
                obs = []
                if not stored_info.local_done[idx]:
                    for i, _ in enumerate(stored_info.visual_observations):
                        obs.append(stored_info.visual_observations[i][idx])
                    if self.policy.use_vec_obs:
                        obs.append(stored_info.vector_observations[idx])
                    if self.policy.use_recurrent:
                        memory = self.policy.retrieve_memories([agent_id])[0, :]
                    else:
                        memory = None

                    done = next_info.local_done[next_idx]
                    max_step = next_info.max_reached[next_idx]

                    # Add the outputs of the last eval
                    action = stored_take_action_outputs["action"][idx]
                    if self.policy.use_continuous_act:
                        action_pre = stored_take_action_outputs["pre_action"][idx]
                    else:
                        action_pre = None
                    action_probs = stored_take_action_outputs["log_probs"][idx]
                    action_masks = stored_info.action_masks[idx]
                    prev_action = self.policy.retrieve_previous_action([agent_id])[0, :]

                    values = stored_take_action_outputs["value_heads"]
                    experience = AgentExperience(
                        obs=obs,
                        reward=tmp_environment_reward[next_idx],
                        done=done,
                        action=action,
                        action_probs=action_probs,
                        action_pre=action_pre,
                        action_mask=action_masks,
                        prev_action=prev_action,
                        max_step=max_step,
                        memory=memory,
                    )
                    # Add the value outputs if needed
                    self.experience_buffers[agent_id].append(experience)

                if (
                    next_info.local_done[next_idx]
                    or (
                        not self.ignore_max_length
                        and len(self.experience_buffers[agent_id])
                        >= self.max_trajectory_length
                    )
                ) and len(self.experience_buffers[agent_id]) > 0:
                    # Make next AgentExperience
                    next_obs = []
                    for i, _ in enumerate(next_info.visual_observations):
                        next_obs.append(next_info.visual_observations[i][next_idx])
                    if self.policy.use_vec_obs:
                        next_obs.append(next_info.vector_observations[next_idx])
                    trajectory = Trajectory(
                        steps=self.experience_buffers[agent_id],
                        agent_id=agent_id,
                        next_obs=next_obs,
                    )
                    # This will eventually be replaced with a queue
                    self.trainer.process_trajectory(trajectory)
                    self.experience_buffers[agent_id] = []
                elif not next_info.local_done[next_idx]:
                    if agent_id not in self.episode_steps:
                        self.episode_steps[agent_id] = 0
                    self.episode_steps[agent_id] += 1
        self.policy.save_previous_action(
            curr_info.agents, take_action_outputs["action"]
        )

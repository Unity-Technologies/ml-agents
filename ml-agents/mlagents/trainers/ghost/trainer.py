# # Unity ML-Agents Toolkit
# ## ML-Agent Learning (Ghost Trainer)

import logging
from typing import Dict

import numpy as np

from mlagents.trainers.brain import BrainParameters, BrainInfo
from mlagents.trainers.tf_policy import TFPolicy
from mlagents.trainers.rl_trainer import RLTrainer, AllRewardsOutput
from mlagents.trainers.action_info import ActionInfoOutputs

from mlagents.trainers.ghost.tf_utils import TensorFlowVariables

logger = logging.getLogger("mlagents.trainers")


class GhostTrainer(RLTrainer):
    def __init__(
        self,
        trainer,
        brain_name,
        reward_buff_cap,
        trainer_parameters,
        training,
        load,
        seed,
        run_id,
    ):
        """
        Responsible for collecting experiences and training PPO model.
        :param trainer_parameters: The parameters for the trainer (dictionary).
        :param reward_buff_cap: Max reward history to track in the reward buffer
        :param training: Whether the trainer is set for training.
        :param load: Whether the model should be loaded.
        :param seed: The seed the model will be initialized with
        :param run_id: The identifier of the current run
        """
        super(GhostTrainer, self).__init__(
            brain_name, trainer_parameters, training, run_id, reward_buff_cap
        )

        self.trainer = trainer
        self.policies: Dict[str, TFPolicy] = {}
        self.policy_snapshots = []
        self.learning_policy_name: str = None
        self.current_policy_snapshot = None
        self.last_step = 0
        self.window = trainer_parameters["ghost"]["window"]
        self.current_prob = trainer_parameters["ghost"]["current_prob"]

    def process_experiences(
        self, name_behavior_id: str, current_info: BrainInfo, new_info: BrainInfo
    ) -> None:
        if name_behavior_id == self.learning_policy_name:
            self.trainer.process_experiences(name_behavior_id, current_info, new_info)
            self.stats = self.trainer.stats

    def add_experiences(
        self,
        name_behavior_id: str,
        curr_info: BrainInfo,
        next_info: BrainInfo,
        take_action_outputs: ActionInfoOutputs,
    ) -> None:
        if name_behavior_id == self.learning_policy_name:
            self.trainer.add_experiences(
                name_behavior_id, curr_info, next_info, take_action_outputs
            )

    def increment_step(self, n_steps: int) -> None:
        for policy in self.policies.values():
            self.step = policy.increment_step(n_steps)

    def add_policy_outputs(
        self, take_action_outputs: ActionInfoOutputs, agent_id: str, agent_idx: int
    ) -> None:
        self.trainer.add_policy_outputs(take_action_outputs, agent_id, agent_idx)

    def add_rewards_outputs(
        self,
        rewards_out: AllRewardsOutput,
        values: Dict[str, np.ndarray],
        agent_id: str,
        agent_idx: int,
        agent_next_idx: int,
    ) -> None:
        self.trainer.add_rewards_outputs(
            rewards_out, values, agent_id, agent_idx, agent_next_idx
        )

    def is_ready_update(self) -> bool:
        return self.trainer.is_ready_update()

    def save_model(self):
        with self.trainer.policy.graph.as_default():
            self.trainer.policy.tfvars.set_weights(self.current_policy_snapshot)
        self.trainer.policy.save_model(self.step)

    def export_model(self):
        self.trainer.export_model()

    def update_policy(self) -> None:
        self.trainer.update_policy()

        with self.trainer.policy.graph.as_default():
            weights = self.trainer.policy.tfvars.get_weights()
            self.current_policy_snapshot = weights

        if self.get_step - self.last_step > 10000:
            self.save_snapshot(self.trainer.policy)
            self.last_step = self.get_step

    def add_policy(self, brain_parameters: BrainParameters) -> None:
        policy = self.trainer.create_policy(brain_parameters)
        # for saving/swapping snapshots
        with policy.graph.as_default():
            policy.tfvars = TensorFlowVariables(policy.model.output, policy.sess)
        if not self.policy_snapshots:
            self.save_snapshot(policy)
            with policy.graph.as_default():
                weights = policy.tfvars.get_weights()
                self.current_policy_snapshot = weights

        self.policies[brain_parameters.brain_name] = policy

        if not self.learning_policy_name:
            self.set_learning_policy(brain_parameters.brain_name)

    def get_policy(self, name_behavior_id: str) -> TFPolicy:
        return self.policies[name_behavior_id]

    def save_snapshot(self, policy: TFPolicy) -> None:
        with policy.graph.as_default():
            weights = policy.tfvars.get_weights()
            self.policy_snapshots.append(weights)
        if len(self.policy_snapshots) > self.window:
            del self.policy_snapshots[0]

    def swap_snapshots(self) -> None:
        for name_behavior_id, policy in self.policies.items():
            # here is the place for a sampling protocol
            if name_behavior_id != self.learning_policy_name and np.random.uniform() < (
                1 - self.current_prob
            ):
                # snapshot = np.random.choice(self.policy_snapshots)
                x = np.random.randint(len(self.policy_snapshots))
                snapshot = self.policy_snapshots[x]
            else:
                snapshot = self.current_policy_snapshot
                x = "current"
            print(
                "Step {}: Swapping snapshot {} to id {} with {} learning".format(
                    self.get_step, x, name_behavior_id, self.learning_policy_name
                )
            )
            with policy.graph.as_default():
                policy.tfvars.set_weights(snapshot)

    def set_learning_policy(self, name_behavior_id: str) -> None:
        self.learning_policy_name = name_behavior_id
        self.swap_snapshots()
        try:
            policy = self.policies[self.learning_policy_name]
            self.trainer.set_policy(policy)
        except KeyError:
            pass

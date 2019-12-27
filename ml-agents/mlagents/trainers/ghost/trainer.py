# # Unity ML-Agents Toolkit
# ## ML-Agent Learning (Ghost Trainer)

import logging
from typing import Dict

import numpy as np

from mlagents.trainers.brain import BrainParameters
from mlagents.trainers.tf_policy import TFPolicy

# from mlagents.trainers.rl_trainer import RLTrainer
from mlagents.trainers.trajectory import Trajectory

from mlagents.trainers.ghost.tf_utils import TensorFlowVariables

logger = logging.getLogger("mlagents.trainers")


class GhostTrainer(object):
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

        self.trainer = trainer

        # super(GhostTrainer, self).__init__(
        #    brain_name, trainer_parameters, training, run_id, reward_buff_cap
        # )

        self.policies: Dict[str, TFPolicy] = {}
        self.policy_snapshots = []
        self.learning_policy_name: str = None
        self.current_policy_snapshot = None
        self.last_step = 0
        self.window = trainer_parameters["ghost"]["window"]
        self.current_prob = trainer_parameters["ghost"]["current_prob"]
        self.steps_between_snapshots = trainer_parameters["ghost"]["snapshot_per"]

    # def __getattribute__(self, name):
    #    trainer = object.__getattribute__(self, "trainer")
    #    return trainer.__getattribute__(name)

    def __getattr__(self, name):
        trainer = object.__getattribute__(self, "trainer")
        return trainer.__getattribute__(name)

    # def __get__(self, name):
    #    print("getting {} ".format(name))
    #    return self.trainer.__get__(name)

    def process_trajectory(self, trajectory: Trajectory) -> None:
        if trajectory.behavior_id == self.learning_policy_name:
            self.trainer.process_trajectory(trajectory)

    # def write_summary(self, global_step: int, delta_train_start: float) -> None:
    #    self.trainer.write_summary(global_step, delta_train_start)

    # def is_ready_update(self) -> bool:
    #    return self.trainer.is_ready_update()

    def save_model(self, name_behavior_id: str) -> None:
        policy = self.trainer.get_policy(name_behavior_id)
        with policy.graph.as_default():
            policy.tfvars.set_weights(self.current_policy_snapshot)
        policy.save_model(self.step)

    def export_model(self, name_behavior_id: str) -> None:
        policy = self.trainer.get_policy(name_behavior_id)
        with policy.graph.as_default():
            policy.tfvars.set_weights(self.current_policy_snapshot)
        policy.export_model()

    def update_policy(self) -> None:
        self.trainer.update_policy()

        with self.trainer.policy.graph.as_default():
            weights = self.trainer.policy.tfvars.get_weights()
            self.current_policy_snapshot = weights

        if self.get_step - self.last_step > self.steps_between_snapshots:
            self.save_snapshot(self.trainer.policy)
            self.last_step = self.get_step

    def create_policy(self, brain_parameters: BrainParameters) -> None:
        return self.trainer.create_policy(brain_parameters)

    def add_policy(self, name_behavior_id: str, policy: TFPolicy) -> None:
        # for saving/swapping snapshots
        with policy.graph.as_default():
            policy.tfvars = TensorFlowVariables(policy.model.output, policy.sess)
        if not self.policy_snapshots:
            self.save_snapshot(policy)
            with policy.graph.as_default():
                weights = policy.tfvars.get_weights()
                self.current_policy_snapshot = weights

        self.policies[name_behavior_id] = policy

        if not self.learning_policy_name:
            self.set_learning_policy(name_behavior_id)

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
            self.trainer.set_policy(self.learning_policy_name, policy)
        except KeyError:
            pass

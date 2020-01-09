# # Unity ML-Agents Toolkit
# ## ML-Agent Learning (Ghost Trainer)

# import logging
from typing import Dict

import numpy as np

from mlagents.trainers.brain import BrainParameters
from mlagents.trainers.policy import Policy
from mlagents.trainers.tf_policy import TFPolicy

from mlagents.trainers.trainer import Trainer
from mlagents.trainers.trajectory import Trajectory
from mlagents.trainers.agent_processor import AgentManagerQueue

from mlagents.trainers.ghost.tf_utils import TensorFlowVariables

# logger = logging.getLogger("mlagents.trainers")


class GhostTrainer(Trainer):
    def __init__(
        self, trainer, brain_name, reward_buff_cap, trainer_parameters, training, run_id
    ):
        """
        Responsible for collecting experiences and training trainer model via self_play.
        :param trainer: The trainer of the policy/policies being trained with self_play
        :param brain_name: The name of the brain associated with trainer config
        :param reward_buff_cap: Max reward history to track in the reward buffer
        :param trainer_parameters: The parameters for the trainer (dictionary).
        :param training: Whether the trainer is set for training.
        :param run_id: The identifier of the current run
        """

        super(GhostTrainer, self).__init__(
            brain_name, trainer_parameters, training, run_id, reward_buff_cap
        )

        self.trainer = trainer

        # assign ghost's stats collection to wrapped trainer's
        self.stats_reporter = self.trainer.stats_reporter

        self.policies: Dict[str, TFPolicy] = {}
        self.policy_snapshots = []
        self.learning_behavior_name: str = None
        self.current_policy_snapshot = None
        self.last_step = 0
        self_play_parameters = trainer_parameters["ghost"]

        self.window = self_play_parameters["window"]
        self.current_prob = self_play_parameters["current_prob"]
        self.steps_between_snapshots = self_play_parameters["snapshot_per"]

        self.initial_elo: float = 1200.0
        self.current_elo: float = self.initial_elo
        self.policy_elos: List[float] = []
        self.current_opponent: int = 0

    #def _write_summary(self, step: int) -> int:
    #@    self.trainer._write_summary(step)

    @property
    def get_step(self) -> int:
        """
        Returns the number of steps the trainer has performed
        :return: the step count of the trainer
        """
        return self.trainer.get_step

    def _process_trajectory(self, trajectory: Trajectory) -> None:
        pass
        #self._increment_step(len(trajectory.steps), trajectory.behavior_id)
        #self.trainer.process_trajectory(trajectory)

    def _is_ready_update(self) -> bool:
        pass
        #return self.trainer._is_ready_update()

    #def _increment_step(self, n_steps: int, name_behavior_id: str) -> None:
    #    self.trainer._increment_step(n_steps, name_behavior_id)

    def _update_policy(self) -> None:
        pass
    #    self.trainer.update_policy()

    #    with self.trainer.policy.graph.as_default():
    #        weights = self.trainer.policy.tfvars.get_weights()
    #        self.current_policy_snapshot = weights

    #    if self.get_step - self.last_step > self.steps_between_snapshots:
    #        self.save_snapshot(self.trainer.policy)
    #        self.last_step = self.get_step
    def advance(self) -> None:
        self.trainer.advance()
        if self.get_step - self.last_step > self.steps_between_snapshots:
            self.save_snapshot(self.trainer.policy)
            self.last_step = self.get_step

    def end_episode(self):
        self.trainer.end_episode()

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

        if not self.learning_behavior_name:
            self.trainer.add_policy(name_behavior_id, policy)
            self.learning_behavior_name = name_behavior_id

    def get_policy(self, name_behavior_id: str) -> TFPolicy:
        return self.policies[name_behavior_id]

    def save_snapshot(self, policy: TFPolicy) -> None:
        with policy.graph.as_default():
            weights = policy.tfvars.get_weights()
            self.policy_snapshots.append(weights)
        if len(self.policy_snapshots) > self.window:
            del self.policy_snapshots[0]
        self.policy_elos[-1] = self.initial_elo

    def swap_snapshots(self) -> None:
        for name_behavior_id, policy in self.policies.items():
            # here is the place for a sampling protocol
            if name_behavior_id == self.learning_behavior_name and self.is_training:
                continue
            elif np.random.uniform() < (1 - self.current_prob):
                x = np.random.randint(len(self.policy_snapshots))
                snapshot = self.policy_snapshots[x]
            else:
                with self.trainer.policy.graph.as_default():
                    weights = self.trainer.policy.tfvars.get_weights()
                    self.current_policy_snapshot = weights
                snapshot = self.current_policy_snapshot
                x = "current"
            self.current_opponent = x
            print(
                "Step {}: Swapping snapshot {} to id {} with {} learning".format(
                    self.get_step, x, name_behavior_id, self.learning_behavior_name
                )
            )
            with policy.graph.as_default():
                policy.tfvars.set_weights(snapshot)

    def set_learning(self, training: bool) -> None:
        self.is_training = training
        self.swap_snapshots()

    def publish_policy_queue(self, policy_queue: AgentManagerQueue[Policy]) -> None:
        """
        Adds a policy queue to the list of queues to publish to when this Trainer
        makes a policy update
        :param queue: Policy queue to publish to.
        """
        if policy_queue.behavior_id == self.learning_behavior_name:
            self.trainer.policy_queues.append(policy_queue)

    def subscribe_trajectory_queue(
        self, trajectory_queue: AgentManagerQueue[Trajectory]
    ) -> None:
        if trajectory_queue.behavior_id == self.learning_behavior_name:
            self.trainer.subscribe_trajectory_queue(trajectory_queue)

#ELO calculation
#Taken from https://github.com/Unity-Technologies/ml-agents/pull/1975
def compute_elo_rating_changes(rating1, rating2, result):
    r1 = pow(10, rating1 / 400)
    r2 = pow(10, rating2 / 400)

    sum = r1 + r2
    e1 = r1 / sum

    s1 = 1 if result == "win" else 0

    change = K * (s1 - e1)

    return change

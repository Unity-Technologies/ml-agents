# # Unity ML-Agents Toolkit
# ## ML-Agent Learning (Ghost Trainer)

from typing import Deque, Dict, List, Any, cast

import numpy as np
import logging

from mlagents.trainers.brain import BrainParameters
from mlagents.trainers.policy import Policy
from mlagents.trainers.policy.tf_policy import TFPolicy

from mlagents.trainers.trainer import Trainer
from mlagents.trainers.trajectory import Trajectory
from mlagents.trainers.agent_processor import AgentManagerQueue
from mlagents.trainers.stats import StatsPropertyType
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers

logger = logging.getLogger("mlagents.trainers")


class GhostTrainer(Trainer):
    def __init__(
        self,
        trainer,
        brain_name,
        controller,
        reward_buff_cap,
        trainer_parameters,
        training,
        run_id,
    ):
        """
        Responsible for collecting experiences and training trainer model via self_play.
        :param trainer: The trainer of the policy/policies being trained with self_play
        :param brain_name: The name of the brain associated with trainer config
        :param controller: Object that coordinates all ghost trainers
        :param reward_buff_cap: Max reward history to track in the reward buffer
        :param trainer_parameters: The parameters for the trainer (dictionary).
        :param training: Whether the trainer is set for training.
        :param run_id: The identifier of the current run
        """

        super(GhostTrainer, self).__init__(
            brain_name, trainer_parameters, training, run_id, reward_buff_cap
        )

        self.trainer = trainer
        self.controller = controller

        self.internal_trajectory_queues: Dict[str, AgentManagerQueue[Trajectory]] = {}
        self._name_to_trajectory_queue: Dict[str, AgentManagerQueue[Trajectory]] = {}

        self.internal_policy_queues: Dict[str, AgentManagerQueue[Policy]] = {}
        self._name_to_policy_queue: Dict[str, AgentManagerQueue[Policy]] = {}

        self._name_to_parsed_behavior_id: Dict[str, BehaviorIdentifiers] = {}

        # assign ghost's stats collection to wrapped trainer's
        self._stats_reporter = self.trainer.stats_reporter
        # Set the logging to print ELO in the console
        self._stats_reporter.add_property(StatsPropertyType.SELF_PLAY, True)

        self_play_parameters = trainer_parameters["self_play"]
        self.window = self_play_parameters.get("window", 10)
        self.play_against_current_self_ratio = self_play_parameters.get(
            "play_against_current_self_ratio", 0.5
        )
        self.steps_between_save = self_play_parameters.get("save_steps", 20000)
        self.steps_between_swap = self_play_parameters.get("swap_steps", 20000)

        self.policies: Dict[str, TFPolicy] = {}
        self.policy_snapshots: List[Any] = []
        self.snapshot_counter: int = 0
        self.learning_team: int = None
        self.current_policy_snapshot = None
        self.last_save = 0
        self.last_swap = 0

        # Chosen because it is the initial ELO in Chess
        self.initial_elo: float = self_play_parameters.get("initial_elo", 1200.0)
        self.current_elo: float = self.initial_elo
        self.policy_elos: List[float] = [self.initial_elo] * (
            self.window + 1
        )  # for learning policy
        self.current_opponent: int = 0

    @property
    def get_step(self) -> int:
        """
        Returns the number of steps the trainer has performed
        :return: the step count of the trainer
        """
        return self.trainer.get_step

    @property
    def reward_buffer(self) -> Deque[float]:
        """
         Returns the reward buffer. The reward buffer contains the cumulative
         rewards of the most recent episodes completed by agents using this
         trainer.
         :return: the reward buffer.
         """
        return self.trainer.reward_buffer

    def _process_trajectory(self, trajectory: Trajectory) -> None:
        if trajectory.done_reached and not trajectory.max_step_reached:
            # Assumption is that final reward is 1/.5/0 for win/draw/loss
            final_reward = trajectory.steps[-1].reward
            result = 0.5
            if final_reward > 0:
                result = 1.0
            elif final_reward < 0:
                result = 0.0

            change = compute_elo_rating_changes(
                self.current_elo, self.policy_elos[self.current_opponent], result
            )
            self.current_elo += change
            self.policy_elos[self.current_opponent] -= change
            opponents = np.array(self.policy_elos, dtype=np.float32)
            self._stats_reporter.add_stat("Self-play/ELO", self.current_elo)
            self._stats_reporter.add_stat(
                "Self-play/Mean Opponent ELO", opponents.mean()
            )
            self._stats_reporter.add_stat("Self-play/Std Opponent ELO", opponents.std())

    def advance(self) -> None:
        """
        Steps the trainer, passing trajectories to wrapped trainer and calling trainer advance
        """
        for trajectory_queue in self.trajectory_queues:
            parsed_behavior_id = self._name_to_parsed_behavior_id[
                trajectory_queue.behavior_id
            ]
            if parsed_behavior_id.team_id == self.learning_team:
                # With a future multiagent trainer, this will be indexed by 'role'
                internal_trajectory_queue = self.internal_trajectory_queues[
                    parsed_behavior_id.brain_name
                ]
                try:
                    # We grab at most the maximum length of the queue.
                    # This ensures that even if the queue is being filled faster than it is
                    # being emptied, the trajectories in the queue are on-policy.
                    for _ in range(trajectory_queue.maxlen):
                        t = trajectory_queue.get_nowait()
                        # adds to wrapped trainers queue
                        internal_trajectory_queue.put(t)
                        self._process_trajectory(t)
                except AgentManagerQueue.Empty:
                    pass
            else:
                # Dump trajectories from non-learning policy
                try:
                    for _ in range(trajectory_queue.maxlen):
                        trajectory_queue.get_nowait()
                except AgentManagerQueue.Empty:
                    pass

        self.next_summary_step = self.trainer.next_summary_step
        self.trainer.advance()

        for policy_queue in self.policy_queues:
            parsed_behavior_id = self._name_to_parsed_behavior_id[
                policy_queue.behavior_id
            ]
            if parsed_behavior_id.team_id == self.learning_team:
                # With a future multiagent trainer, this will be indexed by 'role'
                internal_policy_queue = self.internal_policy_queues[
                    parsed_behavior_id.brain_name
                ]
                # Get policies that correspond to the policy queue in question
                try:
                    policy = cast(TFPolicy, internal_policy_queue.get_nowait())
                    self.current_policy_snapshot = policy.get_weights()
                    policy_queue.put(policy)
                except AgentManagerQueue.Empty:
                    pass

        if self.get_step - self.last_save > self.steps_between_save:
            self._save_snapshot(self.trainer.policy)
            self.last_save = self.get_step

        if self.get_step - self.last_swap > self.steps_between_swap:
            self._swap_snapshots()
            self.last_swap = self.get_step

        self.learning_team = self.controller.get_learning_team(self.get_step)

    def end_episode(self):
        self.trainer.end_episode()

    def save_model(self, name_behavior_id: str) -> None:
        self.trainer.save_model(name_behavior_id)

    def export_model(self, name_behavior_id: str) -> None:
        self.trainer.export_model(name_behavior_id)

    def create_policy(self, brain_parameters: BrainParameters) -> TFPolicy:
        return self.trainer.create_policy(brain_parameters)

    def add_policy(
        self, parsed_behavior_id: BehaviorIdentifiers, policy: TFPolicy
    ) -> None:
        """
        Adds policy to trainer. For the first policy added, add a trainer
        to the policy and set the learning behavior name to name_behavior_id.
        :param name_behavior_id: Behavior ID that the policy should belong to.
        :param policy: Policy to associate with name_behavior_id.
        """
        name_behavior_id = parsed_behavior_id.behavior_id
        team_id = parsed_behavior_id.team_id
        self.controller.subscribe_team_id(team_id)
        self.policies[name_behavior_id] = policy
        policy.create_tf_graph()

        self._name_to_parsed_behavior_id[name_behavior_id] = parsed_behavior_id

        # First policy encountered
        if not self.learning_team:
            weights = policy.get_weights()
            self.current_policy_snapshot = weights
            self.trainer.add_policy(parsed_behavior_id, policy)
            self._save_snapshot(policy)  # Need to save after trainer initializes policy
            self.learning_team = team_id
            self._stats_reporter.add_property(StatsPropertyType.SELF_PLAY_TEAM, team_id)
        else:
            # for saving/swapping snapshots
            policy.init_load_weights()

    def get_policy(self, name_behavior_id: str) -> TFPolicy:
        return self.policies[name_behavior_id]

    def _save_snapshot(self, policy: TFPolicy) -> None:
        weights = policy.get_weights()
        try:
            self.policy_snapshots[self.snapshot_counter] = weights
        except IndexError:
            self.policy_snapshots.append(weights)
        self.policy_elos[self.snapshot_counter] = self.current_elo
        self.snapshot_counter = (self.snapshot_counter + 1) % self.window

    def _swap_snapshots(self) -> None:
        for policy_queue in self.policy_queues:
            parsed_behavior_id = self._name_to_parsed_behavior_id[
                policy_queue.behavior_id
            ]
            # here is the place for a sampling protocol
            if parsed_behavior_id.team_id == self.learning_team:
                continue
            elif np.random.uniform() < (1 - self.play_against_current_self_ratio):
                x = np.random.randint(len(self.policy_snapshots))
                snapshot = self.policy_snapshots[x]
            else:
                snapshot = self.current_policy_snapshot
                x = "current"
                self.policy_elos[-1] = self.current_elo
            self.current_opponent = -1 if x == "current" else x
            logger.debug(
                "Step {}: Swapping snapshot {} to id {} with {} learning".format(
                    self.get_step, x, parsed_behavior_id.behavior_id, self.learning_team
                )
            )
            policy = self.get_policy(parsed_behavior_id.behavior_id)
            policy.load_weights(snapshot)
            policy_queue.put(policy)

    def publish_policy_queue(self, policy_queue: AgentManagerQueue[Policy]) -> None:
        """
        Adds a policy queue to the list of queues to publish to when this Trainer
        makes a policy update
        :param queue: Policy queue to publish to.
        """
        super().publish_policy_queue(policy_queue)
        parsed_behavior_id = self._name_to_parsed_behavior_id[policy_queue.behavior_id]
        self._name_to_policy_queue[parsed_behavior_id.behavior_id] = policy_queue
        if parsed_behavior_id.team_id == self.learning_team:

            internal_policy_queue: AgentManagerQueue[Policy] = AgentManagerQueue(
                parsed_behavior_id.brain_name
            )

            self.internal_policy_queues[
                parsed_behavior_id.brain_name
            ] = internal_policy_queue
            self.trainer.publish_policy_queue(internal_policy_queue)

    def subscribe_trajectory_queue(
        self, trajectory_queue: AgentManagerQueue[Trajectory]
    ) -> None:
        """
        Adds a trajectory queue to the list of queues for the trainer to ingest Trajectories from.
        :param queue: Trajectory queue to publish to.
        """
        super().subscribe_trajectory_queue(trajectory_queue)
        parsed_behavior_id = self._name_to_parsed_behavior_id[
            trajectory_queue.behavior_id
        ]
        self._name_to_trajectory_queue[
            parsed_behavior_id.behavior_id
        ] = trajectory_queue

        if parsed_behavior_id.team_id == self.learning_team:
            # With a future multiagent trainer, this will be indexed by 'role'
            internal_trajectory_queue: AgentManagerQueue[
                Trajectory
            ] = AgentManagerQueue(parsed_behavior_id.brain_name)

            self.internal_trajectory_queues[
                parsed_behavior_id.brain_name
            ] = internal_trajectory_queue
            self.trainer.subscribe_trajectory_queue(internal_trajectory_queue)


# Taken from https://github.com/Unity-Technologies/ml-agents/pull/1975 and
# https://metinmediamath.wordpress.com/2013/11/27/how-to-calculate-the-elo-rating-including-example/
# ELO calculation


def compute_elo_rating_changes(rating1: float, rating2: float, result: float) -> float:
    r1 = pow(10, rating1 / 400)
    r2 = pow(10, rating2 / 400)

    summed = r1 + r2
    e1 = r1 / summed

    change = result - e1
    return change

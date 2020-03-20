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
    """
    The GhostTrainer trains agents in adversarial games (there are teams in opposition) using a self-play mechanism.
    In adversarial settings with self-play, at any time, there is only a single learning team. The other team(s) is
    "ghosted" which means that its agents are executing fixed policies and not learning. The GhostTrainer wraps
    a standard RL trainer which trains the learning team and ensures that only the trajectories collected
    by the learning team are used for training.  The GhostTrainer also maintains past policy snapshots to be used
    as the fixed policies when the team is not learning. The GhostTrainer is 1:1 with brain_names as the other
    trainers, and is responsible for one or more teams. Note, a GhostTrainer can have only one team in
    asymmetric games where there is only one team with a particular behavior i.e. Hide and Seek.
    The GhostController manages high level coordination between multiple ghost trainers. The learning team id
    is cycled throughout a training run.
    """

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
        Creates a GhostTrainer.
        :param trainer: The trainer of the policy/policies being trained with self_play
        :param brain_name: The name of the brain associated with trainer config
        :param controller: GhostController that coordinates all ghost trainers and calculates ELO
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

        self._internal_trajectory_queues: Dict[str, AgentManagerQueue[Trajectory]] = {}
        self._internal_policy_queues: Dict[str, AgentManagerQueue[Policy]] = {}

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
        self.ghost_step: int = 0

        self.policies: Dict[str, TFPolicy] = {}
        self.policy_snapshots: List[Any] = []
        self.snapshot_counter: int = 0

        # wrapped_training_team and learning team need to be separate
        # in the situation where new agents are created destroyed
        # after learning team switches. These agents need to be added
        # to trainers properly.
        self._learning_team: int = None
        self.wrapped_trainer_team: int = None
        self.current_policy_snapshot = None
        self.last_save = 0
        self.last_swap = 0

        # Chosen because it is the initial ELO in Chess
        self.initial_elo: float = self_play_parameters.get("initial_elo", 1200.0)
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

    @property
    def current_elo(self) -> float:
        """
        Gets ELO of current policy which is always last in the list
        :return: ELO of current policy
        """
        return self.policy_elos[-1]

    def change_current_elo(self, change: float) -> None:
        """
        Changes elo of current policy which is always last in the list
        :param change: Amount to change current elo by
        """
        self.policy_elos[-1] += change

    def get_opponent_elo(self) -> float:
        """
        Get elo of current opponent policy
        :return: ELO of current opponent policy
        """
        return self.policy_elos[self.current_opponent]

    def change_opponent_elo(self, change: float) -> None:
        """
        Changes elo of current opponent policy
        :param change: Amount to change current opponent elo by
        """
        self.policy_elos[self.current_opponent] -= change

    def _process_trajectory(self, trajectory: Trajectory) -> None:
        """
        Determines the final result of an episode and asks the GhostController
        to calculate the ELO change. The GhostController changes the ELO
        of the opponent policy since this may be in a different GhostTrainer
        i.e. in asymmetric games. We assume the last reward determines the winner.
        :param trajectory: Trajectory.
        """
        if trajectory.done_reached and not trajectory.max_step_reached:
            # Assumption is that final reward is 1/.5/0 for win/draw/loss
            final_reward = trajectory.steps[-1].reward
            result = 0.5
            if final_reward > 0:
                result = 1.0
            elif final_reward < 0:
                result = 0.0

            change = self.controller.compute_elo_rating_changes(
                self.current_elo, result
            )
            self.change_current_elo(change)
            self._stats_reporter.add_stat("Self-play/ELO", self.current_elo)

    def advance(self) -> None:
        """
        Steps the trainer, passing trajectories to wrapped trainer and calling trainer advance
        """
        for trajectory_queue in self.trajectory_queues:
            parsed_behavior_id = self._name_to_parsed_behavior_id[
                trajectory_queue.behavior_id
            ]
            if parsed_behavior_id.team_id == self._learning_team:
                # With a future multiagent trainer, this will be indexed by 'role'
                internal_trajectory_queue = self._internal_trajectory_queues[
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
                        t = trajectory_queue.get_nowait()
                        # count ghost steps
                        self.ghost_step += len(t.steps)
                except AgentManagerQueue.Empty:
                    pass

        self.next_summary_step = self.trainer.next_summary_step
        self.trainer.advance()

        for policy_queue in self.policy_queues:
            parsed_behavior_id = self._name_to_parsed_behavior_id[
                policy_queue.behavior_id
            ]
            if parsed_behavior_id.team_id == self._learning_team:
                # With a future multiagent trainer, this will be indexed by 'role'
                internal_policy_queue = self._internal_policy_queues[
                    parsed_behavior_id.brain_name
                ]
                # Get policies that correspond to the policy queue in question
                try:
                    policy = cast(TFPolicy, internal_policy_queue.get_nowait())
                    self.current_policy_snapshot = policy.get_weights()
                    policy_queue.put(policy)
                except AgentManagerQueue.Empty:
                    pass

        self._learning_team = self.controller.get_learning_team(self.ghost_step)

        # Note save and swap should be on different step counters.
        # We don't want to save unless the policy is learning.
        if self.get_step - self.last_save > self.steps_between_save:
            self._save_snapshot(self.trainer.policy)
            self.last_save = self.get_step

        if self.ghost_step - self.last_swap > self.steps_between_swap:
            self._swap_snapshots()
            self.last_swap = self.ghost_step

    def end_episode(self):
        """
        Forwarding call to wrapped trainers end_episode
        """
        self.trainer.end_episode()

    def save_model(self, name_behavior_id: str) -> None:
        """
        Forwarding call to wrapped trainers save_model
        """
        self.trainer.save_model(name_behavior_id)

    def export_model(self, name_behavior_id: str) -> None:
        """
        Forwarding call to wrapped trainers export_model
        """
        self.trainer.export_model(name_behavior_id)

    def create_policy(self, brain_parameters: BrainParameters) -> TFPolicy:
        """
        Creates policy with the wrapped trainer's create_policy function
        """
        return self.trainer.create_policy(brain_parameters)

    def add_policy(
        self, parsed_behavior_id: BehaviorIdentifiers, policy: TFPolicy
    ) -> None:
        """
        Adds policy to trainer. The first policy encountered sets the wrapped
        trainer team.  This is to ensure that all agents from the same multi-agent
        team are grouped. All policies associated with this team are added to the
        wrapped trainer to be trained.
        :param name_behavior_id: Behavior ID that the policy should belong to.
        :param policy: Policy to associate with name_behavior_id.
        """
        name_behavior_id = parsed_behavior_id.behavior_id
        team_id = parsed_behavior_id.team_id
        self.controller.subscribe_team_id(team_id, self)
        self.policies[name_behavior_id] = policy
        policy.create_tf_graph()

        self._name_to_parsed_behavior_id[name_behavior_id] = parsed_behavior_id

        # First policy or a new agent on the same team encountered
        if self.wrapped_trainer_team is None or team_id == self.wrapped_trainer_team:
            weights = policy.get_weights()
            self.current_policy_snapshot = weights
            self.trainer.add_policy(parsed_behavior_id, policy)
            self._save_snapshot(policy)  # Need to save after trainer initializes policy
            self._learning_team = self.controller.get_learning_team(self.ghost_step)
            self.wrapped_trainer_team = team_id
        else:
            # for saving/swapping snapshots
            policy.init_load_weights()

    def get_policy(self, name_behavior_id: str) -> TFPolicy:
        """
        Gets policy associated with name_behavior_id
        :param name_behavior_id: Fully qualified behavior name
        :return: Policy associated with name_behavior_id
        """
        return self.policies[name_behavior_id]

    def _save_snapshot(self, policy: TFPolicy) -> None:
        """
        Saves a snapshot of the weights of the policy and maintains the policy_snapshots
        according to the window size
        :param policy: The policy to be snapshotted
        """
        weights = policy.get_weights()
        try:
            self.policy_snapshots[self.snapshot_counter] = weights
        except IndexError:
            self.policy_snapshots.append(weights)
        self.policy_elos[self.snapshot_counter] = self.current_elo
        self.snapshot_counter = (self.snapshot_counter + 1) % self.window

    def _swap_snapshots(self) -> None:
        """
        Swaps the appropriate weight to the policy and pushes it to respective policy queues
        """

        for policy_queue in self.policy_queues:
            parsed_behavior_id = self._name_to_parsed_behavior_id[
                policy_queue.behavior_id
            ]
            # Here is the place for a sampling protocol. If the learning team switches
            # immediately before swapping, this first check ensures that the new learning
            # team gets the current policy snapshot. Otherwise, it redundantly swaps
            # the current policy.
            if parsed_behavior_id.team_id != self._learning_team and np.random.uniform() < (
                1 - self.play_against_current_self_ratio
            ):
                x = np.random.randint(len(self.policy_snapshots))
                snapshot = self.policy_snapshots[x]
            else:
                snapshot = self.current_policy_snapshot
                x = "current"
            self.current_opponent = -1 if x == "current" else x
            logger.debug(
                "Step {}: Swapping snapshot {} to id {} with {} learning".format(
                    self.ghost_step,
                    x,
                    parsed_behavior_id.behavior_id,
                    self._learning_team,
                )
            )
            policy = self.get_policy(parsed_behavior_id.behavior_id)
            policy.load_weights(snapshot)
            policy_queue.put(policy)

    def publish_policy_queue(self, policy_queue: AgentManagerQueue[Policy]) -> None:
        """
        Adds a policy queue for every member of the team to the list of queues to publish to when this Trainer
        makes a policy update.  Creates an internal policy queue for the wrapped
        trainer to push to.  The GhostTrainer pushes all policies to the env.
        :param queue: Policy queue to publish to.
        """
        super().publish_policy_queue(policy_queue)
        parsed_behavior_id = self._name_to_parsed_behavior_id[policy_queue.behavior_id]
        if parsed_behavior_id.team_id == self.wrapped_trainer_team:
            # With a future multiagent trainer, this will be indexed by 'role'
            internal_policy_queue: AgentManagerQueue[Policy] = AgentManagerQueue(
                parsed_behavior_id.brain_name
            )

            self._internal_policy_queues[
                parsed_behavior_id.brain_name
            ] = internal_policy_queue
            self.trainer.publish_policy_queue(internal_policy_queue)

    def subscribe_trajectory_queue(
        self, trajectory_queue: AgentManagerQueue[Trajectory]
    ) -> None:
        """
        Adds a trajectory queue for every member of the team to the list of queues for the trainer
        to ingest Trajectories from. Creates an internal trajectory queue to push trajectories from
        the learning team.  The wrapped trainer subscribes to this queue.
        :param queue: Trajectory queue to publish to.
        """
        super().subscribe_trajectory_queue(trajectory_queue)
        parsed_behavior_id = self._name_to_parsed_behavior_id[
            trajectory_queue.behavior_id
        ]
        if parsed_behavior_id.team_id == self.wrapped_trainer_team:
            # With a future multiagent trainer, this will be indexed by 'role'
            internal_trajectory_queue: AgentManagerQueue[
                Trajectory
            ] = AgentManagerQueue(parsed_behavior_id.brain_name)

            self._internal_trajectory_queues[
                parsed_behavior_id.brain_name
            ] = internal_trajectory_queue
            self.trainer.subscribe_trajectory_queue(internal_trajectory_queue)

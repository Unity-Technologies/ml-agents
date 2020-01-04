# # Unity ML-Agents Toolkit
import logging
from typing import Dict, List, Any
from collections import defaultdict
import abc
import time

from mlagents.tf_utils import tf
from mlagents.trainers.tf_policy import TFPolicy
from mlagents_envs.timers import set_gauge
from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.trajectory import Trajectory
from mlagents.trainers.trainer import Trainer, UnityTrainerException
from mlagents.trainers.agent_processor import AgentManagerQueue
from mlagents.trainers.components.reward_signals import RewardSignalResult
from mlagents_envs.timers import hierarchical_timer

LOGGER = logging.getLogger("mlagents.trainers")

RewardSignalResults = Dict[str, RewardSignalResult]


class RLTrainer(Trainer, abc.ABC):  # pylint: disable=abstract-method
    """
    This class is the base class for trainers that use Reward Signals.
    """

    def __init__(self, *args, **kwargs):
        super(RLTrainer, self).__init__(*args, **kwargs)
        self.param_keys: List[str] = []
        self.step: int = 0
        self.training_start_time = time.time()
        self.summary_freq = self.trainer_parameters["summary_freq"]
        self.next_update_step = self.summary_freq
        # Make sure we have at least one reward_signal
        if not self.trainer_parameters["reward_signals"]:
            raise UnityTrainerException(
                "No reward signals were defined. At least one must be used with {}.".format(
                    self.__class__.__name__
                )
            )
        # collected_rewards is a dictionary from name of reward signal to a dictionary of agent_id to cumulative reward
        # used for reporting only. We always want to report the environment reward to Tensorboard, regardless
        # of what reward signals are actually present.
        self.collected_rewards: Dict[str, Dict[str, int]] = {
            "environment": defaultdict(lambda: 0)
        }
        self.update_buffer: AgentBuffer = AgentBuffer()
        # Write hyperparameters to Tensorboard
        if self.is_training:
            self.write_tensorboard_text("Hyperparameters", self.trainer_parameters)

    def _check_param_keys(self):
        for k in self.param_keys:
            if k not in self.trainer_parameters:
                raise UnityTrainerException(
                    "The hyper-parameter {0} could not be found for the {1} trainer of "
                    "brain {2}.".format(k, self.__class__, self.brain_name)
                )

    def _increment_step(self, n_steps: int, name_behavior_id: str) -> None:
        """
        Increment the step count of the trainer
        :param n_steps: number of steps to increment the step count by
        """
        self.step += n_steps
        self.next_update_step = self.step + (
            self.summary_freq - self.step % self.summary_freq
        )
        p = self.get_policy(name_behavior_id)
        if p:
            p.increment_step(n_steps)

    def _write_summary(self, step: int) -> None:
        """
        Saves training statistics to Tensorboard.
        """
        is_training = "Training." if self.training_progress < 1.0 else "Not Training."
        stats_summary = self.stats_reporter.get_stats_summaries(
            "Environment/Cumulative Reward"
        )
        if stats_summary.num > 0:
            LOGGER.info(
                " {}: {}: Step: {}. "
                "Time Elapsed: {:0.3f} s "
                "Mean "
                "Reward: {:0.3f}"
                ". Std of Reward: {:0.3f}. {}".format(
                    self.run_id,
                    self.brain_name,
                    step,
                    time.time() - self.training_start_time,
                    stats_summary.mean,
                    stats_summary.std,
                    is_training,
                )
            )
            set_gauge(f"{self.brain_name}.mean_reward", stats_summary.mean)
        else:
            LOGGER.info(
                " {}: {}: Step: {}. No episode was completed since last summary. {}".format(
                    self.run_id, self.brain_name, step, is_training
                )
            )
        self.stats_reporter.write_stats(int(step))

    def _maybe_write_summary(self, step_after_process: int) -> None:
        """
        If processing the trajectory will make the step exceed the next summary write,
        write the summary. This logic ensures summaries are written on the update step and not in between.
        :param step_after_process: the step count after processing the next trajectory.
        """
        if step_after_process >= self.next_update_step and self.step != 0:
            self._write_summary(self.next_update_step)

    def end_episode(self) -> None:
        """
        A signal that the Episode has ended. The buffer must be reset.
        Get only called when the academy resets.
        """
        for rewards in self.collected_rewards.values():
            for agent_id in rewards:
                rewards[agent_id] = 0

    def _update_end_episode_stats(self, agent_id: str, policy: TFPolicy) -> None:
        for name, rewards in self.collected_rewards.items():
            if name == "environment":
                self.reward_buffer.appendleft(rewards.get(agent_id, 0))
                rewards[agent_id] = 0
            else:
                self.stats_reporter.add_stat(
                    policy.reward_signals[name].stat_name, rewards.get(agent_id, 0)
                )
                rewards[agent_id] = 0

    def clear_update_buffer(self) -> None:
        """
        Clear the buffers that have been built up during inference.
        """
        self.update_buffer.reset_agent()

    def advance(self) -> None:
        """
        Steps the trainer, taking in trajectories and updates if ready
        """
        with hierarchical_timer("process_trajectory"):
            for traj_queue in self.trajectory_queues:
                try:
                    t = traj_queue.get_nowait()
                    self._process_trajectory(t)
                except AgentManagerQueue.Empty:
                    pass
        if self.training_progress < 1.0:
            if self._is_ready_update():
                with hierarchical_timer("update_policy"):
                    self._update_policy()
                    for q in self.policy_queues:
                        # Get policies that correspond to the policy queue in question
                        q.put(self.get_policy(q.behavior_id))
        if not self.is_training:
            self.clear_update_buffer()

    @property
    def training_progress(self) -> float:
        """
        Returns a float between 0 and 1 indicating how far along in the training progress the Trainer is.
        If 1, the Trainer wasn't training to begin with, or max_steps
        is reached.
        """
        if self.is_training:
            return min(self.step / float(self.trainer_parameters["max_steps"]), 1.0)
        else:
            return 1.0

    def write_tensorboard_text(self, key: str, input_dict: Dict[str, Any]) -> None:
        """
        Saves text to Tensorboard.
        Note: Only works on tensorflow r1.2 or above.
        :param key: The name of the text.
        :param input_dict: A dictionary that will be displayed in a table on Tensorboard.
        """
        try:
            with tf.Session() as sess:
                s_op = tf.summary.text(
                    key,
                    tf.convert_to_tensor(
                        ([[str(x), str(input_dict[x])] for x in input_dict])
                    ),
                )
                s = sess.run(s_op)
                self.stats_reporter.write_text(s, self.step)
        except Exception:
            LOGGER.info("Could not write text summary for Tensorboard.")
            pass

    def save_model(self, name_behavior_id: str) -> None:
        """
        Saves the model
        """
        self.get_policy(name_behavior_id).save_model(self.step)

    def export_model(self, name_behavior_id: str) -> None:
        """
        Exports the model
        """
        self.get_policy(name_behavior_id).export_model()

    @abc.abstractmethod
    def _update_policy(self) -> None:
        """
        Uses update buffer to update the policy.
        The reward signal generators must be updated in this method at their own pace.
        """
        pass

    @abc.abstractmethod
    def _process_trajectory(self, trajectory: Trajectory) -> None:
        """
        Takes a trajectory and processes it, putting it into the update buffer.
        :param trajectory: The Trajectory tuple containing the steps to be processed.
        """
        self._maybe_write_summary(self.step + len(trajectory.steps))
        self._increment_step(len(trajectory.steps), trajectory.behavior_id)

    @abc.abstractmethod
    def _is_ready_update(self):
        """
        Returns whether or not the trainer has enough elements to run update model
        :return: A boolean corresponding to wether or not update_model() can be run
        """
        return False

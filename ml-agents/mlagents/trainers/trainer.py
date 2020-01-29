# # Unity ML-Agents Toolkit
import logging
from typing import Dict, List, Deque, Any
import time
import abc

from mlagents.tf_utils import tf
from mlagents import tf_utils

from collections import deque

from mlagents_envs.exception import UnityException
from mlagents_envs.timers import set_gauge
from mlagents.trainers.tf_policy import TFPolicy
from mlagents.trainers.stats import StatsReporter
from mlagents.trainers.trajectory import Trajectory
from mlagents.trainers.agent_processor import AgentManagerQueue
from mlagents.trainers.brain import BrainParameters
from mlagents.trainers.policy import Policy
from mlagents_envs.timers import hierarchical_timer

LOGGER = logging.getLogger("mlagents.trainers")


class UnityTrainerException(UnityException):
    """
    Related to errors with the Trainer.
    """

    pass


class Trainer(abc.ABC):
    """This class is the base class for the mlagents_envs.trainers"""

    def __init__(
        self,
        brain_name: str,
        trainer_parameters: dict,
        training: bool,
        run_id: str,
        reward_buff_cap: int = 1,
    ):
        """
        Responsible for collecting experiences and training a neural network model.
        :BrainParameters brain: Brain to be trained.
        :dict trainer_parameters: The parameters for the trainer (dictionary).
        :bool training: Whether the trainer is set for training.
        :str run_id: The identifier of the current run
        :int reward_buff_cap:
        """
        self.param_keys: List[str] = []
        self.brain_name = brain_name
        self.run_id = run_id
        self.trainer_parameters = trainer_parameters
        self.summary_path = trainer_parameters["summary_path"]
        self.stats_reporter = StatsReporter(self.summary_path)
        self.cumulative_returns_since_policy_update: List[float] = []
        self.is_training = training
        self._reward_buffer: Deque[float] = deque(maxlen=reward_buff_cap)
        self.policy_queues: List[AgentManagerQueue[Policy]] = []
        self.trajectory_queues: List[AgentManagerQueue[Trajectory]] = []
        self.step: int = 0
        self.training_start_time = time.time()
        self.summary_freq = self.trainer_parameters["summary_freq"]
        self.next_summary_step = self.summary_freq

    def _check_param_keys(self):
        for k in self.param_keys:
            if k not in self.trainer_parameters:
                raise UnityTrainerException(
                    "The hyper-parameter {0} could not be found for the {1} trainer of "
                    "brain {2}.".format(k, self.__class__, self.brain_name)
                )

    def write_tensorboard_text(self, key: str, input_dict: Dict[str, Any]) -> None:
        """
        Saves text to Tensorboard.
        Note: Only works on tensorflow r1.2 or above.
        :param key: The name of the text.
        :param input_dict: A dictionary that will be displayed in a table on Tensorboard.
        """
        try:
            with tf.Session(config=tf_utils.generate_session_config()) as sess:
                s_op = tf.summary.text(
                    key,
                    tf.convert_to_tensor(
                        ([[str(x), str(input_dict[x])] for x in input_dict])
                    ),
                )
                s = sess.run(s_op)
                self.stats_reporter.write_text(s, self.get_step)
        except Exception:
            LOGGER.info("Could not write text summary for Tensorboard.")
            pass

    def _dict_to_str(self, param_dict: Dict[str, Any], num_tabs: int) -> str:
        """
        Takes a parameter dictionary and converts it to a human-readable string.
        Recurses if there are multiple levels of dict. Used to print out hyperaparameters.
        param: param_dict: A Dictionary of key, value parameters.
        return: A string version of this dictionary.
        """
        if not isinstance(param_dict, dict):
            return str(param_dict)
        else:
            append_newline = "\n" if num_tabs > 0 else ""
            return append_newline + "\n".join(
                [
                    "\t"
                    + "  " * num_tabs
                    + "{0}:\t{1}".format(
                        x, self._dict_to_str(param_dict[x], num_tabs + 1)
                    )
                    for x in param_dict
                ]
            )

    def __str__(self) -> str:
        return """Hyperparameters for the {0} of brain {1}: \n{2}""".format(
            self.__class__.__name__,
            self.brain_name,
            self._dict_to_str(self.trainer_parameters, 0),
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        """
        Returns the trainer parameters of the trainer.
        """
        return self.trainer_parameters

    @property
    def get_max_steps(self) -> int:
        """
        Returns the maximum number of steps. Is used to know when the trainer should be stopped.
        :return: The maximum number of steps of the trainer
        """
        return int(float(self.trainer_parameters["max_steps"]))

    @property
    def get_step(self) -> int:
        """
        Returns the number of steps the trainer has performed
        :return: the step count of the trainer
        """
        return self.step

    @property
    def should_still_train(self) -> bool:
        """
        Returns whether or not the trainer should train. A Trainer could
        stop training if it wasn't training to begin with, or if max_steps
        is reached.
        """
        return self.is_training and self.get_step <= self.get_max_steps

    @property
    def reward_buffer(self) -> Deque[float]:
        """
        Returns the reward buffer. The reward buffer contains the cumulative
        rewards of the most recent episodes completed by agents using this
        trainer.
        :return: the reward buffer.
        """
        return self._reward_buffer

    def _increment_step(self, n_steps: int, name_behavior_id: str) -> None:
        """
        Increment the step count of the trainer
        :param n_steps: number of steps to increment the step count by
        """
        self.step += n_steps
        self.next_summary_step = self._get_next_summary_step()
        p = self.get_policy(name_behavior_id)
        if p:
            p.increment_step(n_steps)

    def _get_next_summary_step(self) -> int:
        """
        Get the next step count that should result in a summary write.
        """
        return self.step + (self.summary_freq - self.step % self.summary_freq)

    def save_model(self, name_behavior_id: str) -> None:
        """
        Saves the model
        """
        self.get_policy(name_behavior_id).save_model(self.get_step)

    def export_model(self, name_behavior_id: str) -> None:
        """
        Exports the model
        """
        self.get_policy(name_behavior_id).export_model()

    def _write_summary(self, step: int) -> None:
        """
        Saves training statistics to Tensorboard.
        """
        is_training = "Training." if self.should_still_train else "Not Training."
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

    @abc.abstractmethod
    def _process_trajectory(self, trajectory: Trajectory) -> None:
        """
        Takes a trajectory and processes it, putting it into the update buffer.
        :param trajectory: The Trajectory tuple containing the steps to be processed.
        """
        self._maybe_write_summary(self.get_step + len(trajectory.steps))
        self._increment_step(len(trajectory.steps), trajectory.behavior_id)

    def _maybe_write_summary(self, step_after_process: int) -> None:
        """
        If processing the trajectory will make the step exceed the next summary write,
        write the summary. This logic ensures summaries are written on the update step and not in between.
        :param step_after_process: the step count after processing the next trajectory.
        """
        if step_after_process >= self.next_summary_step and self.get_step != 0:
            self._write_summary(self.next_summary_step)

    @abc.abstractmethod
    def end_episode(self):
        """
        A signal that the Episode has ended. The buffer must be reset.
        Get only called when the academy resets.
        """
        pass

    @abc.abstractmethod
    def create_policy(self, brain_parameters: BrainParameters) -> TFPolicy:
        """
        Creates policy
        """
        pass

    @abc.abstractmethod
    def add_policy(self, name_behavior_id: str, policy: TFPolicy) -> None:
        """
        Adds policy to trainer
        """
        pass

    @abc.abstractmethod
    def get_policy(self, name_behavior_id: str) -> TFPolicy:
        """
        Gets policy from trainer
        """
        pass

    @abc.abstractmethod
    def _is_ready_update(self):
        """
        Returns whether or not the trainer has enough elements to run update model
        :return: A boolean corresponding to wether or not update_model() can be run
        """
        return False

    @abc.abstractmethod
    def _update_policy(self):
        """
        Uses demonstration_buffer to update model.
        """
        pass

    def advance(self) -> None:
        """
        Steps the trainer, taking in trajectories and updates if ready.
        """
        with hierarchical_timer("process_trajectory"):
            for traj_queue in self.trajectory_queues:
                # We grab at most the maximum length of the queue.
                # This ensures that even if the queue is being filled faster than it is
                # being emptied, the trajectories in the queue are on-policy.
                for _ in range(traj_queue.maxlen):
                    try:
                        t = traj_queue.get_nowait()
                        self._process_trajectory(t)
                    except AgentManagerQueue.Empty:
                        break
        if self.should_still_train:
            if self._is_ready_update():
                with hierarchical_timer("_update_policy"):
                    self._update_policy()
                    for q in self.policy_queues:
                        # Get policies that correspond to the policy queue in question
                        q.put(self.get_policy(q.behavior_id))

    def publish_policy_queue(self, policy_queue: AgentManagerQueue[Policy]) -> None:
        """
        Adds a policy queue to the list of queues to publish to when this Trainer
        makes a policy update
        :param queue: Policy queue to publish to.
        """
        self.policy_queues.append(policy_queue)

    def subscribe_trajectory_queue(
        self, trajectory_queue: AgentManagerQueue[Trajectory]
    ) -> None:
        """
        Adds a trajectory queue to the list of queues for the trainer to ingest Trajectories from.
        :param queue: Trajectory queue to publish to.
        """
        self.trajectory_queues.append(trajectory_queue)

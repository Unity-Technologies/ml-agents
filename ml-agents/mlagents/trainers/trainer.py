# # Unity ML-Agents Toolkit
import logging
from typing import Dict, List, Deque, Any

from mlagents.tf_utils import tf

from collections import deque

from mlagents_envs.exception import UnityException
from mlagents_envs.timers import set_gauge
from mlagents.trainers.trainer_metrics import TrainerMetrics
from mlagents.trainers.tf_policy import TFPolicy
from mlagents.trainers.stats import StatsReporter
from mlagents.trainers.trajectory import Trajectory
from mlagents.trainers.brain import BrainParameters

LOGGER = logging.getLogger("mlagents.trainers")


class UnityTrainerException(UnityException):
    """
    Related to errors with the Trainer.
    """

    pass


class Trainer(object):
    """This class is the base class for the mlagents_envs.trainers"""

    def __init__(
        self,
        brain: BrainParameters,
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
        self.brain_name = brain.brain_name
        self.run_id = run_id
        self.trainer_parameters = trainer_parameters
        self.summary_path = trainer_parameters["summary_path"]
        self.stats_reporter = StatsReporter(self.summary_path)
        self.cumulative_returns_since_policy_update: List[float] = []
        self.is_training = training
        self.trainer_metrics = TrainerMetrics(
            path=self.summary_path + ".csv", brain_name=self.brain_name
        )
        self._reward_buffer: Deque[float] = deque(maxlen=reward_buff_cap)
        self.policy: TFPolicy = None  # type: ignore  # this will always get set
        self.step: int = 0

    def check_param_keys(self):
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
            with tf.Session() as sess:
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

    def dict_to_str(self, param_dict: Dict[str, Any], num_tabs: int) -> str:
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
                        x, self.dict_to_str(param_dict[x], num_tabs + 1)
                    )
                    for x in param_dict
                ]
            )

    def __str__(self) -> str:
        return """Hyperparameters for the {0} of brain {1}: \n{2}""".format(
            self.__class__.__name__,
            self.brain_name,
            self.dict_to_str(self.trainer_parameters, 0),
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        """
        Returns the trainer parameters of the trainer.
        """
        return self.trainer_parameters

    @property
    def get_max_steps(self) -> float:
        """
        Returns the maximum number of steps. Is used to know when the trainer should be stopped.
        :return: The maximum number of steps of the trainer
        """
        return float(self.trainer_parameters["max_steps"])

    @property
    def get_step(self) -> int:
        """
        Returns the number of steps the trainer has performed
        :return: the step count of the trainer
        """
        return self.step

    @property
    def reward_buffer(self) -> Deque[float]:
        """
        Returns the reward buffer. The reward buffer contains the cumulative
        rewards of the most recent episodes completed by agents using this
        trainer.
        :return: the reward buffer.
        """
        return self._reward_buffer

    def increment_step(self, n_steps: int) -> None:
        """
        Increment the step count of the trainer

        :param n_steps: number of steps to increment the step count by
        """
        self.step = self.policy.increment_step(n_steps)

    def save_model(self) -> None:
        """
        Saves the model
        """
        self.policy.save_model(self.get_step)

    def export_model(self) -> None:
        """
        Exports the model
        """
        self.policy.export_model()

    def write_training_metrics(self) -> None:
        """
        Write training metrics to a CSV  file
        :return:
        """
        self.trainer_metrics.write_training_metrics()

    def write_summary(self, global_step: int, delta_train_start: float) -> None:
        """
        Saves training statistics to Tensorboard.
        :param delta_train_start:  Time elapsed since training started.
        :param global_step: The number of steps the simulation has been going for
        """
        if (
            global_step % self.trainer_parameters["summary_freq"] == 0
            and global_step != 0
        ):
            is_training = (
                "Training."
                if self.is_training and self.get_step <= self.get_max_steps
                else "Not Training."
            )
            step = min(self.get_step, self.get_max_steps)
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
                        delta_train_start,
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

    def process_trajectory(self, trajectory: Trajectory) -> None:
        """
        Takes a trajectory and processes it, putting it into the update buffer.
        Processing involves calculating value and advantage targets for model updating step.
        :param trajectory: The Trajectory tuple containing the steps to be processed.
        """
        raise UnityTrainerException(
            "The process_experiences method was not implemented."
        )

    def end_episode(self):
        """
        A signal that the Episode has ended. The buffer must be reset.
        Get only called when the academy resets.
        """
        raise UnityTrainerException("The end_episode method was not implemented.")

    def is_ready_update(self):
        """
        Returns whether or not the trainer has enough elements to run update model
        :return: A boolean corresponding to wether or not update_model() can be run
        """
        raise UnityTrainerException("The is_ready_update method was not implemented.")

    def update_policy(self):
        """
        Uses demonstration_buffer to update model.
        """
        raise UnityTrainerException("The update_model method was not implemented.")

    def advance(self) -> None:
        pass

# # Unity ML-Agents Toolkit
import logging
from typing import Dict, List, Deque, Any
import os
import tensorflow as tf
import numpy as np
from collections import deque, defaultdict

from mlagents.envs.action_info import ActionInfoOutputs
from mlagents.envs.exception import UnityException
from mlagents.envs.timers import set_gauge
from mlagents.trainers.trainer_metrics import TrainerMetrics
from mlagents.trainers.tf_policy import TFPolicy
from mlagents.envs.brain import BrainParameters, AllBrainInfo

LOGGER = logging.getLogger("mlagents.trainers")


class UnityTrainerException(UnityException):
    """
    Related to errors with the Trainer.
    """

    pass


class Trainer(object):
    """This class is the base class for the mlagents.envs.trainers"""

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
        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path)
        self.cumulative_returns_since_policy_update: List[float] = []
        self.is_training = training
        self.stats: Dict[str, List] = defaultdict(list)
        self.trainer_metrics = TrainerMetrics(
            path=self.summary_path + ".csv", brain_name=self.brain_name
        )
        self.summary_writer = tf.summary.FileWriter(self.summary_path)
        self._reward_buffer: Deque[float] = deque(maxlen=reward_buff_cap)
        self.policy: TFPolicy = None
        self.step: int = 0

    def check_param_keys(self):
        for k in self.param_keys:
            if k not in self.trainer_parameters:
                raise UnityTrainerException(
                    "The hyper-parameter {0} could not be found for the {1} trainer of "
                    "brain {2}.".format(k, self.__class__, self.brain_name)
                )

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

    def write_summary(
        self, global_step: int, delta_train_start: float, lesson_num: int = 0
    ) -> None:
        """
        Saves training statistics to Tensorboard.
        :param delta_train_start:  Time elapsed since training started.
        :param lesson_num: Current lesson number in curriculum.
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
            if len(self.stats["Environment/Cumulative Reward"]) > 0:
                mean_reward = np.mean(self.stats["Environment/Cumulative Reward"])
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
                        mean_reward,
                        np.std(self.stats["Environment/Cumulative Reward"]),
                        is_training,
                    )
                )
                set_gauge(f"{self.brain_name}.mean_reward", mean_reward)
            else:
                LOGGER.info(
                    " {}: {}: Step: {}. No episode was completed since last summary. {}".format(
                        self.run_id, self.brain_name, step, is_training
                    )
                )
            summary = tf.Summary()
            for key in self.stats:
                if len(self.stats[key]) > 0:
                    stat_mean = float(np.mean(self.stats[key]))
                    summary.value.add(tag="{}".format(key), simple_value=stat_mean)
                    self.stats[key] = []
            summary.value.add(tag="Environment/Lesson", simple_value=lesson_num)
            self.summary_writer.add_summary(summary, step)
            self.summary_writer.flush()

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
                self.summary_writer.add_summary(s, self.get_step)
        except Exception:
            LOGGER.info(
                "Cannot write text summary for Tensorboard. Tensorflow version must be r1.2 or above."
            )
            pass

    def add_experiences(
        self,
        curr_all_info: AllBrainInfo,
        next_all_info: AllBrainInfo,
        take_action_outputs: ActionInfoOutputs,
    ) -> None:
        """
        Adds experiences to each agent's experience history.
        :param curr_all_info: Dictionary of all current brains and corresponding BrainInfo.
        :param next_all_info: Dictionary of all current brains and corresponding BrainInfo.
        :param take_action_outputs: The outputs of the Policy's get_action method.
        """
        raise UnityTrainerException(
            "The process_experiences method was not implemented."
        )

    def process_experiences(
        self, current_info: AllBrainInfo, next_info: AllBrainInfo
    ) -> None:
        """
        Checks agent histories for processing condition, and processes them as necessary.
        Processing involves calculating value and advantage targets for model updating step.
        :param current_info: Dictionary of all current-step brains and corresponding BrainInfo.
        :param next_info: Dictionary of all next-step brains and corresponding BrainInfo.
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

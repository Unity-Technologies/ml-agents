# # Unity ML-Agents Toolkit
# ## ML-Agent Learning
"""Launches trainers for each External Brains in a Unity Environment."""

import os
import logging
import shutil
import sys
from typing import *

import numpy as np
import tensorflow as tf
from time import time

from mlagents.envs import AllBrainInfo, BrainParameters
from mlagents.envs.base_unity_environment import BaseUnityEnvironment
from mlagents.envs.exception import UnityEnvironmentException
from mlagents.trainers import Trainer
from mlagents.trainers.ppo.trainer import PPOTrainer
from mlagents.trainers.bc.offline_trainer import OfflineBCTrainer
from mlagents.trainers.bc.online_trainer import OnlineBCTrainer
from mlagents.trainers.meta_curriculum import MetaCurriculum


class TrainerController(object):
    def __init__(
        self,
        model_path: str,
        summaries_dir: str,
        run_id: str,
        save_freq: int,
        meta_curriculum: Optional[MetaCurriculum],
        load: bool,
        train: bool,
        keep_checkpoints: int,
        lesson: Optional[int],
        external_brains: Dict[str, BrainParameters],
        training_seed: int,
        fast_simulation: bool,
    ):
        """
        :param model_path: Path to save the model.
        :param summaries_dir: Folder to save training summaries.
        :param run_id: The sub-directory name for model and summary statistics
        :param save_freq: Frequency at which to save model
        :param meta_curriculum: MetaCurriculum object which stores information about all curricula.
        :param load: Whether to load the model or randomly initialize.
        :param train: Whether to train model, or only run inference.
        :param keep_checkpoints: How many model checkpoints to keep.
        :param lesson: Start learning from this lesson.
        :param external_brains: dictionary of external brain names to BrainInfo objects.
        :param training_seed: Seed to use for Numpy and Tensorflow random number generation.
        """

        self.model_path = model_path
        self.summaries_dir = summaries_dir
        self.external_brains = external_brains
        self.external_brain_names = external_brains.keys()
        self.logger = logging.getLogger("mlagents.envs")
        self.run_id = run_id
        self.save_freq = save_freq
        self.lesson = lesson
        self.load_model = load
        self.train_model = train
        self.keep_checkpoints = keep_checkpoints
        self.trainers: Dict[str, Trainer] = {}
        self.trainer_metrics: Dict[str, TrainerMetrics] = {}
        self.global_step = 0
        self.meta_curriculum = meta_curriculum
        self.seed = training_seed
        self.training_start_time = time()
        self.fast_simulation = fast_simulation
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

    def _get_measure_vals(self):
        if self.meta_curriculum:
            brain_names_to_measure_vals = {}
            for (
                brain_name,
                curriculum,
            ) in self.meta_curriculum.brains_to_curriculums.items():
                if curriculum.measure == "progress":
                    measure_val = (
                        self.trainers[brain_name].get_step
                        / self.trainers[brain_name].get_max_steps
                    )
                    brain_names_to_measure_vals[brain_name] = measure_val
                elif curriculum.measure == "reward":
                    measure_val = np.mean(self.trainers[brain_name].reward_buffer)
                    brain_names_to_measure_vals[brain_name] = measure_val
            return brain_names_to_measure_vals
        else:
            return None

    def _save_model(self, steps=0):
        """
        Saves current model to checkpoint folder.
        :param steps: Current number of steps in training process.
        :param saver: Tensorflow saver for session.
        """
        for brain_name in self.trainers.keys():
            self.trainers[brain_name].save_model()
        self.logger.info("Saved Model")

    def _save_model_when_interrupted(self, steps=0):
        self.logger.info(
            "Learning was interrupted. Please wait " "while the graph is generated."
        )
        self._save_model(steps)

    def _write_training_metrics(self):
        """
        Write all CSV metrics
        :return:
        """
        for brain_name in self.trainers.keys():
            if brain_name in self.trainer_metrics:
                self.trainers[brain_name].write_training_metrics()

    def _export_graph(self):
        """
        Exports latest saved models to .nn format for Unity embedding.
        """
        for brain_name in self.trainers.keys():
            self.trainers[brain_name].export_model()

    def initialize_trainers(self, trainer_config: Dict[str, Dict[str, str]]):
        """
        Initialization of the trainers
        :param trainer_config: The configurations of the trainers
        """
        trainer_parameters_dict = {}
        for brain_name in self.external_brains:
            trainer_parameters = trainer_config["default"].copy()
            trainer_parameters["summary_path"] = "{basedir}/{name}".format(
                basedir=self.summaries_dir, name=str(self.run_id) + "_" + brain_name
            )
            trainer_parameters["model_path"] = "{basedir}/{name}".format(
                basedir=self.model_path, name=brain_name
            )
            trainer_parameters["keep_checkpoints"] = self.keep_checkpoints
            if brain_name in trainer_config:
                _brain_key = brain_name
                while not isinstance(trainer_config[_brain_key], dict):
                    _brain_key = trainer_config[_brain_key]
                for k in trainer_config[_brain_key]:
                    trainer_parameters[k] = trainer_config[_brain_key][k]
            trainer_parameters_dict[brain_name] = trainer_parameters.copy()
        for brain_name in self.external_brains:
            if trainer_parameters_dict[brain_name]["trainer"] == "offline_bc":
                self.trainers[brain_name] = OfflineBCTrainer(
                    self.external_brains[brain_name],
                    trainer_parameters_dict[brain_name],
                    self.train_model,
                    self.load_model,
                    self.seed,
                    self.run_id,
                )
            elif trainer_parameters_dict[brain_name]["trainer"] == "online_bc":
                self.trainers[brain_name] = OnlineBCTrainer(
                    self.external_brains[brain_name],
                    trainer_parameters_dict[brain_name],
                    self.train_model,
                    self.load_model,
                    self.seed,
                    self.run_id,
                )
            elif trainer_parameters_dict[brain_name]["trainer"] == "ppo":
                self.trainers[brain_name] = PPOTrainer(
                    self.external_brains[brain_name],
                    self.meta_curriculum.brains_to_curriculums[
                        brain_name
                    ].min_lesson_length
                    if self.meta_curriculum
                    else 0,
                    trainer_parameters_dict[brain_name],
                    self.train_model,
                    self.load_model,
                    self.seed,
                    self.run_id,
                )
                self.trainer_metrics[brain_name] = self.trainers[
                    brain_name
                ].trainer_metrics
            else:
                raise UnityEnvironmentException(
                    "The trainer config contains "
                    "an unknown trainer type for "
                    "brain {}".format(brain_name)
                )

    @staticmethod
    def _create_model_path(model_path):
        try:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
        except Exception:
            raise UnityEnvironmentException(
                "The folder {} containing the "
                "generated model could not be "
                "accessed. Please make sure the "
                "permissions are set correctly.".format(model_path)
            )

    def _reset_env(self, env: BaseUnityEnvironment):
        """Resets the environment.

        Returns:
            A Data structure corresponding to the initial reset state of the
            environment.
        """
        if self.meta_curriculum is not None:
            return env.reset(
                train_mode=self.fast_simulation,
                config=self.meta_curriculum.get_config(),
            )
        else:
            return env.reset(train_mode=self.fast_simulation)

    def start_learning(self, env: BaseUnityEnvironment, trainer_config):
        # TODO: Should be able to start learning at different lesson numbers
        # for each curriculum.
        if self.meta_curriculum is not None:
            self.meta_curriculum.set_all_curriculums_to_lesson_num(self.lesson)
        self._create_model_path(self.model_path)

        tf.reset_default_graph()

        # Prevent a single session from taking all GPU memory.
        self.initialize_trainers(trainer_config)
        for _, t in self.trainers.items():
            self.logger.info(t)

        if self.train_model:
            for brain_name, trainer in self.trainers.items():
                trainer.write_tensorboard_text("Hyperparameters", trainer.parameters)
        try:
            curr_info = self._reset_env(env)
            while (
                any([t.get_step <= t.get_max_steps for k, t in self.trainers.items()])
                or not self.train_model
            ):
                new_info = self.take_step(env, curr_info)
                self.global_step += 1
                if (
                    self.global_step % self.save_freq == 0
                    and self.global_step != 0
                    and self.train_model
                ):
                    # Save Tensorflow model
                    self._save_model(steps=self.global_step)
                curr_info = new_info
            # Final save Tensorflow model
            if self.global_step != 0 and self.train_model:
                self._save_model(steps=self.global_step)
        except KeyboardInterrupt:
            if self.train_model:
                self._save_model_when_interrupted(steps=self.global_step)
            pass
        env.close()
        if self.train_model:
            self._write_training_metrics()
            self._export_graph()

    def take_step(self, env: BaseUnityEnvironment, curr_info: AllBrainInfo):
        if self.meta_curriculum:
            # Get the sizes of the reward buffers.
            reward_buff_sizes = {
                k: len(t.reward_buffer) for (k, t) in self.trainers.items()
            }
            # Attempt to increment the lessons of the brains who
            # were ready.
            lessons_incremented = self.meta_curriculum.increment_lessons(
                self._get_measure_vals(), reward_buff_sizes=reward_buff_sizes
            )
        else:
            lessons_incremented = {}

        # If any lessons were incremented or the environment is
        # ready to be reset
        if self.meta_curriculum and any(lessons_incremented.values()):
            curr_info = self._reset_env(env)
            for brain_name, trainer in self.trainers.items():
                trainer.end_episode()
            for brain_name, changed in lessons_incremented.items():
                if changed:
                    self.trainers[brain_name].reward_buffer.clear()

        # Decide and take an action
        take_action_vector = {}
        take_action_memories = {}
        take_action_text = {}
        take_action_value = {}
        take_action_outputs = {}
        for brain_name, trainer in self.trainers.items():
            action_info = trainer.get_action(curr_info[brain_name])
            take_action_vector[brain_name] = action_info.action
            take_action_memories[brain_name] = action_info.memory
            take_action_text[brain_name] = action_info.text
            take_action_value[brain_name] = action_info.value
            take_action_outputs[brain_name] = action_info.outputs
        time_start_step = time()
        new_info = env.step(
            vector_action=take_action_vector,
            memory=take_action_memories,
            text_action=take_action_text,
            value=take_action_value,
        )
        delta_time_step = time() - time_start_step
        for brain_name, trainer in self.trainers.items():
            if brain_name in self.trainer_metrics:
                self.trainer_metrics[brain_name].add_delta_step(delta_time_step)
            trainer.add_experiences(
                curr_info, new_info, take_action_outputs[brain_name]
            )
            trainer.process_experiences(curr_info, new_info)
            if (
                trainer.is_ready_update()
                and self.train_model
                and trainer.get_step <= trainer.get_max_steps
            ):
                # Perform gradient descent with experience buffer

                trainer.update_policy()
            # Write training statistics to Tensorboard.
            delta_train_start = time() - self.training_start_time
            if self.meta_curriculum is not None:
                trainer.write_summary(
                    self.global_step,
                    delta_train_start,
                    lesson_num=self.meta_curriculum.brains_to_curriculums[
                        brain_name
                    ].lesson_num,
                )
            else:
                trainer.write_summary(self.global_step, delta_train_start)
            if self.train_model and trainer.get_step <= trainer.get_max_steps:
                trainer.increment_step_and_update_last_reward()
        return new_info

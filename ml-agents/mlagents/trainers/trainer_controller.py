# # Unity ML-Agents Toolkit
# ## ML-Agent Learning
"""Launches trainers for each External Brains in a Unity Environment."""

import json
import logging
from typing import *

import numpy as np
import tensorflow as tf
from time import time

from mlagents.envs.env_manager import StepInfo
from mlagents.envs.env_manager import EnvManager
from mlagents.envs.timers import hierarchical_timer, get_timer_tree, timed
from mlagents.trainers import Trainer, TrainerMetrics
from mlagents.trainers.meta_curriculum import MetaCurriculum
from mlagents.trainers.session_config import SessionConfig


class TrainerController(object):
    def __init__(
        self,
        trainers: Dict[str, Trainer],
        sess_config: SessionConfig,
        training_seed: int,
        meta_curriculum: MetaCurriculum,
    ):
        """
        :param trainers: Trainers we're training in this session.
        :param sess_config: Config options for the training session.
        :param training_seed: Seed to use for Numpy and Tensorflow random number generation.
        :param meta_curriculum: MetaCurriculum object which stores information about all curricula.
        """
        self.trainers = trainers
        self.meta_curriculum = meta_curriculum
        self.summaries_dir = sess_config.summaries_dir
        self.logger = logging.getLogger("mlagents.envs")
        self.run_id = sess_config.sub_run_id
        self.save_freq = sess_config.save_freq
        self.lesson = sess_config.lesson
        self.train_model = sess_config.train_model
        self.keep_checkpoints = sess_config.keep_checkpoints
        self.trainer_metrics: Dict[str, TrainerMetrics] = {}
        self.training_start_time = time()
        self.fast_simulation = sess_config.fast_simulation
        np.random.seed(training_seed)
        tf.set_random_seed(training_seed)

    def _get_measure_vals(self):
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

    def _save_model(self):
        """
        Saves current model to checkpoint folder.
        :param steps: Current number of steps in training process.
        :param saver: Tensorflow saver for session.
        """
        for brain_name in self.trainers.keys():
            self.trainers[brain_name].save_model()
        self.logger.info("Saved Model")

    def _save_model_when_interrupted(self):
        self.logger.info(
            "Learning was interrupted. Please wait while the graph is generated."
        )
        self._save_model()

    def _write_training_metrics(self):
        """
        Write all CSV metrics
        :return:
        """
        for brain_name in self.trainers.keys():
            if brain_name in self.trainer_metrics:
                self.trainers[brain_name].write_training_metrics()

    def _write_timing_tree(self) -> None:
        timing_path = f"{self.summaries_dir}/{self.run_id}_timers.json"
        try:
            with open(timing_path, "w") as f:
                json.dump(get_timer_tree(), f, indent=2)
        except FileNotFoundError:
            self.logger.warning(
                f"Unable to save to {timing_path}. Make sure the directory exists"
            )

    def _export_graph(self):
        """
        Exports latest saved models to .nn format for Unity embedding.
        """
        for brain_name in self.trainers.keys():
            self.trainers[brain_name].export_model()

    def _reset_env(self, env: EnvManager) -> List[StepInfo]:
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

    def _should_save_model(self, global_step: int) -> bool:
        return (
            global_step % self.save_freq == 0 and global_step != 0 and self.train_model
        )

    def _not_done_training(self) -> bool:
        return (
            any([t.get_step <= t.get_max_steps for k, t in self.trainers.items()])
            or not self.train_model
        )

    def write_to_tensorboard(self, global_step: int) -> None:
        for brain_name, trainer in self.trainers.items():
            # Write training statistics to Tensorboard.
            delta_train_start = time() - self.training_start_time
            if self.meta_curriculum is not None:
                trainer.write_summary(
                    global_step,
                    delta_train_start,
                    lesson_num=self.meta_curriculum.brains_to_curriculums[
                        brain_name
                    ].lesson_num,
                )
            else:
                trainer.write_summary(global_step, delta_train_start)

    def start_learning(self, env_manager: EnvManager) -> None:
        tf.reset_default_graph()

        for _, t in self.trainers.items():
            self.logger.info(t)

        global_step = 0

        if self.train_model:
            for brain_name, trainer in self.trainers.items():
                trainer.write_tensorboard_text("Hyperparameters", trainer.parameters)
        try:
            for brain_name, trainer in self.trainers.items():
                self.trainer_metrics[brain_name] = self.trainers[
                    brain_name
                ].trainer_metrics
                env_manager.set_policy(brain_name, trainer.policy)
            self._reset_env(env_manager)
            while self._not_done_training():
                n_steps = self.advance(env_manager)
                for i in range(n_steps):
                    global_step += 1
                    if self._should_save_model(global_step):
                        # Save Tensorflow model
                        self._save_model()
                    self.write_to_tensorboard(global_step)
            # Final save Tensorflow model
            if global_step != 0 and self.train_model:
                self._save_model()
        except KeyboardInterrupt:
            if self.train_model:
                self._save_model_when_interrupted()
            pass
        env_manager.close()
        if self.train_model:
            self._write_training_metrics()
            self._export_graph()
        self._write_timing_tree()

    @timed
    def advance(self, env: EnvManager) -> int:
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
            self._reset_env(env)
            for brain_name, trainer in self.trainers.items():
                trainer.end_episode()
            for brain_name, changed in lessons_incremented.items():
                if changed:
                    self.trainers[brain_name].reward_buffer.clear()

        with hierarchical_timer("env_step"):
            time_start_step = time()
            new_step_infos = env.step()
            delta_time_step = time() - time_start_step

        for step_info in new_step_infos:
            for brain_name, trainer in self.trainers.items():
                if brain_name in self.trainer_metrics:
                    self.trainer_metrics[brain_name].add_delta_step(delta_time_step)
                trainer.add_experiences(
                    step_info.previous_all_brain_info,
                    step_info.current_all_brain_info,
                    step_info.brain_name_to_action_info[brain_name].outputs,
                )
                trainer.process_experiences(
                    step_info.previous_all_brain_info, step_info.current_all_brain_info
                )
        for brain_name, trainer in self.trainers.items():
            if brain_name in self.trainer_metrics:
                self.trainer_metrics[brain_name].add_delta_step(delta_time_step)
            if self.train_model and trainer.get_step <= trainer.get_max_steps:
                trainer.increment_step(len(new_step_infos))
                if trainer.is_ready_update():
                    # Perform gradient descent with experience buffer
                    with hierarchical_timer("update_policy"):
                        trainer.update_policy()
                    env.set_policy(brain_name, trainer.policy)
        return len(new_step_infos)

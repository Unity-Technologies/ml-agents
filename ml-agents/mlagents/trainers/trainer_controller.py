# # Unity ML-Agents Toolkit
# ## ML-Agent Learning
"""Launches trainers for each External Brains in a Unity Environment."""

import os
import json
import logging
from typing import Dict, List, Optional, Set

import numpy as np
import tensorflow as tf
from time import time

from mlagents.envs.env_manager import EnvironmentStep
from mlagents.envs.env_manager import EnvManager
from mlagents.envs.exception import (
    UnityEnvironmentException,
    UnityCommunicationException,
)
from mlagents.envs.sampler_class import SamplerManager
from mlagents.envs.timers import hierarchical_timer, get_timer_tree, timed
from mlagents.trainers.trainer import Trainer, TrainerMetrics
from mlagents.trainers.meta_curriculum import MetaCurriculum
from mlagents.trainers.trainer_util import TrainerFactory


class TrainerController(object):
    def __init__(
        self,
        trainer_factory: TrainerFactory,
        model_path: str,
        summaries_dir: str,
        run_id: str,
        save_freq: int,
        meta_curriculum: Optional[MetaCurriculum],
        train: bool,
        training_seed: int,
        fast_simulation: bool,
        sampler_manager: SamplerManager,
        resampling_interval: Optional[int],
    ):
        """
        :param model_path: Path to save the model.
        :param summaries_dir: Folder to save training summaries.
        :param run_id: The sub-directory name for model and summary statistics
        :param save_freq: Frequency at which to save model
        :param meta_curriculum: MetaCurriculum object which stores information about all curricula.
        :param train: Whether to train model, or only run inference.
        :param training_seed: Seed to use for Numpy and Tensorflow random number generation.
        :param sampler_manager: SamplerManager object handles samplers for resampling the reset parameters.
        :param resampling_interval: Specifies number of simulation steps after which reset parameters are resampled.
        """
        self.trainers: Dict[str, Trainer] = {}
        self.trainer_factory = trainer_factory
        self.model_path = model_path
        self.summaries_dir = summaries_dir
        self.logger = logging.getLogger("mlagents.envs")
        self.run_id = run_id
        self.save_freq = save_freq
        self.train_model = train
        self.trainer_metrics: Dict[str, TrainerMetrics] = {}
        self.meta_curriculum = meta_curriculum
        self.training_start_time = time()
        self.fast_simulation = fast_simulation
        self.sampler_manager = sampler_manager
        self.resampling_interval = resampling_interval
        np.random.seed(training_seed)
        tf.set_random_seed(training_seed)

    def _get_measure_vals(self):
        brain_names_to_measure_vals = {}
        if self.meta_curriculum:
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
        else:
            for brain_name, trainer in self.trainers.items():
                measure_val = np.mean(trainer.reward_buffer)
                brain_names_to_measure_vals[brain_name] = measure_val
        return brain_names_to_measure_vals

    def _save_model(self):
        """
        Saves current model to checkpoint folder.
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

    def _reset_env(self, env: EnvManager) -> List[EnvironmentStep]:
        """Resets the environment.

        Returns:
            A Data structure corresponding to the initial reset state of the
            environment.
        """
        sampled_reset_param = self.sampler_manager.sample_all()
        new_meta_curriculum_config = (
            self.meta_curriculum.get_config() if self.meta_curriculum else {}
        )
        sampled_reset_param.update(new_meta_curriculum_config)
        return env.reset(train_mode=self.fast_simulation, config=sampled_reset_param)

    def _should_save_model(self, global_step: int) -> bool:
        return (
            global_step % self.save_freq == 0 and global_step != 0 and self.train_model
        )

    def _not_done_training(self) -> bool:
        return (
            any(t.get_step <= t.get_max_steps for k, t in self.trainers.items())
            or not self.train_model
        ) or len(self.trainers) == 0

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

    def start_trainer(self, trainer: Trainer, env_manager: EnvManager) -> None:
        self.trainers[trainer.brain_name] = trainer
        self.logger.info(trainer)
        if self.train_model:
            trainer.write_tensorboard_text("Hyperparameters", trainer.parameters)
        env_manager.set_policy(trainer.brain_name, trainer.policy)

    def start_learning(self, env_manager: EnvManager) -> None:
        self._create_model_path(self.model_path)
        tf.reset_default_graph()
        global_step = 0
        last_brain_names: Set[str] = set()
        try:
            self._reset_env(env_manager)
            while self._not_done_training():
                external_brains = set(env_manager.external_brains.keys())
                new_brains = external_brains - last_brain_names
                if last_brain_names != env_manager.external_brains.keys():
                    for name in new_brains:
                        trainer = self.trainer_factory.generate(
                            env_manager.external_brains[name]
                        )
                        self.start_trainer(trainer, env_manager)
                    last_brain_names = external_brains
                n_steps = self.advance(env_manager)
                for i in range(n_steps):
                    global_step += 1
                    self.reset_env_if_ready(env_manager, global_step)
                    if self._should_save_model(global_step):
                        # Save Tensorflow model
                        self._save_model()
                    self.write_to_tensorboard(global_step)
            # Final save Tensorflow model
            if global_step != 0 and self.train_model:
                self._save_model()
        except (KeyboardInterrupt, UnityCommunicationException):
            if self.train_model:
                self._save_model_when_interrupted()
            pass
        if self.train_model:
            self._write_training_metrics()
            self._export_graph()
        self._write_timing_tree()
        env_manager.close()

    def end_trainer_episodes(
        self, env: EnvManager, lessons_incremented: Dict[str, bool]
    ) -> None:
        self._reset_env(env)
        # Reward buffers reset takes place only for curriculum learning
        # else no reset.
        for brain_name, trainer in self.trainers.items():
            trainer.end_episode()
        for brain_name, changed in lessons_incremented.items():
            if changed:
                self.trainers[brain_name].reward_buffer.clear()

    def reset_env_if_ready(self, env: EnvManager, steps: int) -> None:
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
        meta_curriculum_reset = any(lessons_incremented.values())
        # Check if we are performing generalization training and we have finished the
        # specified number of steps for the lesson
        generalization_reset = (
            not self.sampler_manager.is_empty()
            and (steps != 0)
            and (self.resampling_interval)
            and (steps % self.resampling_interval == 0)
        )
        if meta_curriculum_reset or generalization_reset:
            self.end_trainer_episodes(env, lessons_incremented)

    @timed
    def advance(self, env: EnvManager) -> int:
        with hierarchical_timer("env_step"):
            time_start_step = time()
            new_step_infos = env.step()
            delta_time_step = time() - time_start_step
        for step_info in new_step_infos:
            for brain_name, trainer in self.trainers.items():
                if brain_name in self.trainer_metrics:
                    self.trainer_metrics[brain_name].add_delta_step(delta_time_step)
                if brain_name in step_info.brain_name_to_action_info:
                    trainer.add_experiences(
                        step_info.previous_all_brain_info,
                        step_info.current_all_brain_info,
                        step_info.brain_name_to_action_info[brain_name].outputs,
                    )
                    trainer.process_experiences(
                        step_info.previous_all_brain_info,
                        step_info.current_all_brain_info,
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
            else:
                # Avoid memory leak during inference
                trainer.clear_update_buffer()
        return len(new_step_infos)

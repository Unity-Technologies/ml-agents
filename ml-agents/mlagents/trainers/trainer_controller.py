# # Unity ML-Agents Toolkit
# ## ML-Agent Learning
"""Launches trainers for each External Brains in a Unity Environment."""

import os
import sys
import json
import logging
from typing import Dict, List, Optional, Set
from collections import defaultdict

import numpy as np
from mlagents.tf_utils import tf

from mlagents.trainers.env_manager import EnvManager, EnvironmentStep
from mlagents_envs.exception import (
    UnityEnvironmentException,
    UnityCommunicationException,
)
from mlagents.trainers.sampler_class import SamplerManager
from mlagents_envs.timers import hierarchical_timer, get_timer_tree, timed
from mlagents.trainers.trainer import Trainer
from mlagents.trainers.meta_curriculum import MetaCurriculum
from mlagents.trainers.trainer_util import TrainerFactory
from mlagents.trainers.agent_processor import AgentManager, AgentManagerQueue


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
        self.brain_name_to_identifier: Dict[str, Set] = defaultdict(set)
        self.managers: Dict[str, AgentManager] = {}
        self.trainer_factory = trainer_factory
        self.model_path = model_path
        self.summaries_dir = summaries_dir
        self.logger = logging.getLogger("mlagents.trainers")
        self.run_id = run_id
        self.save_freq = save_freq
        self.train_model = train
        self.meta_curriculum = meta_curriculum
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
            ) in self.meta_curriculum.brains_to_curricula.items():
                # Skip brains that are in the metacurriculum but no trainer yet.
                if brain_name not in self.trainers:
                    continue
                if curriculum.measure == "progress":
                    measure_val = self.trainers[brain_name].get_step / float(
                        self.trainers[brain_name].get_max_steps
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
            for name_behavior_id in self.brain_name_to_identifier[brain_name]:
                self.trainers[brain_name].save_model(name_behavior_id)
        self.logger.info("Saved Model")

    def _save_model_when_interrupted(self):
        self.logger.info(
            "Learning was interrupted. Please wait while the graph is generated."
        )
        self._save_model()

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
            for name_behavior_id in self.brain_name_to_identifier[brain_name]:
                self.trainers[brain_name].export_model(name_behavior_id)

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
        return env.reset(config=sampled_reset_param)

    def _should_save_model(self, global_step: int) -> bool:
        return (
            global_step % self.save_freq == 0 and global_step != 0 and self.train_model
        )

    def _not_done_training(self) -> bool:
        return (
            any(t.should_still_train for t in self.trainers.values())
            or not self.train_model
        ) or len(self.trainers) == 0

    def _create_trainer_and_manager(
        self, env_manager: EnvManager, name_behavior_id: str
    ) -> None:
        try:
            brain_name, _ = name_behavior_id.split("?")
        except ValueError:
            brain_name = name_behavior_id

        try:
            trainer = self.trainers[brain_name]
        except KeyError:
            trainer = self.trainer_factory.generate(brain_name)
            self.trainers[brain_name] = trainer
            self.logger.info(trainer)
            if self.train_model:
                trainer.write_tensorboard_text("Hyperparameters", trainer.parameters)

        policy = trainer.create_policy(env_manager.external_brains[name_behavior_id])
        trainer.add_policy(name_behavior_id, policy)

        env_manager.set_policy(name_behavior_id, policy)

        self.brain_name_to_identifier[brain_name].add(name_behavior_id)

        agent_manager = AgentManager(
            policy,
            name_behavior_id,
            trainer.stats_reporter,
            trainer.parameters.get("time_horizon", sys.maxsize),
        )
        trainer.publish_policy_queue(agent_manager.policy_queue)
        trainer.subscribe_trajectory_queue(agent_manager.trajectory_queue)
        self.managers[name_behavior_id] = agent_manager

    def start_learning(self, env_manager: EnvManager) -> None:
        self._create_model_path(self.model_path)
        tf.reset_default_graph()
        global_step = 0
        last_brain_behavior_ids: Set[str] = set()
        try:
            self._reset_env(env_manager)
            while self._not_done_training():
                external_brain_behavior_ids = set(env_manager.external_brains.keys())
                new_behavior_ids = external_brain_behavior_ids - last_brain_behavior_ids
                for name_behavior_id in new_behavior_ids:
                    self._create_trainer_and_manager(env_manager, name_behavior_id)
                last_brain_behavior_ids = external_brain_behavior_ids
                n_steps = self.advance(env_manager)
                for _ in range(n_steps):
                    global_step += 1
                    self.reset_env_if_ready(env_manager, global_step)
                    if self._should_save_model(global_step):
                        # Save Tensorflow model
                        self._save_model()
            # Final save Tensorflow model
            if global_step != 0 and self.train_model:
                self._save_model()
        except (KeyboardInterrupt, UnityCommunicationException):
            if self.train_model:
                self._save_model_when_interrupted()
            pass
        if self.train_model:
            self._export_graph()
        self._write_timing_tree()

    def end_trainer_episodes(
        self, env: EnvManager, lessons_incremented: Dict[str, bool]
    ) -> None:
        self._reset_env(env)
        # Reward buffers reset takes place only for curriculum learning
        # else no reset.
        for trainer in self.trainers.values():
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

    def _get_and_process_experiences(self, env: EnvManager) -> int:
        with hierarchical_timer("env_step"):
            # Get new policies if found
            for brain_name in self.trainers.keys():
                for name_behavior_id in self.brain_name_to_identifier[brain_name]:
                    try:
                        _policy = self.managers[
                            name_behavior_id
                        ].policy_queue.get_nowait()
                        env.set_policy(name_behavior_id, _policy)
                    except AgentManagerQueue.Empty:
                        pass
            # Step the environment
            new_step_infos = env.step()
        # Add to AgentProcessor
        for step_info in new_step_infos:
            for name_behavior_id in step_info.name_behavior_ids:
                if name_behavior_id not in self.managers:
                    self.logger.warning(
                        "Agent manager was not created for behavior id {}.".format(
                            name_behavior_id
                        )
                    )
                    continue
                self.managers[name_behavior_id].add_experiences(
                    step_info.previous_all_brain_info[name_behavior_id],
                    step_info.current_all_brain_info[name_behavior_id],
                    step_info.brain_name_to_action_info[name_behavior_id].outputs,
                )
        return len(new_step_infos)

    @timed
    def advance(self, env: EnvManager) -> int:
        # Get steps
        num_steps = self._get_and_process_experiences(env)

        # Report current lesson
        if self.meta_curriculum:
            for brain_name, curr in self.meta_curriculum.brains_to_curricula.items():
                if brain_name in self.trainers:
                    self.trainers[brain_name].stats_reporter.set_stat(
                        "Environment/Lesson", curr.lesson_num
                    )

        # Advance trainers. This can be done in a separate loop in the future.
        with hierarchical_timer("trainer_advance"):
            for trainer in self.trainers.values():
                trainer.advance()

        return num_steps

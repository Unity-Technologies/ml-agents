# # Unity ML-Agents Toolkit
# ## ML-Agent Learning
"""Launches trainers for each External Brains in a Unity Environment."""

import os
import threading
from typing import Dict, Optional, Set, List
from collections import defaultdict

import numpy as np
from mlagents.tf_utils import tf

from mlagents_envs.logging_util import get_logger
from mlagents.trainers.env_manager import EnvManager
from mlagents_envs.exception import (
    UnityEnvironmentException,
    UnityCommunicationException,
    UnityCommunicatorStoppedException,
)
from mlagents.trainers.sampler_class import SamplerManager
from mlagents_envs.timers import (
    hierarchical_timer,
    timed,
    get_timer_stack_for_thread,
    merge_gauges,
)
from mlagents.trainers.trainer import Trainer
from mlagents.trainers.meta_curriculum import MetaCurriculum
from mlagents.trainers.trainer_util import TrainerFactory
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.agent_processor import AgentManager
from mlagents.trainers.settings import CurriculumSettings
from mlagents.trainers.training_status import GlobalTrainingStatus, StatusType


class TrainerController(object):
    def __init__(
        self,
        trainer_factory: TrainerFactory,
        output_path: str,
        run_id: str,
        meta_curriculum: Optional[MetaCurriculum],
        train: bool,
        training_seed: int,
        sampler_manager: SamplerManager,
        resampling_interval: Optional[int],
    ):
        """
        :param output_path: Path to save the model.
        :param summaries_dir: Folder to save training summaries.
        :param run_id: The sub-directory name for model and summary statistics
        :param meta_curriculum: MetaCurriculum object which stores information about all curricula.
        :param train: Whether to train model, or only run inference.
        :param training_seed: Seed to use for Numpy and Tensorflow random number generation.
        :param sampler_manager: SamplerManager object handles samplers for resampling the reset parameters.
        :param resampling_interval: Specifies number of simulation steps after which reset parameters are resampled.
        :param threaded: Whether or not to run trainers in a separate thread. Disable for testing/debugging.
        """
        self.trainers: Dict[str, Trainer] = {}
        self.brain_name_to_identifier: Dict[str, Set] = defaultdict(set)
        self.trainer_factory = trainer_factory
        self.output_path = output_path
        self.logger = get_logger(__name__)
        self.run_id = run_id
        self.train_model = train
        self.meta_curriculum = meta_curriculum
        self.sampler_manager = sampler_manager
        self.resampling_interval = resampling_interval
        self.ghost_controller = self.trainer_factory.ghost_controller

        self.trainer_threads: List[threading.Thread] = []
        self.kill_trainers = False
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
                if curriculum.measure == CurriculumSettings.MeasureType.PROGRESS:
                    measure_val = self.trainers[brain_name].get_step / float(
                        self.trainers[brain_name].get_max_steps
                    )
                    brain_names_to_measure_vals[brain_name] = measure_val
                elif curriculum.measure == CurriculumSettings.MeasureType.REWARD:
                    measure_val = np.mean(self.trainers[brain_name].reward_buffer)
                    brain_names_to_measure_vals[brain_name] = measure_val
        else:
            for brain_name, trainer in self.trainers.items():
                measure_val = np.mean(trainer.reward_buffer)
                brain_names_to_measure_vals[brain_name] = measure_val
        return brain_names_to_measure_vals

    @timed
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

    def _export_graph(self):
        """
        Exports latest saved models to .nn format for Unity embedding.
        """
        for brain_name in self.trainers.keys():
            for name_behavior_id in self.brain_name_to_identifier[brain_name]:
                self.trainers[brain_name].export_model(name_behavior_id)

    @staticmethod
    def _create_output_path(output_path):
        try:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
        except Exception:
            raise UnityEnvironmentException(
                f"The folder {output_path} containing the "
                "generated model could not be "
                "accessed. Please make sure the "
                "permissions are set correctly."
            )

    @timed
    def _reset_env(self, env: EnvManager) -> None:
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
        env.reset(config=sampled_reset_param)

    def _not_done_training(self) -> bool:
        return (
            any(t.should_still_train for t in self.trainers.values())
            or not self.train_model
        ) or len(self.trainers) == 0

    def _create_trainer_and_manager(
        self, env_manager: EnvManager, name_behavior_id: str
    ) -> None:

        parsed_behavior_id = BehaviorIdentifiers.from_name_behavior_id(name_behavior_id)
        brain_name = parsed_behavior_id.brain_name
        trainerthread = None
        try:
            trainer = self.trainers[brain_name]
        except KeyError:
            trainer = self.trainer_factory.generate(brain_name)
            self.trainers[brain_name] = trainer
            if trainer.threaded:
                # Only create trainer thread for new trainers
                trainerthread = threading.Thread(
                    target=self.trainer_update_func, args=(trainer,), daemon=True
                )
                self.trainer_threads.append(trainerthread)

        policy = trainer.create_policy(
            parsed_behavior_id, env_manager.external_brains[name_behavior_id]
        )
        trainer.add_policy(parsed_behavior_id, policy)

        agent_manager = AgentManager(
            policy,
            name_behavior_id,
            trainer.stats_reporter,
            trainer.parameters.time_horizon,
            threaded=trainer.threaded,
        )
        env_manager.set_agent_manager(name_behavior_id, agent_manager)
        env_manager.set_policy(name_behavior_id, policy)
        self.brain_name_to_identifier[brain_name].add(name_behavior_id)

        trainer.publish_policy_queue(agent_manager.policy_queue)
        trainer.subscribe_trajectory_queue(agent_manager.trajectory_queue)

        # Only start new trainers
        if trainerthread is not None:
            trainerthread.start()

    def _create_trainers_and_managers(
        self, env_manager: EnvManager, behavior_ids: Set[str]
    ) -> None:
        for behavior_id in behavior_ids:
            self._create_trainer_and_manager(env_manager, behavior_id)

    @timed
    def start_learning(self, env_manager: EnvManager) -> None:
        self._create_output_path(self.output_path)
        tf.reset_default_graph()
        global_step = 0
        last_brain_behavior_ids: Set[str] = set()
        try:
            # Initial reset
            self._reset_env(env_manager)
            while self._not_done_training():
                external_brain_behavior_ids = set(env_manager.external_brains.keys())
                new_behavior_ids = external_brain_behavior_ids - last_brain_behavior_ids
                self._create_trainers_and_managers(env_manager, new_behavior_ids)
                last_brain_behavior_ids = external_brain_behavior_ids
                n_steps = self.advance(env_manager)
                for _ in range(n_steps):
                    global_step += 1
                    self.reset_env_if_ready(env_manager, global_step)
            # Stop advancing trainers
            self.join_threads()
        except (
            KeyboardInterrupt,
            UnityCommunicationException,
            UnityEnvironmentException,
            UnityCommunicatorStoppedException,
        ) as ex:
            self.join_threads()
            self.logger.info(
                "Learning was interrupted. Please wait while the graph is generated."
            )
            if isinstance(ex, KeyboardInterrupt) or isinstance(
                ex, UnityCommunicatorStoppedException
            ):
                pass
            else:
                # If the environment failed, we want to make sure to raise
                # the exception so we exit the process with an return code of 1.
                raise ex
        finally:
            if self.train_model:
                self._save_model()
                self._export_graph()

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
        ghost_controller_reset = self.ghost_controller.should_reset()
        if meta_curriculum_reset or generalization_reset or ghost_controller_reset:
            self.end_trainer_episodes(env, lessons_incremented)

    @timed
    def advance(self, env: EnvManager) -> int:
        # Get steps
        with hierarchical_timer("env_step"):
            num_steps = env.advance()

        # Report current lesson
        if self.meta_curriculum:
            for brain_name, curr in self.meta_curriculum.brains_to_curricula.items():
                if brain_name in self.trainers:
                    self.trainers[brain_name].stats_reporter.set_stat(
                        "Environment/Lesson", curr.lesson_num
                    )
                    GlobalTrainingStatus.set_parameter_state(
                        brain_name, StatusType.LESSON_NUM, curr.lesson_num
                    )

        for trainer in self.trainers.values():
            if not trainer.threaded:
                with hierarchical_timer("trainer_advance"):
                    trainer.advance()

        return num_steps

    def join_threads(self, timeout_seconds: float = 1.0) -> None:
        """
        Wait for threads to finish, and merge their timer information into the main thread.
        :param timeout_seconds:
        :return:
        """
        self.kill_trainers = True
        for t in self.trainer_threads:
            try:
                t.join(timeout_seconds)
            except Exception:
                pass

        with hierarchical_timer("trainer_threads") as main_timer_node:
            for trainer_thread in self.trainer_threads:
                thread_timer_stack = get_timer_stack_for_thread(trainer_thread)
                if thread_timer_stack:
                    main_timer_node.merge(
                        thread_timer_stack.root,
                        root_name="thread_root",
                        is_parallel=True,
                    )
                    merge_gauges(thread_timer_stack.gauges)

    def trainer_update_func(self, trainer: Trainer) -> None:
        while not self.kill_trainers:
            with hierarchical_timer("trainer_advance"):
                trainer.advance()

# # Unity ML-Agents Toolkit
# ## ML-Agent Learning
"""Launches trainers for each External Brains in a Unity Environment."""
import multiprocessing
import os
import threading
from typing import Dict, Set, List
from collections import defaultdict

import numpy as np

from mlagents_envs.logging_util import get_logger
from mlagents.trainers.env_manager import EnvManager, EnvironmentStep
from mlagents.torch_utils import torch, mp
from mlagents_envs.exception import (
    UnityEnvironmentException,
    UnityCommunicationException,
    UnityCommunicatorStoppedException,
)
from mlagents_envs.timers import (
    hierarchical_timer,
    timed,
    get_timer_stack_for_thread,
    merge_gauges,
)
from mlagents.trainers.trainer import Trainer
from mlagents.trainers.environment_parameter_manager import EnvironmentParameterManager
from mlagents.trainers.trainer import TrainerFactory
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.agent_processor import AgentManagerQueue, AgentManager
from mlagents.trainers.trajectory import Trajectory
from mlagents.trainers.policy import Policy
from mlagents import torch_utils
from mlagents.torch_utils.globals import get_rank


class TrainerController:
    def __init__(
        self,
        trainer_factory: TrainerFactory,
        output_path: str,
        run_id: str,
        param_manager: EnvironmentParameterManager,
        train: bool,
        training_seed: int,
    ):
        """
        :param output_path: Path to save the model.
        :param summaries_dir: Folder to save training summaries.
        :param run_id: The sub-directory name for model and summary statistics
        :param param_manager: EnvironmentParameterManager object which stores information about all
        environment parameters.
        :param train: Whether to train model, or only run inference.
        :param training_seed: Seed to use for Numpy and Torch random number generation.
        :param threaded: Whether or not to run trainers in a separate thread. Disable for testing/debugging.
        """
        self.trainers: Dict[str, Trainer] = {}
        self.brain_name_to_identifier: Dict[str, Set] = defaultdict(set)
        self.trainer_factory = trainer_factory
        self.output_path = output_path
        self.logger = get_logger(__name__)
        self.run_id = run_id
        self.train_model = train
        self.param_manager = param_manager
        self.ghost_controller = self.trainer_factory.ghost_controller
        self.registered_behavior_ids: Set[str] = set()

        # self.trainer_threads: List[threading.Thread] = []
        self.trainer_processes: List[mp.Process] = []
        self.kill_trainers = multiprocessing.Value('b', False)
        np.random.seed(training_seed)
        torch_utils.torch.manual_seed(training_seed)
        self.rank = get_rank()

    @timed
    def _save_models(self):
        """
        Saves current model to checkpoint folder.
        """
        if self.rank is not None and self.rank != 0:
            return

        for brain_name in self.trainers.keys():
            self.trainers[brain_name].save_model()
        self.logger.debug("Saved Model")

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
    def _reset_env(self, env_manager: EnvManager) -> None:
        """Resets the environment.

        Returns:
            A Data structure corresponding to the initial reset state of the
            environment.
        """
        new_config = self.param_manager.get_current_samplers()
        env_manager.reset(config=new_config)
        # Register any new behavior ids that were generated on the reset.
        self._register_new_behaviors(env_manager, env_manager.first_step_infos)

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
        trainerprocess = None
        if brain_name in self.trainers:
            trainer = self.trainers[brain_name]
        else:
            trainer = self.trainer_factory.generate(brain_name)
            self.trainers[brain_name] = trainer
            if trainer.threaded:
                # trainerprocess = mp.Process(target=self.trainer_update_func, args=(trainer,), daemon=True)
                trainerprocess = mp.Process(target=self.trainer_update_func,
                                            args=(trainer,
                                                  [tajectory_queue for tajectory_queue in trainer.trajectory_queues],
                                                  [policy_queue for policy_queue in trainer.policy_queues],), daemon=True)
                self.trainer_processes.append(trainerprocess)
                print("maryam made process")
                # Only create trainer thread for new trainers
                # trainerthread = threading.Thread(
                #     target=self.trainer_update_func, args=(trainer,), daemon=True
                # )
                # self.trainer_threads.append(trainerthread)
            env_manager.on_training_started(
                brain_name, self.trainer_factory.trainer_config[brain_name]
            )

        policy = trainer.create_policy(
            parsed_behavior_id,
            env_manager.training_behaviors[name_behavior_id],
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
        self.shareParameters(trainer)
        # Only start new trainers
        if trainerprocess is not None:
            print("maryam in process start")
            trainerprocess.start()
            # trainerthread.start()

    def _create_trainers_and_managers(
        self, env_manager: EnvManager, behavior_ids: Set[str]
    ) -> None:
        for behavior_id in behavior_ids:
            self._create_trainer_and_manager(env_manager, behavior_id)

    @timed
    def start_learning(self, env_manager: EnvManager) -> None:
        self._create_output_path(self.output_path)
        try:
            # Initial reset
            self._reset_env(env_manager)
            self.param_manager.log_current_lesson()
            while self._not_done_training():
                n_steps = self.advance(env_manager)
                for _ in range(n_steps):
                    self.reset_env_if_ready(env_manager)
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
                self._save_models()

    def end_trainer_episodes(self) -> None:
        # Reward buffers reset takes place only for curriculum learning
        # else no reset.
        for trainer in self.trainers.values():
            trainer.end_episode()

    def reset_env_if_ready(self, env: EnvManager) -> None:
        # Get the sizes of the reward buffers.
        reward_buff = {k: list(t.reward_buffer) for (k, t) in self.trainers.items()}
        curr_step = {k: int(t.get_step) for (k, t) in self.trainers.items()}
        max_step = {k: int(t.get_max_steps) for (k, t) in self.trainers.items()}
        # Attempt to increment the lessons of the brains who
        # were ready.
        updated, param_must_reset = self.param_manager.update_lessons(
            curr_step, max_step, reward_buff
        )
        if updated:
            for trainer in self.trainers.values():
                trainer.reward_buffer.clear()
        # If ghost trainer swapped teams
        ghost_controller_reset = self.ghost_controller.should_reset()
        if param_must_reset or ghost_controller_reset:
            self._reset_env(env)  # This reset also sends the new config to env
            self.end_trainer_episodes()
        elif updated:
            env.set_env_parameters(self.param_manager.get_current_samplers())

    @timed
    def advance(self, env_manager: EnvManager) -> int:
        # Get steps
        with hierarchical_timer("env_step"):
            new_step_infos = env_manager.get_steps()
            self._register_new_behaviors(env_manager, new_step_infos)
            num_steps = env_manager.process_steps(new_step_infos)

        # Report current lesson for each environment parameter
        for (
            param_name,
            lesson_number,
        ) in self.param_manager.get_current_lesson_number().items():
            for trainer in self.trainers.values():
                trainer.stats_reporter.set_stat(
                    f"Environment/Lesson Number/{param_name}", lesson_number
                )
        for trainer in self.trainers.values():
            trainer.advance_process()
            if not trainer.threaded:
                with hierarchical_timer("trainer_advance"):
                    trainer.advance_update()

        return num_steps

    def _register_new_behaviors(
        self, env_manager: EnvManager, step_infos: List[EnvironmentStep]
    ) -> None:
        """
        Handle registration (adding trainers and managers) of new behaviors ids.
        :param env_manager:
        :param step_infos:
        :return:
        """
        step_behavior_ids: Set[str] = set()
        for s in step_infos:
            step_behavior_ids |= set(s.name_behavior_ids)
        new_behavior_ids = step_behavior_ids - self.registered_behavior_ids
        self._create_trainers_and_managers(env_manager, new_behavior_ids)
        self.registered_behavior_ids |= step_behavior_ids

    def shareParameters(self, trainer:Trainer):
        # TODO: This specific to PPO and Adam optimizer
        trainer.policy.actor.share_memory()
        trainer.optimizer.critic.share_memory()
        self._shareOptimizer(trainer.optimizer.optimizer)

        # trainer.optimizer.
        # self.policy_optimizer = torch.optim.Adam(
        #     policy_params, lr=hyperparameters.learning_rate
        # )
        # self.value_optimizer = torch.optim.Adam(
        #     value_params, lr=hyperparameters.learning_rate
        # )
        # self.entropy_optimizer

    def _shareOptimizer(self, optimizerTensor):
        print(f"maryam in swhere")
        for group in optimizerTensor.param_groups:
            for p in group['params']:
                state = optimizerTensor.state[p]
                # initialize: have to initialize here, or else cannot find
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    def join_threads(self, timeout_seconds: float = 1.0) -> None:
        """
        Wait for threads to finish, and merge their timer information into the main thread.
        :param timeout_seconds:
        :return:
        """
        self.kill_trainers.value = True


        for p in self.trainer_processes:
            try:
                p.join(timeout_seconds)
            except Exception:
                pass

            # sigkill child

        # with hierarchical_timer("trainer_threads") as main_timer_node:
        #     for trainer_thread in self.trainer_processes:
        #         thread_timer_stack = get_timer_stack_for_thread(trainer_thread)
        #         if thread_timer_stack:
        #             main_timer_node.merge(
        #                 thread_timer_stack.root,
        #                 root_name="thread_root",
        #                 is_parallel=True,
        #             )
        #             merge_gauges(thread_timer_stack.gauges)

        # for t in self.trainer_threads:
        #     try:
        #         t.join(timeout_seconds)
        #     except Exception:
        #         pass
        #
        # with hierarchical_timer("trainer_threads") as main_timer_node:
        #     for trainer_thread in self.trainer_threads:
        #         thread_timer_stack = get_timer_stack_for_thread(trainer_thread)
        #         if thread_timer_stack:
        #             main_timer_node.merge(
        #                 thread_timer_stack.root,
        #                 root_name="thread_root",
        #                 is_parallel=True,
        #             )
        #             merge_gauges(thread_timer_stack.gauges)

    # def trainer_update_func(self, trainer: Trainer) -> None:

    def trainer_update_func(self, trainer:Trainer, trajectory_list:List[AgentManagerQueue[Trajectory]],
                       policy_list:List[AgentManagerQueue[Policy]]) -> None:
        trainer.trajectory_queues = trajectory_list
        trainer.policy_queues = policy_list
        print("maryam is in child process!", flush=True)
        import time

        with open("./maryamFile.txt", 'w') as file:
            file.write(f" Maryam time: {time.ctime(time.time())}")
        while not self.kill_trainers.value:
            with hierarchical_timer("trainer_advance"):
                trainer.advance_update()

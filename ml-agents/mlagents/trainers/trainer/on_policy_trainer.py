# # Unity ML-Agents Toolkit
# ## ML-Agent Learning (PPO)
# Contains an implementation of PPO as described in: https://arxiv.org/abs/1707.06347

from collections import defaultdict
from typing import cast

import numpy as np

from mlagents_envs.logging_util import get_logger
from mlagents.trainers.buffer import BufferKey
from mlagents.trainers.trainer.rl_trainer import RLTrainer
from mlagents.trainers.policy import Policy
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.settings import TrainerSettings, OnPolicyHyperparamSettings

logger = get_logger(__name__)


class OnPolicyTrainer(RLTrainer):
    """The PPOTrainer is an implementation of the PPO algorithm."""

    def __init__(
        self,
        behavior_name: str,
        reward_buff_cap: int,
        trainer_settings: TrainerSettings,
        training: bool,
        load: bool,
        seed: int,
        artifact_path: str,
    ):
        """
        Responsible for collecting experiences and training an on-policy model.
        :param behavior_name: The name of the behavior associated with trainer config
        :param reward_buff_cap: Max reward history to track in the reward buffer
        :param trainer_settings: The parameters for the trainer.
        :param training: Whether the trainer is set for training.
        :param load: Whether the model should be loaded.
        :param seed: The seed the model will be initialized with
        :param artifact_path: The directory within which to store artifacts from this trainer.
        """
        super().__init__(
            behavior_name,
            trainer_settings,
            training,
            load,
            artifact_path,
            reward_buff_cap,
        )
        self.hyperparameters = cast(
            OnPolicyHyperparamSettings, self.trainer_settings.hyperparameters
        )
        self.seed = seed
        self.policy: Policy = None  # type: ignore
        self.optimizer: TorchOptimizer = None  # type: ignore

    def _is_ready_update(self):
        """
        Returns whether or not the trainer has enough elements to run update model
        :return: A boolean corresponding to whether or not update_model() can be run
        """
        size_of_buffer = self.update_buffer.num_experiences
        return size_of_buffer > self.hyperparameters.buffer_size

    def _update_policy(self):
        """
        Uses demonstration_buffer to update the policy.
        The reward signal generators must be updated in this method at their own pace.
        """
        buffer_length = self.update_buffer.num_experiences
        self.cumulative_returns_since_policy_update.clear()

        # Make sure batch_size is a multiple of sequence length. During training, we
        # will need to reshape the data into a batch_size x sequence_length tensor.
        batch_size = (
            self.hyperparameters.batch_size
            - self.hyperparameters.batch_size % self.policy.sequence_length
        )
        # Make sure there is at least one sequence
        batch_size = max(batch_size, self.policy.sequence_length)

        n_sequences = max(
            int(self.hyperparameters.batch_size / self.policy.sequence_length), 1
        )

        advantages = np.array(
            self.update_buffer[BufferKey.ADVANTAGES].get_batch(), dtype=np.float32
        )
        self.update_buffer[BufferKey.ADVANTAGES].set(
            (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        )
        num_epoch = self.hyperparameters.num_epoch
        batch_update_stats = defaultdict(list)
        for _ in range(num_epoch):
            self.update_buffer.shuffle(sequence_length=self.policy.sequence_length)
            buffer = self.update_buffer
            max_num_batch = buffer_length // batch_size
            for i in range(0, max_num_batch * batch_size, batch_size):
                minibatch = buffer.make_mini_batch(i, i + batch_size)
                update_stats = self.optimizer.update(minibatch, n_sequences)
                update_stats.update(self.optimizer.update_reward_signals(minibatch))
                for stat_name, value in update_stats.items():
                    batch_update_stats[stat_name].append(value)

        for stat, stat_list in batch_update_stats.items():
            self._stats_reporter.add_stat(stat, np.mean(stat_list))

        if self.optimizer.bc_module:
            update_stats = self.optimizer.bc_module.update()
            for stat, val in update_stats.items():
                self._stats_reporter.add_stat(stat, val)
        self._clear_update_buffer()
        return True

    def add_policy(
        self, parsed_behavior_id: BehaviorIdentifiers, policy: Policy
    ) -> None:
        """
        Adds policy to trainer.
        :param parsed_behavior_id: Behavior identifiers that the policy should belong to.
        :param policy: Policy to associate with name_behavior_id.
        """
        if self.policy:
            logger.warning(
                "Your environment contains multiple teams, but {} doesn't support adversarial games. Enable self-play to \
                    train adversarial games.".format(
                    self.__class__.__name__
                )
            )
        self.policy = policy
        self.policies[parsed_behavior_id.behavior_id] = policy

        self.optimizer = self.create_optimizer()
        for _reward_signal in self.optimizer.reward_signals.keys():
            self.collected_rewards[_reward_signal] = defaultdict(lambda: 0)

        self.model_saver.register(self.policy)
        self.model_saver.register(self.optimizer)
        self.model_saver.initialize_or_load()

        # Needed to resume loads properly
        self._step = policy.get_current_step()

# # Unity ML-Agents Toolkit
# ## ML-Agent Learning (PPO)
# Contains an implementation of PPO as described in: https://arxiv.org/abs/1707.06347

from collections import defaultdict
from typing import cast

import numpy as np
import copy

from mlagents_envs.logging_util import get_logger
from mlagents.trainers.policy.nn_policy import NNPolicy
from mlagents.trainers.policy.transfer_policy import TransferPolicy
from mlagents.trainers.trainer.rl_trainer import RLTrainer
from mlagents.trainers.brain import BrainParameters
from mlagents.trainers.policy.tf_policy import TFPolicy
from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.ppo_transfer.optimizer import PPOTransferOptimizer
from mlagents.trainers.trajectory import Trajectory
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.settings import TrainerSettings, PPOSettings, PPOTransferSettings

BUFFER_TRUNCATE_PERCENT = 0.6
logger = get_logger(__name__)


class PPOTransferTrainer(RLTrainer):
    """The PPOTrainer is an implementation of the PPO algorithm."""

    def __init__(
        self,
        brain_name: str,
        reward_buff_cap: int,
        trainer_settings: TrainerSettings,
        training: bool,
        load: bool,
        seed: int,
        artifact_path: str,
    ):
        """
        Responsible for collecting experiences and training PPO model.
        :param brain_name: The name of the brain associated with trainer config
        :param reward_buff_cap: Max reward history to track in the reward buffer
        :param trainer_settings: The parameters for the trainer.
        :param training: Whether the trainer is set for training.
        :param load: Whether the model should be loaded.
        :param seed: The seed the model will be initialized with
        :param artifact_path: The directory within which to store artifacts from this trainer.
        """
        super(PPOTransferTrainer, self).__init__(
            brain_name, trainer_settings, training, artifact_path, reward_buff_cap
        )
        self.hyperparameters: PPOTransferSettings = cast(
            PPOTransferSettings, self.trainer_settings.hyperparameters
        )
        self.load = load
        self.seed = seed
        self.policy: TransferPolicy = None  # type: ignore
        self.off_policy_buffer: AgentBuffer = AgentBuffer()
        self.use_iealter = self.hyperparameters.in_epoch_alter
        self.use_op_buffer = self.hyperparameters.use_op_buffer
        self.conv_thres = self.hyperparameters.conv_thres
        self.use_bisim = self.hyperparameters.use_bisim
        self.num_check = 0
        self.train_model = True
        self.old_loss = np.inf
        print("The current algorithm is PPO Transfer")

    def _process_trajectory(self, trajectory: Trajectory) -> None:
        """
        Takes a trajectory and processes it, putting it into the update buffer.
        Processing involves calculating value and advantage targets for model updating step.
        :param trajectory: The Trajectory tuple containing the steps to be processed.
        """
        super()._process_trajectory(trajectory)
        agent_id = trajectory.agent_id  # All the agents should have the same ID

        agent_buffer_trajectory = trajectory.to_agentbuffer()
        # Update the normalization
        if self.is_training:
            self.policy.update_normalization(agent_buffer_trajectory["vector_obs"])

        # Get all value estimates
        value_estimates, value_next = self.optimizer.get_trajectory_value_estimates(
            agent_buffer_trajectory,
            trajectory.next_obs,
            trajectory.done_reached and not trajectory.interrupted,
        )
        for name, v in value_estimates.items():
            agent_buffer_trajectory["{}_value_estimates".format(name)].extend(v)
            self._stats_reporter.add_stat(
                self.optimizer.reward_signals[name].value_name, np.mean(v)
            )

        # Evaluate all reward functions
        self.collected_rewards["environment"][agent_id] += np.sum(
            agent_buffer_trajectory["environment_rewards"]
        )
        for name, reward_signal in self.optimizer.reward_signals.items():
            evaluate_result = reward_signal.evaluate_batch(
                agent_buffer_trajectory
            ).scaled_reward
            agent_buffer_trajectory["{}_rewards".format(name)].extend(evaluate_result)
            # Report the reward signals
            self.collected_rewards[name][agent_id] += np.sum(evaluate_result)

        # Compute GAE and returns
        tmp_advantages = []
        tmp_returns = []
        for name in self.optimizer.reward_signals:
            bootstrap_value = value_next[name]

            local_rewards = agent_buffer_trajectory[
                "{}_rewards".format(name)
            ].get_batch()
            local_value_estimates = agent_buffer_trajectory[
                "{}_value_estimates".format(name)
            ].get_batch()
            local_advantage = get_gae(
                rewards=local_rewards,
                value_estimates=local_value_estimates,
                value_next=bootstrap_value,
                gamma=self.optimizer.reward_signals[name].gamma,
                lambd=self.hyperparameters.lambd,
            )
            local_return = local_advantage + local_value_estimates
            # This is later use as target for the different value estimates
            agent_buffer_trajectory["{}_returns".format(name)].set(local_return)
            agent_buffer_trajectory["{}_advantage".format(name)].set(local_advantage)
            tmp_advantages.append(local_advantage)
            tmp_returns.append(local_return)

        # Get global advantages
        global_advantages = list(
            np.mean(np.array(tmp_advantages, dtype=np.float32), axis=0)
        )
        global_returns = list(np.mean(np.array(tmp_returns, dtype=np.float32), axis=0))
        agent_buffer_trajectory["advantages"].set(global_advantages)
        agent_buffer_trajectory["discounted_returns"].set(global_returns)

        # Append to update buffer
        agent_buffer_trajectory.resequence_and_append(
            self.update_buffer, training_length=self.policy.sequence_length
        )
        # the off-policy buffer
        if self.use_op_buffer:
            agent_buffer_trajectory.resequence_and_append(
                self.off_policy_buffer, training_length=self.policy.sequence_length
            )

        # If this was a terminal trajectory, append stats and reset reward collection
        if trajectory.done_reached:
            self._update_end_episode_stats(agent_id, self.optimizer)

    def _is_ready_update(self):
        """
        Returns whether or not the trainer has enough elements to run update model
        :return: A boolean corresponding to whether or not update_model() can be run
        """
        # if  self.train_model and self.use_op_buffer:
        #     size_of_buffer = self.off_policy_buffer.num_experiences
        #     self.num_check += 1
        #     if self.num_check % 50 == 0 and size_of_buffer >= self.hyperparameters.buffer_size:
        #         return True
        #     else:
        #         return False
        # else:
        size_of_buffer = self.update_buffer.num_experiences
        return size_of_buffer > self.hyperparameters.buffer_size

    def _update_policy(self):
        """
        Uses demonstration_buffer to update the policy.
        The reward signal generators must be updated in this method at their own pace.
        """
        if self.train_model and self.use_op_buffer:
            self._update_model()
            # if self.update_buffer.num_experiences < self.hyperparameters.buffer_size:
            #     return True
            
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

        advantages = self.update_buffer["advantages"].get_batch()
        self.update_buffer["advantages"].set(
            (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        )
        num_epoch = self.hyperparameters.num_epoch
        batch_update_stats = defaultdict(list)
        for _ in range(num_epoch):
            if self.use_iealter:
                self.update_buffer.shuffle(sequence_length=self.policy.sequence_length)
                buffer = self.update_buffer
                max_num_batch = buffer_length // batch_size
                for i in range(0, max_num_batch * batch_size, batch_size):
                    update_stats = self.optimizer.update_part(
                        buffer.make_mini_batch(i, i + batch_size), n_sequences, "model"
                    )
                    for stat_name, value in update_stats.items():
                        batch_update_stats[stat_name].append(value)

                self.update_buffer.shuffle(sequence_length=self.policy.sequence_length)
                buffer = self.update_buffer
                max_num_batch = buffer_length // batch_size
                for i in range(0, max_num_batch * batch_size, batch_size):
                    update_stats = self.optimizer.update_part(
                        buffer.make_mini_batch(i, i + batch_size), n_sequences, "policy"
                    )
                    for stat_name, value in update_stats.items():
                        batch_update_stats[stat_name].append(value)
                if self.use_bisim:
                    self.update_buffer.shuffle(sequence_length=self.policy.sequence_length)
                    buffer1 = copy.deepcopy(self.update_buffer)
                    self.update_buffer.shuffle(sequence_length=self.policy.sequence_length)
                    buffer2 = copy.deepcopy(self.update_buffer)
                    self.update_buffer.shuffle(sequence_length=self.policy.sequence_length)
                    buffer3 = copy.deepcopy(self.update_buffer)
                    max_num_batch = buffer_length // batch_size
                    for i in range(0, max_num_batch * batch_size, batch_size):
                        update_stats = self.optimizer.update_encoder(
                            buffer1.make_mini_batch(i, i + batch_size), 
                            buffer2.make_mini_batch(i, i + batch_size), 
                            buffer3.make_mini_batch(i, i + batch_size), 
                        )
                        for stat_name, value in update_stats.items():
                            batch_update_stats[stat_name].append(value)
            else:
                self.update_buffer.shuffle(sequence_length=self.policy.sequence_length)
                buffer = self.update_buffer
                max_num_batch = buffer_length // batch_size
                for i in range(0, max_num_batch * batch_size, batch_size):
                    update_stats = self.optimizer.update(
                        buffer.make_mini_batch(i, i + batch_size), n_sequences
                    )
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
    
    def _update_model(self):
        """
        Uses demonstration_buffer to update the policy.
        The reward signal generators must be updated in this method at their own pace.
        """
        buffer_length = self.off_policy_buffer.num_experiences
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
        num_epoch = self.hyperparameters.num_epoch
        batch_update_stats = defaultdict(list)
        for _ in range(num_epoch):
            self.off_policy_buffer.shuffle(sequence_length=self.policy.sequence_length)
            buffer = self.off_policy_buffer
            max_num_batch = buffer_length // batch_size
            for i in range(0, max_num_batch * batch_size, batch_size):
                update_stats = self.optimizer.update_part(
                    buffer.make_mini_batch(i, i + batch_size), n_sequences, "model"
                )
                for stat_name, value in update_stats.items():
                    batch_update_stats[stat_name].append(value)
        for stat, stat_list in batch_update_stats.items():
            self._stats_reporter.add_stat(stat, np.mean(stat_list))
            if stat == "Losses/Model Loss": # and np.mean(stat_list) < 0.01:
                if abs(self.old_loss - np.mean(stat_list)) < 1e-3:
                    self.train_model = False
                else:
                    self.old_loss = np.mean(stat_list)
                print(stat, np.mean(stat_list))

        if self.optimizer.bc_module:
            update_stats = self.optimizer.bc_module.update()
            for stat, val in update_stats.items():
                self._stats_reporter.add_stat(stat, val)
        
        # self.off_policy_buffer.reset_agent()
        if self.off_policy_buffer.num_experiences > 10 * self.hyperparameters.buffer_size:
            print("truncate")
            self.off_policy_buffer.truncate(
                int(5 * self.hyperparameters.buffer_size)
            )
        
        return True

    def create_policy(
        self, parsed_behavior_id: BehaviorIdentifiers, brain_parameters: BrainParameters
    ) -> TFPolicy:
        """
        Creates a PPO policy to trainers list of policies.
        :param brain_parameters: specifications for policy construction
        :return policy
        """
        policy = TransferPolicy(
            self.seed,
            brain_parameters,
            self.trainer_settings,
            self.is_training,
            self.artifact_path,
            self.load,
            condition_sigma_on_obs=False,  # Faster training for PPO
            create_tf_graph=False,  # We will create the TF graph in the Optimizer
        )

        return policy

    def add_policy(
        self, parsed_behavior_id: BehaviorIdentifiers, policy: TFPolicy
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
        if not isinstance(policy, TransferPolicy):
            raise RuntimeError("Non-NNPolicy passed to PPOTrainer.add_policy()")
        self.policy = policy
        self.optimizer = PPOTransferOptimizer(self.policy, self.trainer_settings)
        for _reward_signal in self.optimizer.reward_signals.keys():
            self.collected_rewards[_reward_signal] = defaultdict(lambda: 0)
        # Needed to resume loads properly
        self.step = policy.get_current_step()

    def get_policy(self, name_behavior_id: str) -> TFPolicy:
        """
        Gets policy from trainer associated with name_behavior_id
        :param name_behavior_id: full identifier of policy
        """

        return self.policy


def discount_rewards(r, gamma=0.99, value_next=0.0):
    """
    Computes discounted sum of future rewards for use in updating value estimate.
    :param r: List of rewards.
    :param gamma: Discount factor.
    :param value_next: T+1 value estimate for returns calculation.
    :return: discounted sum of future rewards as list.
    """
    discounted_r = np.zeros_like(r)
    running_add = value_next
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def get_gae(rewards, value_estimates, value_next=0.0, gamma=0.99, lambd=0.95):
    """
    Computes generalized advantage estimate for use in updating policy.
    :param rewards: list of rewards for time-steps t to T.
    :param value_next: Value estimate for time-step T+1.
    :param value_estimates: list of value estimates for time-steps t to T.
    :param gamma: Discount factor.
    :param lambd: GAE weighing factor.
    :return: list of advantage estimates for time-steps t to T.
    """
    value_estimates = np.append(value_estimates, value_next)
    delta_t = rewards + gamma * value_estimates[1:] - value_estimates[:-1]
    advantage = discount_rewards(r=delta_t, gamma=gamma * lambd)
    return advantage

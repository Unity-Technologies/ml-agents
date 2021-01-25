# # Unity ML-Agents Toolkit
# ## ML-Agent Learning (PPO)
# Contains an implementation of PPO as described in: https://arxiv.org/abs/1707.06347

from collections import defaultdict
from typing import cast

import numpy as np

from mlagents_envs.logging_util import get_logger
from mlagents_envs.base_env import BehaviorSpec
from mlagents.trainers.trainer.rl_trainer import RLTrainer
from mlagents.trainers.policy import Policy
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.ppo.optimizer_torch import TorchPPOOptimizer
from mlagents.trainers.trajectory import Trajectory
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.settings import TrainerSettings, PPOSettings

logger = get_logger(__name__)


class PPOTrainer(RLTrainer):
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
        Responsible for collecting experiences and training PPO model.
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
        self.hyperparameters: PPOSettings = cast(
            PPOSettings, self.trainer_settings.hyperparameters
        )
        self.seed = seed
        self.policy: Policy = None  # type: ignore

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
            self.policy.update_normalization(agent_buffer_trajectory)

        # Get all value estimates
        (
            q_estimates,
            baseline_estimates,
            value_estimates,
            value_next,
            baseline_next,
        ) = self.optimizer.get_trajectory_value_estimates(
            agent_buffer_trajectory,
            trajectory.next_obs,
            trajectory.next_collab_obs,
            trajectory.done_reached and not trajectory.interrupted,
        )

        for name, v in q_estimates.items():
            agent_buffer_trajectory[f"{name}_q_estimates"].extend(v)
            agent_buffer_trajectory[f"{name}_baseline_estimates"].extend(
                baseline_estimates[name]
            )
            agent_buffer_trajectory[f"{name}_value_estimates"].extend(value_estimates[name])
            self._stats_reporter.add_stat(
                f"Policy/{self.optimizer.reward_signals[name].name.capitalize()} Q Estimate",
                np.mean(v),
            )
            self._stats_reporter.add_stat(
                f"Policy/{self.optimizer.reward_signals[name].name.capitalize()} Baseline Estimate",
                np.mean(baseline_estimates[name]),
            )
            self._stats_reporter.add_stat(
                f"Policy/{self.optimizer.reward_signals[name].name.capitalize()} Value Estimate",
                np.mean(value_estimates[name]),
            )


        #for name, v in value_next.items():
        #    agent_buffer_trajectory[f"{name}_value_estimates_next"].extend(v)
        #    agent_buffer_trajectory[f"{name}_marginalized_value_estimates_next"].extend(
        #        marg_value_next[name]
        #    )

        # Evaluate all reward functions
        self.collected_rewards["environment"][agent_id] += np.sum(
            agent_buffer_trajectory["environment_rewards"]
        )
        for name, reward_signal in self.optimizer.reward_signals.items():
            evaluate_result = (
                reward_signal.evaluate(agent_buffer_trajectory) * reward_signal.strength
            )
            agent_buffer_trajectory[f"{name}_rewards"].extend(evaluate_result)
            # Report the reward signals
            self.collected_rewards[name][agent_id] += np.sum(evaluate_result)

        # Compute GAE and returns
        tmp_advantages = []
        tmp_returns = []
        for name in self.optimizer.reward_signals:

            local_rewards = agent_buffer_trajectory[f"{name}_rewards"].get_batch()
            q_estimates = agent_buffer_trajectory[
                f"{name}_q_estimates"
            ].get_batch()
            baseline_estimates = agent_buffer_trajectory[
                f"{name}_baseline_estimates"
            ].get_batch()
            v_estimates = agent_buffer_trajectory[
                f"{name}_value_estimates"
            ].get_batch()

            #next_value_estimates = agent_buffer_trajectory[
            #    f"{name}_value_estimates_next"
            #].get_batch()
            #next_m_value_estimates = agent_buffer_trajectory[
            #    f"{name}_marginalized_value_estimates_next"
            #].get_batch()

            returns_q, returns_b, returns_v = get_team_returns(
                rewards=local_rewards,
                q_estimates=q_estimates,
                baseline_estimates=baseline_estimates,
                v_estimates=v_estimates,
                value_next=value_next[name],
                baseline_next=baseline_next[name],
                gamma=self.optimizer.reward_signals[name].gamma,
                lambd=self.hyperparameters.lambd,
            )
            #gae_advantage = get_team_gae(
            #    rewards=local_rewards,
            #    value_estimates=v_estimates,
            #    baseline=baseline_estimates,
            #    value_next=bootstrap_value,
            #    gamma=self.optimizer.reward_signals[name].gamma,
            #    lambd=self.hyperparameters.lambd,
            #)
            self._stats_reporter.add_stat(
                f"Policy/{self.optimizer.reward_signals[name].name.capitalize()} TD Lam",
                np.mean(returns_v),
            )
            


            #local_advantage = np.array(q_estimates) - np.array(
            local_advantage = np.array(returns_v) - np.array(
                baseline_estimates
            )
            #self._stats_reporter.add_stat(
            #    f"Policy/{self.optimizer.reward_signals[name].name.capitalize()} GAE Advantage Estimate",
            #    np.mean(gae_advantage),
            #)

            self._stats_reporter.add_stat(
                f"Policy/{self.optimizer.reward_signals[name].name.capitalize()} TD Advantage Estimate",
                np.mean(local_advantage),
            )
            #local_return = local_advantage + q_estimates
            local_return = local_advantage + baseline_estimates
            # This is later use as target for the different value estimates
            # agent_buffer_trajectory[f"{name}_returns"].set(local_return)
            agent_buffer_trajectory[f"{name}_returns_q"].set(returns_v)
            agent_buffer_trajectory[f"{name}_returns_b"].set(returns_b)
            agent_buffer_trajectory[f"{name}_returns_v"].set(returns_v)
            agent_buffer_trajectory[f"{name}_advantage"].set(local_advantage)
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

        # If this was a terminal trajectory, append stats and reset reward collection
        if trajectory.done_reached:
            self._update_end_episode_stats(agent_id, self.optimizer)

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
        #Normalize advantages
        advantages = np.array(self.update_buffer["advantages"].get_batch())
        self.update_buffer["advantages"].set(
           list((advantages - advantages.mean()) / (advantages.std() + 1e-10))
        )
        num_epoch = self.hyperparameters.num_epoch
        batch_update_stats = defaultdict(list)
        for _ in range(num_epoch):
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

    def create_torch_policy(
        self, parsed_behavior_id: BehaviorIdentifiers, behavior_spec: BehaviorSpec
    ) -> TorchPolicy:
        """
        Creates a policy with a PyTorch backend and PPO hyperparameters
        :param parsed_behavior_id:
        :param behavior_spec: specifications for policy construction
        :return policy
        """
        policy = TorchPolicy(
            self.seed,
            behavior_spec,
            self.trainer_settings,
            condition_sigma_on_obs=False,  # Faster training for PPO
            separate_critic=True,  # Match network architecture with TF
        )
        return policy

    def create_ppo_optimizer(self) -> TorchPPOOptimizer:
        return TorchPPOOptimizer(  # type: ignore
            cast(TorchPolicy, self.policy), self.trainer_settings  # type: ignore
        )  # type: ignore

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

        self.optimizer = self.create_ppo_optimizer()
        for _reward_signal in self.optimizer.reward_signals.keys():
            self.collected_rewards[_reward_signal] = defaultdict(lambda: 0)

        self.model_saver.register(self.policy)
        self.model_saver.register(self.optimizer)
        self.model_saver.initialize_or_load()

        # Needed to resume loads properly
        self.step = policy.get_current_step()

    def get_policy(self, name_behavior_id: str) -> Policy:
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


def lambd_return(r, value_estimates, gamma=0.99, lambd=0.8, value_next=0.0, baseline=False):
    returns = np.zeros_like(r)
    if baseline:
        # this is incorrect
        if value_next == 0.0:
            returns[-1] = r[-1]
        else:
            returns[-1] = value_estimates[-1]
    else:
        returns[-1] = r[-1] + gamma * value_next
    for t in reversed(range(0, r.size - 1)):
        returns[t] = gamma * lambd * returns[t + 1]  + r[t] + (1 - lambd) * gamma * value_estimates[t + 1]
        
    return returns


def get_team_gae(rewards, value_estimates, baseline, value_next=0.0, gamma=0.99, lambd=0.95):
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
    delta_t = rewards + gamma * value_estimates[1:] - baseline
    advantage = discount_rewards(r=delta_t, gamma=gamma * lambd)
    return advantage


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


def get_team_returns(
    rewards,
    q_estimates,
    baseline_estimates,
    v_estimates,
    value_next=0.0,
    baseline_next=0.0,
    gamma=0.99,
    lambd=0.8,
):
    """
    Computes generalized advantage estimate for use in updating policy.
    :param rewards: list of rewards for time-steps t to T.
    :param value_next: Value estimate for time-step T+1.
    :param value_estimates: list of value estimates for time-steps t to T.
    :param gamma: Discount factor.
    :param lambd: GAE weighing factor.
    :return: list of advantage estimates for time-steps t to T.
    """
    rewards = np.array(rewards)
    returns_q = lambd_return(rewards, q_estimates, gamma=gamma, lambd=lambd, value_next=value_next)
    returns_b = lambd_return(
        rewards, baseline_estimates, gamma=gamma, lambd=lambd, value_next=baseline_next, baseline=True
    )
    returns_v = lambd_return(
        rewards, v_estimates, gamma=gamma, lambd=lambd, value_next=value_next
    )
    #if rewards[-1] > 0:
    #    print(returns_v)
    #    print(rewards)

    return returns_q, returns_b, returns_v

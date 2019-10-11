# # Unity ML-Agents Toolkit
# ## ML-Agent Learning (PPO)
# Contains an implementation of PPO as described in: https://arxiv.org/abs/1707.06347

import logging
from collections import defaultdict
from typing import Dict

import numpy as np

from mlagents.envs.brain import AllBrainInfo
from mlagents.trainers.ppo.policy import PPOPolicy
from mlagents.trainers.ppo.multi_gpu_policy import MultiGpuPPOPolicy, get_devices
from mlagents.trainers.rl_trainer import RLTrainer, AllRewardsOutput
from mlagents.envs.action_info import ActionInfoOutputs

logger = logging.getLogger("mlagents.trainers")


class PPOTrainer(RLTrainer):
    """The PPOTrainer is an implementation of the PPO algorithm."""

    def __init__(
        self,
        brain,
        reward_buff_cap,
        trainer_parameters,
        training,
        load,
        seed,
        run_id,
        multi_gpu,
    ):
        """
        Responsible for collecting experiences and training PPO model.
        :param trainer_parameters: The parameters for the trainer (dictionary).
        :param reward_buff_cap: Max reward history to track in the reward buffer
        :param training: Whether the trainer is set for training.
        :param load: Whether the model should be loaded.
        :param seed: The seed the model will be initialized with
        :param run_id: The identifier of the current run
        """
        super(PPOTrainer, self).__init__(
            brain, trainer_parameters, training, run_id, reward_buff_cap
        )
        self.param_keys = [
            "batch_size",
            "beta",
            "buffer_size",
            "epsilon",
            "hidden_units",
            "lambd",
            "learning_rate",
            "max_steps",
            "normalize",
            "num_epoch",
            "num_layers",
            "time_horizon",
            "sequence_length",
            "summary_freq",
            "use_recurrent",
            "summary_path",
            "memory_size",
            "model_path",
            "reward_signals",
        ]
        self.check_param_keys()

        if multi_gpu and len(get_devices()) > 1:
            self.policy = MultiGpuPPOPolicy(
                seed, brain, trainer_parameters, self.is_training, load
            )
        else:
            self.policy = PPOPolicy(
                seed, brain, trainer_parameters, self.is_training, load
            )

        for _reward_signal in self.policy.reward_signals.keys():
            self.collected_rewards[_reward_signal] = {}

    def process_experiences(
        self, current_info: AllBrainInfo, new_info: AllBrainInfo
    ) -> None:
        """
        Checks agent histories for processing condition, and processes them as necessary.
        Processing involves calculating value and advantage targets for model updating step.
        :param current_info: Dictionary of all current brains and corresponding BrainInfo.
        :param new_info: Dictionary of all next brains and corresponding BrainInfo.
        """
        info = new_info[self.brain_name]
        if self.is_training:
            self.policy.update_normalization(info.vector_observations)
        for l in range(len(info.agents)):
            agent_actions = self.training_buffer[info.agents[l]]["actions"]
            if (
                info.local_done[l]
                or len(agent_actions) > self.trainer_parameters["time_horizon"]
            ) and len(agent_actions) > 0:
                agent_id = info.agents[l]
                if info.max_reached[l]:
                    bootstrapping_info = self.training_buffer[agent_id].last_brain_info
                    idx = bootstrapping_info.agents.index(agent_id)
                else:
                    bootstrapping_info = info
                    idx = l
                value_next = self.policy.get_value_estimates(
                    bootstrapping_info,
                    idx,
                    info.local_done[l] and not info.max_reached[l],
                )

                tmp_advantages = []
                tmp_returns = []
                for name in self.policy.reward_signals:
                    bootstrap_value = value_next[name]

                    local_rewards = self.training_buffer[agent_id][
                        "{}_rewards".format(name)
                    ].get_batch()
                    local_value_estimates = self.training_buffer[agent_id][
                        "{}_value_estimates".format(name)
                    ].get_batch()
                    local_advantage = get_gae(
                        rewards=local_rewards,
                        value_estimates=local_value_estimates,
                        value_next=bootstrap_value,
                        gamma=self.policy.reward_signals[name].gamma,
                        lambd=self.trainer_parameters["lambd"],
                    )
                    local_return = local_advantage + local_value_estimates
                    # This is later use as target for the different value estimates
                    self.training_buffer[agent_id]["{}_returns".format(name)].set(
                        local_return
                    )
                    self.training_buffer[agent_id]["{}_advantage".format(name)].set(
                        local_advantage
                    )
                    tmp_advantages.append(local_advantage)
                    tmp_returns.append(local_return)

                global_advantages = list(np.mean(np.array(tmp_advantages), axis=0))
                global_returns = list(np.mean(np.array(tmp_returns), axis=0))
                self.training_buffer[agent_id]["advantages"].set(global_advantages)
                self.training_buffer[agent_id]["discounted_returns"].set(global_returns)

                self.training_buffer.append_update_buffer(
                    agent_id,
                    batch_size=None,
                    training_length=self.policy.sequence_length,
                )

                self.training_buffer[agent_id].reset_agent()
                if info.local_done[l]:
                    self.stats["Environment/Episode Length"].append(
                        self.episode_steps.get(agent_id, 0)
                    )
                    self.episode_steps[agent_id] = 0
                    for name, rewards in self.collected_rewards.items():
                        if name == "environment":
                            self.cumulative_returns_since_policy_update.append(
                                rewards.get(agent_id, 0)
                            )
                            self.stats["Environment/Cumulative Reward"].append(
                                rewards.get(agent_id, 0)
                            )
                            self.reward_buffer.appendleft(rewards.get(agent_id, 0))
                            rewards[agent_id] = 0
                        else:
                            self.stats[
                                self.policy.reward_signals[name].stat_name
                            ].append(rewards.get(agent_id, 0))
                            rewards[agent_id] = 0

    def add_policy_outputs(
        self, take_action_outputs: ActionInfoOutputs, agent_id: str, agent_idx: int
    ) -> None:
        """
        Takes the output of the last action and store it into the training buffer.
        """
        actions = take_action_outputs["action"]
        if self.policy.use_continuous_act:
            actions_pre = take_action_outputs["pre_action"]
            self.training_buffer[agent_id]["actions_pre"].append(actions_pre[agent_idx])
            epsilons = take_action_outputs["random_normal_epsilon"]
            self.training_buffer[agent_id]["random_normal_epsilon"].append(
                epsilons[agent_idx]
            )
        a_dist = take_action_outputs["log_probs"]
        # value is a dictionary from name of reward to value estimate of the value head
        self.training_buffer[agent_id]["actions"].append(actions[agent_idx])
        self.training_buffer[agent_id]["action_probs"].append(a_dist[agent_idx])

    def add_rewards_outputs(
        self,
        rewards_out: AllRewardsOutput,
        values: Dict[str, np.ndarray],
        agent_id: str,
        agent_idx: int,
        agent_next_idx: int,
    ) -> None:
        """
        Takes the value output of the last action and store it into the training buffer.
        """
        for name, reward_result in rewards_out.reward_signals.items():
            # 0 because we use the scaled reward to train the agent
            self.training_buffer[agent_id]["{}_rewards".format(name)].append(
                reward_result.scaled_reward[agent_next_idx]
            )
            self.training_buffer[agent_id]["{}_value_estimates".format(name)].append(
                values[name][agent_idx][0]
            )

    def is_ready_update(self):
        """
        Returns whether or not the trainer has enough elements to run update model
        :return: A boolean corresponding to whether or not update_model() can be run
        """
        size_of_buffer = len(self.training_buffer.update_buffer["actions"])
        return size_of_buffer > self.trainer_parameters["buffer_size"]

    def update_policy(self):
        """
        Uses demonstration_buffer to update the policy.
        The reward signal generators must be updated in this method at their own pace.
        """
        buffer_length = len(self.training_buffer.update_buffer["actions"])
        self.trainer_metrics.start_policy_update_timer(
            number_experiences=buffer_length,
            mean_return=float(np.mean(self.cumulative_returns_since_policy_update)),
        )
        self.cumulative_returns_since_policy_update = []

        # Make sure batch_size is a multiple of sequence length. During training, we
        # will need to reshape the data into a batch_size x sequence_length tensor.
        batch_size = (
            self.trainer_parameters["batch_size"]
            - self.trainer_parameters["batch_size"] % self.policy.sequence_length
        )
        # Make sure there is at least one sequence
        batch_size = max(batch_size, self.policy.sequence_length)

        n_sequences = max(
            int(self.trainer_parameters["batch_size"] / self.policy.sequence_length), 1
        )

        advantages = self.training_buffer.update_buffer["advantages"].get_batch()
        self.training_buffer.update_buffer["advantages"].set(
            (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        )
        num_epoch = self.trainer_parameters["num_epoch"]
        batch_update_stats = defaultdict(list)
        for _ in range(num_epoch):
            self.training_buffer.update_buffer.shuffle(
                sequence_length=self.policy.sequence_length
            )
            buffer = self.training_buffer.update_buffer
            max_num_batch = buffer_length // batch_size
            for l in range(0, max_num_batch * batch_size, batch_size):
                update_stats = self.policy.update(
                    buffer.make_mini_batch(l, l + batch_size), n_sequences
                )
                for stat_name, value in update_stats.items():
                    batch_update_stats[stat_name].append(value)

        for stat, stat_list in batch_update_stats.items():
            self.stats[stat].append(np.mean(stat_list))

        if self.policy.bc_module:
            update_stats = self.policy.bc_module.update()
            for stat, val in update_stats.items():
                self.stats[stat].append(val)
        self.clear_update_buffer()
        self.trainer_metrics.end_policy_update()


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

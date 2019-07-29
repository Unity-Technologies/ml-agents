# # Unity ML-Agents Toolkit
# ## ML-Agent Learning (PPO)
# Contains an implementation of PPO as described in: https://arxiv.org/abs/1707.06347

import logging
from collections import defaultdict

import numpy as np

from mlagents.trainers.buffer import Buffer
from mlagents.trainers.ppo.policy import PPOPolicy
from mlagents.trainers.trainer import Trainer, UnityTrainerException
from mlagents.envs.env_manager import AgentStep

logger = logging.getLogger("mlagents.trainers")


class PPOTrainer(Trainer):
    """The PPOTrainer is an implementation of the PPO algorithm."""

    def __init__(
        self, brain, reward_buff_cap, trainer_parameters, training, load, seed, run_id
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
        super().__init__(brain, trainer_parameters, training, run_id, reward_buff_cap)
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
            "vis_encode_type",
        ]
        self.check_param_keys()

        # Make sure we have at least one reward_signal
        if not self.trainer_parameters["reward_signals"]:
            raise UnityTrainerException(
                "No reward signals were defined. At least one must be used with {}.".format(
                    self.__class__.__name__
                )
            )

        self.step = 0
        self.policy = PPOPolicy(seed, brain, trainer_parameters, self.is_training, load)

        stats = defaultdict(list)
        # collected_rewards is a dictionary from name of reward signal to a dictionary of agent_id to cumulative reward
        # used for reporting only. We always want to report the environment reward to Tensorboard, regardless
        # of what reward signals are actually present.
        self.collected_rewards = {"environment": {}}
        for _reward_signal in self.policy.reward_signals.keys():
            self.collected_rewards[_reward_signal] = {}

        self.stats = stats

        self.training_buffer = Buffer()
        self.episode_steps = {}

    def __str__(self):
        return """Hyperparameters for the {0} of brain {1}: \n{2}""".format(
            self.__class__.__name__,
            self.brain_name,
            self.dict_to_str(self.trainer_parameters, 0),
        )

    @property
    def parameters(self):
        """
        Returns the trainer parameters of the trainer.
        """
        return self.trainer_parameters

    @property
    def get_max_steps(self):
        """
        Returns the maximum number of steps. Is used to know when the trainer should be stopped.
        :return: The maximum number of steps of the trainer
        """
        return float(self.trainer_parameters["max_steps"])

    @property
    def get_step(self):
        """
        Returns the number of steps the trainer has performed
        :return: the step count of the trainer
        """
        return self.step

    def increment_step(self, n_steps: int) -> None:
        """
        Increment the step count of the trainer

        :param n_steps: number of steps to increment the step count by
        """
        self.step = self.policy.increment_step(n_steps)

    def add_experiences(self, agent_step: AgentStep) -> None:
        """
        Adds experiences to each agent's experience history.
        :param agent_step: Agent step to be added to the training buffer.
        """
        if agent_step.current_agent_info.brain_name != self.brain_name:
            return

        self.trainer_metrics.start_experience_collection_timer()
        take_action_outputs = agent_step.action_info.outputs
        if agent_step.action_info is not None:
            self.stats["Policy/Entropy"].append(take_action_outputs["entropy"].mean())
            self.stats["Policy/Learning Rate"].append(
                take_action_outputs["learning_rate"]
            )
            for name, signal in self.policy.reward_signals.items():
                self.stats[signal.value_name].append(
                    np.mean(take_action_outputs["value"][name])
                )

        curr_info = agent_step.previous_agent_info
        next_info = agent_step.current_agent_info
        agent_id = curr_info.id

        tmp_rewards_dict = {}
        for name, signal in self.policy.reward_signals.items():
            tmp_rewards_dict[name] = signal.evaluate([agent_step])

        if curr_info is not None:
            if not curr_info.local_done:
                stored_take_action_outputs = agent_step.action_info.outputs
                for i, _ in enumerate(curr_info.visual_observations):
                    self.training_buffer[agent_id]["visual_obs%d" % i].append(
                        curr_info.visual_observations[i]
                    )
                    self.training_buffer[agent_id]["next_visual_obs%d" % i].append(
                        next_info.visual_observations[i]
                    )
                if self.policy.use_vec_obs:
                    self.training_buffer[agent_id]["vector_obs"].append(
                        curr_info.vector_observations
                    )
                    self.training_buffer[agent_id]["next_vector_in"].append(
                        next_info.vector_observations
                    )
                if self.policy.use_recurrent:
                    if curr_info.memories.shape[1] == 0:
                        curr_info.memories = np.zeros((1, self.policy.m_size))
                    self.training_buffer[agent_id]["memory"].append(curr_info.memories)
                actions = stored_take_action_outputs["action"]
                if self.policy.use_continuous_act:
                    actions_pre = stored_take_action_outputs["pre_action"]
                    self.training_buffer[agent_id]["actions_pre"].append(actions_pre)
                    epsilons = stored_take_action_outputs["random_normal_epsilon"]
                    self.training_buffer[agent_id]["random_normal_epsilon"].append(
                        epsilons
                    )
                else:
                    self.training_buffer[agent_id]["action_mask"].append(
                        curr_info.action_mask, padding_value=1
                    )
                a_dist = stored_take_action_outputs["log_probs"]
                # value is a dictionary from name of reward to value estimate of the value head
                value = stored_take_action_outputs["value"]
                self.training_buffer[agent_id]["actions"].append(actions)
                self.training_buffer[agent_id]["prev_action"].append(
                    curr_info.previous_vector_actions
                )
                self.training_buffer[agent_id]["masks"].append(1.0)
                self.training_buffer[agent_id]["done"].append(next_info.local_done)

                for name, reward_result in tmp_rewards_dict.items():
                    # 0 because we use the scaled reward to train the agent
                    self.training_buffer[agent_id]["{}_rewards".format(name)].append(
                        reward_result.scaled_reward[0]
                    )
                    self.training_buffer[agent_id][
                        "{}_value_estimates".format(name)
                    ].append(value[name][0])

                self.training_buffer[agent_id]["action_probs"].append(a_dist)

                for name, rewards in self.collected_rewards.items():
                    if agent_id not in rewards:
                        rewards[agent_id] = 0
                    if name == "environment":
                        # Report the reward from the environment
                        rewards[agent_id] += np.array(next_info.reward)
                    else:
                        # Report the reward signals
                        rewards[agent_id] += tmp_rewards_dict[name].scaled_reward[0]

            if not next_info.local_done:
                if agent_id not in self.episode_steps:
                    self.episode_steps[agent_id] = 0
                self.episode_steps[agent_id] += 1
        self.trainer_metrics.end_experience_collection_timer()

    def process_experiences(self, agent_step: AgentStep) -> None:
        """
        Checks agent histories for processing condition, and processes them as necessary.
        Processing involves calculating value and advantage targets for model updating step.
        :param agent_step: The current agent step to be processed.
        """
        current_agent_info = agent_step.current_agent_info
        agent_actions = self.training_buffer[current_agent_info.id]["actions"]
        if (
            current_agent_info.local_done
            or len(agent_actions) > self.trainer_parameters["time_horizon"]
        ) and len(agent_actions) > 0:
            agent_id = current_agent_info.id
            if current_agent_info.max_step_reached:
                bootstrapping_info = agent_step.previous_agent_info
            else:
                bootstrapping_info = current_agent_info
            value_next = self.policy.get_value_estimates(
                bootstrapping_info,
                current_agent_info.local_done
                and not current_agent_info.max_step_reached,
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
                agent_id, batch_size=None, training_length=self.policy.sequence_length
            )

            self.training_buffer[agent_id].reset_agent()
            if current_agent_info.local_done:
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
                        self.stats[self.policy.reward_signals[name].stat_name].append(
                            rewards.get(agent_id, 0)
                        )
                        rewards[agent_id] = 0

    def end_episode(self):
        """
        A signal that the Episode has ended. The buffer must be reset.
        Get only called when the academy resets.
        """
        self.training_buffer.reset_local_buffers()
        for agent_id in self.episode_steps:
            self.episode_steps[agent_id] = 0
        for rewards in self.collected_rewards.values():
            for agent_id in rewards:
                rewards[agent_id] = 0

    def is_ready_update(self):
        """
        Returns whether or not the trainer has enough elements to run update model
        :return: A boolean corresponding to whether or not update_model() can be run
        """
        size_of_buffer = len(self.training_buffer.update_buffer["actions"])
        return size_of_buffer > max(
            int(self.trainer_parameters["buffer_size"] / self.policy.sequence_length), 1
        )

    def update_policy(self):
        """
        Uses demonstration_buffer to update the policy.
        The reward signal generators must be updated in this method at their own pace.
        """
        self.trainer_metrics.start_policy_update_timer(
            number_experiences=len(self.training_buffer.update_buffer["actions"]),
            mean_return=float(np.mean(self.cumulative_returns_since_policy_update)),
        )
        self.cumulative_returns_since_policy_update = []
        n_sequences = max(
            int(self.trainer_parameters["batch_size"] / self.policy.sequence_length), 1
        )
        value_total, policy_total = [], []
        advantages = self.training_buffer.update_buffer["advantages"].get_batch()
        self.training_buffer.update_buffer["advantages"].set(
            (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        )
        num_epoch = self.trainer_parameters["num_epoch"]
        for _ in range(num_epoch):
            self.training_buffer.update_buffer.shuffle()
            buffer = self.training_buffer.update_buffer
            for l in range(
                len(self.training_buffer.update_buffer["actions"]) // n_sequences
            ):
                start = l * n_sequences
                end = (l + 1) * n_sequences
                run_out = self.policy.update(
                    buffer.make_mini_batch(start, end), n_sequences
                )
                value_total.append(run_out["value_loss"])
                policy_total.append(np.abs(run_out["policy_loss"]))
        self.stats["Losses/Value Loss"].append(np.mean(value_total))
        self.stats["Losses/Policy Loss"].append(np.mean(policy_total))
        for _, reward_signal in self.policy.reward_signals.items():
            update_stats = reward_signal.update(
                self.training_buffer.update_buffer, n_sequences
            )
            for stat, val in update_stats.items():
                self.stats[stat].append(val)
        if self.policy.bc_module:
            update_stats = self.policy.bc_module.update()
            for stat, val in update_stats.items():
                self.stats[stat].append(val)
        self.training_buffer.reset_update_buffer()
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

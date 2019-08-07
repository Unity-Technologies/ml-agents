import logging
import numpy as np
from typing import Any, Dict
import tensorflow as tf

from mlagents.envs.timers import timed
from mlagents.trainers import BrainInfo, ActionInfo
from mlagents.trainers.models import EncoderType
from mlagents.trainers.ppo.models import PPOModel
from mlagents.trainers.tf_policy import TFPolicy
from mlagents.trainers.components.reward_signals.reward_signal_factory import (
    create_reward_signal,
)
from mlagents.trainers.components.bc.module import BCModule

logger = logging.getLogger("mlagents.trainers")


class PPOPolicy(TFPolicy):
    def __init__(self, seed, brain, trainer_params, is_training, load):
        """
        Policy for Proximal Policy Optimization Networks.
        :param seed: Random seed.
        :param brain: Assigned Brain object.
        :param trainer_params: Defined training parameters.
        :param is_training: Whether the model should be trained.
        :param load: Whether a pre-trained model will be loaded or a new one created.
        """
        super().__init__(seed, brain, trainer_params)

        reward_signal_configs = trainer_params["reward_signals"]

        self.create_model(brain, trainer_params, reward_signal_configs, seed)

        self.reward_signals = {}
        with self.graph.as_default():
            # Create reward signals
            for reward_signal, config in reward_signal_configs.items():
                self.reward_signals[reward_signal] = create_reward_signal(
                    self, reward_signal, config
                )

            # Create pretrainer if needed
            if "pretraining" in trainer_params:
                BCModule.check_config(trainer_params["pretraining"])
                self.bc_module = BCModule(
                    self,
                    policy_learning_rate=trainer_params["learning_rate"],
                    default_batch_size=trainer_params["batch_size"],
                    default_num_epoch=trainer_params["num_epoch"],
                    **trainer_params["pretraining"],
                )
            else:
                self.bc_module = None

        if load:
            self._load_graph()
        else:
            self._initialize_graph()

        self.inference_dict = {
            "action": self.model.output,
            "log_probs": self.model.all_log_probs,
            "value": self.model.value_heads,
            "entropy": self.model.entropy,
            "learning_rate": self.model.learning_rate,
        }
        if self.use_continuous_act:
            self.inference_dict["pre_action"] = self.model.output_pre
        if self.use_recurrent:
            self.inference_dict["memory_out"] = self.model.memory_out
        if (
            is_training
            and self.use_vec_obs
            and trainer_params["normalize"]
            and not load
        ):
            self.inference_dict["update_mean"] = self.model.update_normalization
        # Use absolute value for cleaner reporting
        self.total_policy_loss = self.model.abs_policy_loss

        self.update_dict = {
            "value_loss": self.model.value_loss,
            "policy_loss": self.total_policy_loss,
            "update_batch": self.model.update_batch,
        }

        for _, reward_signal in self.reward_signals.items():
            # Add reward signal update to update_dict
            self.update_dict.update(reward_signal.update_dict)

        self.stats_name_to_update_name = {
            "Losses/Value Loss": "value_loss",
            "Losses/Policy Loss": "policy_loss",
        }

    def create_model(self, brain, trainer_params, reward_signal_configs, seed):
        """
        Create PPO model
        :param brain: Assigned Brain object.
        :param trainer_params: Defined training parameters.
        :param reward_signal_configs: Reward signal config
        :param seed: Random seed.
        """
        with self.graph.as_default():
            self.model = PPOModel(
                brain=brain,
                lr=float(trainer_params["learning_rate"]),
                h_size=int(trainer_params["hidden_units"]),
                epsilon=float(trainer_params["epsilon"]),
                beta=float(trainer_params["beta"]),
                max_step=float(trainer_params["max_steps"]),
                normalize=trainer_params["normalize"],
                use_recurrent=trainer_params["use_recurrent"],
                num_layers=int(trainer_params["num_layers"]),
                m_size=self.m_size,
                seed=seed,
                stream_names=list(reward_signal_configs.keys()),
                vis_encode_type=EncoderType(
                    trainer_params.get("vis_encode_type", "simple")
                ),
            )
            self.model.create_ppo_optimizer()

    @timed
    def evaluate(self, brain_info):
        """
        Evaluates policy for the agent experiences provided.
        :param brain_info: BrainInfo object containing inputs.
        :return: Outputs from network as defined by self.inference_dict.
        """
        feed_dict = {
            self.model.batch_size: len(brain_info.vector_observations),
            self.model.sequence_length: 1,
        }
        epsilon = None
        if self.use_recurrent:
            if not self.use_continuous_act:
                feed_dict[
                    self.model.prev_action
                ] = brain_info.previous_vector_actions.reshape(
                    [-1, len(self.model.act_size)]
                )
            if brain_info.memories.shape[1] == 0:
                brain_info.memories = self.make_empty_memory(len(brain_info.agents))
            feed_dict[self.model.memory_in] = brain_info.memories
        if self.use_continuous_act:
            epsilon = np.random.normal(
                size=(len(brain_info.vector_observations), self.model.act_size[0])
            )
            feed_dict[self.model.epsilon] = epsilon
        feed_dict = self.fill_eval_dict(feed_dict, brain_info)
        run_out = self._execute_model(feed_dict, self.inference_dict)
        if self.use_continuous_act:
            run_out["random_normal_epsilon"] = epsilon
        return run_out

    @timed
    def update(self, mini_batch, num_sequences):
        """
        Performs update on model.
        :param mini_batch: Batch of experiences.
        :param num_sequences: Number of sequences to process.
        :return: Results of update.
        """
        feed_dict = {}
        stats_needed = {}
        update_stats = {}
        feed_dict.update(
            self.construct_feed_dict(self.model, mini_batch, num_sequences)
        )
        stats_needed.update(self.stats_name_to_update_name)
        # Collect feed dicts for all reward signals.
        for _, reward_signal in self.reward_signals.items():
            feed_dict.update(reward_signal.prepare_update(mini_batch, num_sequences))
            stats_needed.update(reward_signal.stats_name_to_update_name)

        update_vals = self._execute_model(feed_dict, self.update_dict)
        for stat_name, update_name in stats_needed.items():
            update_stats[stat_name] = update_vals[update_name]
        return update_stats

    def construct_feed_dict(self, model, mini_batch, num_sequences):
        feed_dict = {
            self.model.batch_size: num_sequences,
            self.model.sequence_length: self.sequence_length,
            self.model.mask_input: mini_batch["masks"],
            self.model.advantage: mini_batch["advantages"],
            self.model.all_old_log_probs: mini_batch["action_probs"],
        }
        for name in self.reward_signals:
            feed_dict[model.returns_holders[name]] = mini_batch[
                "{}_returns".format(name)
            ]
            feed_dict[model.old_values[name]] = mini_batch[
                "{}_value_estimates".format(name)
            ]

        if self.use_continuous_act:

            feed_dict[model.output_pre] = mini_batch["actions_pre"]
            feed_dict[model.epsilon] = mini_batch["random_normal_epsilon"]
        else:
            feed_dict[model.action_holder] = mini_batch["actions"]
            if self.use_recurrent:
                feed_dict[model.prev_action] = mini_batch["prev_action"]
            feed_dict[model.action_masks] = mini_batch["action_mask"]
        if self.use_vec_obs:
            feed_dict[model.vector_in] = mini_batch["vector_obs"]
        if self.model.vis_obs_size > 0:
            for i, _ in enumerate(self.model.visual_in):
                feed_dict[model.visual_in[i]] = mini_batch["visual_obs%d" % i]
        if self.use_recurrent:
            mem_in = [
                mini_batch["memory"][i]
                for i in range(0, len(mini_batch["memory"]), self.sequence_length)
            ]
            feed_dict[model.memory_in] = mem_in
        return feed_dict

    def get_value_estimates(
        self, brain_info: BrainInfo, idx: int, done: bool
    ) -> Dict[str, float]:
        """
        Generates value estimates for bootstrapping.
        :param brain_info: BrainInfo to be used for bootstrapping.
        :param idx: Index in BrainInfo of agent.
        :param done: Whether or not this is the last element of the episode, in which case the value estimate will be 0.
        :return: The value estimate dictionary with key being the name of the reward signal and the value the
        corresponding value estimate.
        """

        feed_dict: Dict[tf.Tensor, Any] = {
            self.model.batch_size: 1,
            self.model.sequence_length: 1,
        }
        for i in range(len(brain_info.visual_observations)):
            feed_dict[self.model.visual_in[i]] = [
                brain_info.visual_observations[i][idx]
            ]
        if self.use_vec_obs:
            feed_dict[self.model.vector_in] = [brain_info.vector_observations[idx]]
        if self.use_recurrent:
            if brain_info.memories.shape[1] == 0:
                brain_info.memories = self.make_empty_memory(len(brain_info.agents))
            feed_dict[self.model.memory_in] = [brain_info.memories[idx]]
        if not self.use_continuous_act and self.use_recurrent:
            feed_dict[self.model.prev_action] = [
                brain_info.previous_vector_actions[idx]
            ]
        value_estimates = self.sess.run(self.model.value_heads, feed_dict)

        value_estimates = {k: float(v) for k, v in value_estimates.items()}

        # If we're done, reassign all of the value estimates that need terminal states.
        if done:
            for k in value_estimates:
                if self.reward_signals[k].use_terminal_states:
                    value_estimates[k] = 0.0

        return value_estimates

    def get_action(self, brain_info: BrainInfo) -> ActionInfo:
        """
        Decides actions given observations information, and takes them in environment.
        :param brain_info: A dictionary of brain names and BrainInfo from environment.
        :return: an ActionInfo containing action, memories, values and an object
        to be passed to add experiences
        """
        if len(brain_info.agents) == 0:
            return ActionInfo([], [], [], None, None)

        run_out = self.evaluate(brain_info)
        mean_values = np.mean(
            np.array(list(run_out.get("value").values())), axis=0
        ).flatten()

        return ActionInfo(
            action=run_out.get("action"),
            memory=run_out.get("memory_out"),
            text=None,
            value=mean_values,
            outputs=run_out,
        )

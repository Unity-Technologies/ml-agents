import logging
from typing import Dict, Any, Optional
import numpy as np
from mlagents.tf_utils import tf

from mlagents_envs.timers import timed
from mlagents.trainers.brain import BrainInfo, BrainParameters
from mlagents.trainers.models import EncoderType, LearningRateSchedule
from mlagents.trainers.sac.models import SACModel
from mlagents.trainers.tf_policy import TFPolicy
from mlagents.trainers.components.reward_signals.reward_signal_factory import (
    create_reward_signal,
)
from mlagents.trainers.components.reward_signals import RewardSignal
from mlagents.trainers.components.bc.module import BCModule

logger = logging.getLogger("mlagents.trainers")


class SACPolicy(TFPolicy):
    def __init__(
        self,
        seed: int,
        brain: BrainParameters,
        trainer_params: Dict[str, Any],
        is_training: bool,
        load: bool,
    ) -> None:
        """
        Policy for Proximal Policy Optimization Networks.
        :param seed: Random seed.
        :param brain: Assigned Brain object.
        :param trainer_params: Defined training parameters.
        :param is_training: Whether the model should be trained.
        :param load: Whether a pre-trained model will be loaded or a new one created.
        """
        super().__init__(seed, brain, trainer_params)

        reward_signal_configs = {}
        for key, rsignal in trainer_params["reward_signals"].items():
            if type(rsignal) is dict:
                reward_signal_configs[key] = rsignal

        self.inference_dict: Dict[str, tf.Tensor] = {}
        self.update_dict: Dict[str, tf.Tensor] = {}
        self.create_model(
            brain, trainer_params, reward_signal_configs, is_training, load, seed
        )
        self.create_reward_signals(reward_signal_configs)

        self.stats_name_to_update_name = {
            "Losses/Value Loss": "value_loss",
            "Losses/Policy Loss": "policy_loss",
            "Losses/Q1 Loss": "q1_loss",
            "Losses/Q2 Loss": "q2_loss",
            "Policy/Entropy Coeff": "entropy_coef",
        }

        with self.graph.as_default():
            # Create pretrainer if needed
            self.bc_module: Optional[BCModule] = None
            if "behavioral_cloning" in trainer_params:
                BCModule.check_config(trainer_params["behavioral_cloning"])
                self.bc_module = BCModule(
                    self,
                    policy_learning_rate=trainer_params["learning_rate"],
                    default_batch_size=trainer_params["batch_size"],
                    default_num_epoch=1,
                    samples_per_update=trainer_params["batch_size"],
                    **trainer_params["behavioral_cloning"],
                )
                # SAC-specific setting - we don't want to do a whole epoch each update!
                if "samples_per_update" in trainer_params["behavioral_cloning"]:
                    logger.warning(
                        "Pretraining: Samples Per Update is not a valid setting for SAC."
                    )
                    self.bc_module.samples_per_update = 1

        if load:
            self._load_graph()
        else:
            self._initialize_graph()
            self.sess.run(self.model.target_init_op)

        # Disable terminal states for certain reward signals to avoid survivor bias
        for name, reward_signal in self.reward_signals.items():
            if not reward_signal.use_terminal_states:
                self.sess.run(self.model.disable_use_dones[name])

    def create_model(
        self,
        brain: BrainParameters,
        trainer_params: Dict[str, Any],
        reward_signal_configs: Dict[str, Any],
        is_training: bool,
        load: bool,
        seed: int,
    ) -> None:
        with self.graph.as_default():
            self.model = SACModel(
                brain,
                lr=float(trainer_params["learning_rate"]),
                lr_schedule=LearningRateSchedule(
                    trainer_params.get("learning_rate_schedule", "constant")
                ),
                h_size=int(trainer_params["hidden_units"]),
                init_entcoef=float(trainer_params["init_entcoef"]),
                max_step=float(trainer_params["max_steps"]),
                normalize=trainer_params["normalize"],
                use_recurrent=trainer_params["use_recurrent"],
                num_layers=int(trainer_params["num_layers"]),
                m_size=self.m_size,
                seed=seed,
                stream_names=list(reward_signal_configs.keys()),
                tau=float(trainer_params["tau"]),
                gammas=[_val["gamma"] for _val in reward_signal_configs.values()],
                vis_encode_type=EncoderType(
                    trainer_params.get("vis_encode_type", "simple")
                ),
            )
            self.model.create_sac_optimizers()

        self.inference_dict.update(
            {
                "action": self.model.output,
                "log_probs": self.model.all_log_probs,
                "entropy": self.model.entropy,
                "learning_rate": self.model.learning_rate,
            }
        )
        if self.use_continuous_act:
            self.inference_dict["pre_action"] = self.model.output_pre
        if self.use_recurrent:
            self.inference_dict["memory_out"] = self.model.memory_out

        self.update_dict.update(
            {
                "value_loss": self.model.total_value_loss,
                "policy_loss": self.model.policy_loss,
                "q1_loss": self.model.q1_loss,
                "q2_loss": self.model.q2_loss,
                "entropy_coef": self.model.ent_coef,
                "entropy": self.model.entropy,
                "update_batch": self.model.update_batch_policy,
                "update_value": self.model.update_batch_value,
                "update_entropy": self.model.update_batch_entropy,
            }
        )

    def create_reward_signals(self, reward_signal_configs: Dict[str, Any]) -> None:
        """
        Create reward signals
        :param reward_signal_configs: Reward signal config.
        """
        self.reward_signals: Dict[str, RewardSignal] = {}
        with self.graph.as_default():
            # Create reward signals
            for reward_signal, config in reward_signal_configs.items():
                if type(config) is dict:
                    self.reward_signals[reward_signal] = create_reward_signal(
                        self, self.model, reward_signal, config
                    )

    def evaluate(self, brain_info: BrainInfo) -> Dict[str, np.ndarray]:
        """
        Evaluates policy for the agent experiences provided.
        :param brain_info: BrainInfo object containing inputs.
        :return: Outputs from network as defined by self.inference_dict.
        """
        feed_dict = {
            self.model.batch_size: len(brain_info.vector_observations),
            self.model.sequence_length: 1,
        }
        if self.use_recurrent:
            if not self.use_continuous_act:
                feed_dict[self.model.prev_action] = self.retrieve_previous_action(
                    brain_info.agents
                )
            feed_dict[self.model.memory_in] = self.retrieve_memories(brain_info.agents)

        feed_dict = self.fill_eval_dict(feed_dict, brain_info)
        run_out = self._execute_model(feed_dict, self.inference_dict)
        return run_out

    @timed
    def update(
        self, mini_batch: Dict[str, Any], num_sequences: int
    ) -> Dict[str, float]:
        """
        Updates model using buffer.
        :param num_sequences: Number of trajectories in batch.
        :param mini_batch: Experience batch.
        :param update_target: Whether or not to update target value network
        :param reward_signal_mini_batches: Minibatches to use for updating the reward signals,
            indexed by name. If none, don't update the reward signals.
        :return: Output from update process.
        """
        feed_dict = self.construct_feed_dict(self.model, mini_batch, num_sequences)
        stats_needed = self.stats_name_to_update_name
        update_stats: Dict[str, float] = {}
        update_vals = self._execute_model(feed_dict, self.update_dict)
        for stat_name, update_name in stats_needed.items():
            update_stats[stat_name] = update_vals[update_name]
        # Update target network. By default, target update happens at every policy update.
        self.sess.run(self.model.target_update_op)
        return update_stats

    def update_reward_signals(
        self, reward_signal_minibatches: Dict[str, Dict], num_sequences: int
    ) -> Dict[str, float]:
        """
        Only update the reward signals.
        :param reward_signal_mini_batches: Minibatches to use for updating the reward signals,
            indexed by name. If none, don't update the reward signals.
        """
        # Collect feed dicts for all reward signals.
        feed_dict: Dict[tf.Tensor, Any] = {}
        update_dict: Dict[str, tf.Tensor] = {}
        update_stats: Dict[str, float] = {}
        stats_needed: Dict[str, str] = {}
        if reward_signal_minibatches:
            self.add_reward_signal_dicts(
                feed_dict,
                update_dict,
                stats_needed,
                reward_signal_minibatches,
                num_sequences,
            )
        update_vals = self._execute_model(feed_dict, update_dict)
        for stat_name, update_name in stats_needed.items():
            update_stats[stat_name] = update_vals[update_name]
        return update_stats

    def add_reward_signal_dicts(
        self,
        feed_dict: Dict[tf.Tensor, Any],
        update_dict: Dict[str, tf.Tensor],
        stats_needed: Dict[str, str],
        reward_signal_minibatches: Dict[str, Dict],
        num_sequences: int,
    ) -> None:
        """
        Adds the items needed for reward signal updates to the feed_dict and stats_needed dict.
        :param feed_dict: Feed dict needed update
        :param update_dit: Update dict that needs update
        :param stats_needed: Stats needed to get from the update.
        :param reward_signal_minibatches: Minibatches to use for updating the reward signals,
            indexed by name.
        """
        for name, r_mini_batch in reward_signal_minibatches.items():
            feed_dict.update(
                self.reward_signals[name].prepare_update(
                    self.model, r_mini_batch, num_sequences
                )
            )
            update_dict.update(self.reward_signals[name].update_dict)
            stats_needed.update(self.reward_signals[name].stats_name_to_update_name)

    def construct_feed_dict(
        self, model: SACModel, mini_batch: Dict[str, Any], num_sequences: int
    ) -> Dict[tf.Tensor, Any]:
        """
        Builds the feed dict for updating the SAC model.
        :param model: The model to update. May be different when, e.g. using multi-GPU.
        :param mini_batch: Mini-batch to use to update.
        :param num_sequences: Number of LSTM sequences in mini_batch.
        """
        feed_dict = {
            self.model.batch_size: num_sequences,
            self.model.sequence_length: self.sequence_length,
            self.model.next_sequence_length: self.sequence_length,
            self.model.mask_input: mini_batch["masks"],
        }
        for name in self.reward_signals:
            feed_dict[model.rewards_holders[name]] = mini_batch[
                "{}_rewards".format(name)
            ]

        if self.use_continuous_act:
            feed_dict[model.action_holder] = mini_batch["actions"]
        else:
            feed_dict[model.action_holder] = mini_batch["actions"]
            if self.use_recurrent:
                feed_dict[model.prev_action] = mini_batch["prev_action"]
            feed_dict[model.action_masks] = mini_batch["action_mask"]
        if self.use_vec_obs:
            feed_dict[model.vector_in] = mini_batch["vector_obs"]
            feed_dict[model.next_vector_in] = mini_batch["next_vector_in"]
        if self.model.vis_obs_size > 0:
            for i, _ in enumerate(model.visual_in):
                _obs = mini_batch["visual_obs%d" % i]
                feed_dict[model.visual_in[i]] = _obs
            for i, _ in enumerate(model.next_visual_in):
                _obs = mini_batch["next_visual_obs%d" % i]
                feed_dict[model.next_visual_in[i]] = _obs
        if self.use_recurrent:
            mem_in = [
                mini_batch["memory"][i]
                for i in range(0, len(mini_batch["memory"]), self.sequence_length)
            ]
            # LSTM shouldn't have sequence length <1, but stop it from going out of the index if true.
            offset = 1 if self.sequence_length > 1 else 0
            next_mem_in = [
                mini_batch["memory"][i][
                    : self.m_size // 4
                ]  # only pass value part of memory to target network
                for i in range(offset, len(mini_batch["memory"]), self.sequence_length)
            ]
            feed_dict[model.memory_in] = mem_in
            feed_dict[model.next_memory_in] = next_mem_in
        feed_dict[model.dones_holder] = mini_batch["done"]
        return feed_dict

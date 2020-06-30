from typing import Optional, Any, Dict, cast
import numpy as np
import os
from mlagents.tf_utils import tf
from mlagents_envs.timers import timed
from mlagents.trainers.models import ModelUtils, EncoderType
from mlagents.trainers.policy.tf_policy import TFPolicy
from mlagents.trainers.components.reward_signals.curiosity.model import CuriosityModel
from mlagents.trainers.policy.transfer_policy import TransferPolicy
from mlagents.trainers.optimizer.tf_optimizer import TFOptimizer
from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.settings import TrainerSettings, PPOSettings, PPOTransferSettings
import tf_slim as slim

class PPOTransferOptimizer(TFOptimizer):
    def __init__(self, policy: TransferPolicy, trainer_params: TrainerSettings):
        """
        Takes a Policy and a Dict of trainer parameters and creates an Optimizer around the policy.
        The PPO optimizer has a value esåtimator and a loss function.
        :param policy: A TFPolicy object that will be updated by this PPO Optimizer.
        :param trainer_params: Trainer parameters dictionary that specifies the properties of the trainer.
        """
        hyperparameters: PPOTransferSettings = cast(
            PPOTransferSettings, trainer_params.hyperparameters
        )

        self.separate_value_train = hyperparameters.separate_value_train
        self.separate_policy_train = hyperparameters.separate_policy_train
        self.use_var_encoder = hyperparameters.use_var_encoder
        self.use_var_predict = hyperparameters.use_var_predict
        self.with_prior = hyperparameters.with_prior
        self.use_inverse_model = hyperparameters.use_inverse_model
        self.predict_return = hyperparameters.predict_return
        self.reuse_encoder = hyperparameters.reuse_encoder

        self.use_alter = hyperparameters.use_alter
        self.in_batch_alter = hyperparameters.in_batch_alter
        self.in_epoch_alter = hyperparameters.in_epoch_alter
        
        self.train_type = hyperparameters.train_type
        
        # Transfer
        self.use_transfer = hyperparameters.use_transfer
        self.transfer_path = hyperparameters.transfer_path #"results/BallSingle_nosep_cmodel_small/3DBall"
        self.smart_transfer = hyperparameters.smart_transfer
        self.conv_thres = hyperparameters.conv_thres
        self.transfer_type = hyperparameters.transfer_type

        self.ppo_update_dict: Dict[str, tf.Tensor] = {}
        self.model_update_dict: Dict[str, tf.Tensor] = {}

        # Create the graph here to give more granular control of the TF graph to the Optimizer.
        policy.create_tf_graph(hyperparameters.encoder_layers, hyperparameters.policy_layers, 
            self.use_transfer, self.separate_policy_train, self.use_var_encoder, self.use_var_predict, 
            self.predict_return, self.use_inverse_model, self.reuse_encoder)
        
        with policy.graph.as_default():
            super().__init__(policy, trainer_params)

            lr = float(hyperparameters.learning_rate)
            self._schedule = hyperparameters.learning_rate_schedule
            epsilon = float(hyperparameters.epsilon)
            beta = float(hyperparameters.beta)
            max_step = float(trainer_params.max_steps)
            policy_network_settings = policy.network_settings
            h_size = int(policy_network_settings.hidden_units)
            num_layers = policy_network_settings.num_layers
            vis_encode_type = policy_network_settings.vis_encode_type
            self.burn_in_ratio = 0.0
            
            self.num_updates = 0
            self.alter_every = 400
            self.copy_every = 10
            self.old_loss = np.inf
            self.update_mode = "model"

            self.stream_names = list(self.reward_signals.keys())

            self.tf_optimizer: Optional[tf.train.AdamOptimizer] = None
            self.grads = None
            self.update_batch: Optional[tf.Operation] = None

            self.stats_name_to_update_name = {
                "Losses/Value Loss": "value_loss",
                "Losses/Policy Loss": "policy_loss",
                "Losses/Model Loss": "model_loss",
                "Policy/Learning Rate": "learning_rate",
                "Policy/Epsilon": "decay_epsilon",
                "Policy/Beta": "decay_beta",
            }
            if self.policy.use_recurrent:
                self.m_size = self.policy.m_size
                self.memory_in = tf.placeholder(
                    shape=[None, self.m_size],
                    dtype=tf.float32,
                    name="recurrent_value_in",
                )
            if num_layers < 1:
                num_layers = 1

            with tf.variable_scope("value"):
                if policy.use_continuous_act:
                    self._create_cc_critic(h_size, hyperparameters.value_layers, vis_encode_type)
                else:
                    self._create_dc_critic(h_size, hyperparameters.value_layers, vis_encode_type)
            
            with tf.variable_scope("optimizer/"):
                self.learning_rate = ModelUtils.create_schedule(
                    self._schedule,
                    lr,
                    self.policy.global_step,
                    int(max_step),
                    min_value=1e-10,
                )
                self._create_losses(
                    self.policy.total_log_probs,
                    self.old_log_probs,
                    self.value_heads,
                    self.policy.entropy,
                    self.policy.targ_encoder,
                    self.policy.predict,
                    self.policy.encoder_distribution,
                    beta,
                    epsilon,
                    lr,
                    max_step,
                )
                self._create_ppo_optimizer_ops()

                self.update_dict.update(
                    {
                        "value_loss": self.value_loss,
                        "policy_loss": self.abs_policy_loss,
                        "model_loss": self.model_loss,
                        "update_batch": self.update_batch,
                        "learning_rate": self.learning_rate,
                        "decay_epsilon": self.decay_epsilon,
                        "decay_beta": self.decay_beta,
                    }
                )

                if self.use_alter or self.smart_transfer or self.in_batch_alter or self.in_epoch_alter:
                    self._init_alter_update()
            
            self.policy.initialize_or_load()
            if self.use_transfer:
                self.policy.load_graph_partial(self.transfer_path, self.transfer_type)
            self.policy.get_encoder_weights()
            self.policy.get_policy_weights()
            # saver = tf.train.Saver()
            # model_checkpoint = os.path.join(self.transfer_path, f"model-4000544.ckpt")
            # saver.restore(self.sess, model_checkpoint)
            # self.policy._set_step(0)

            slim.model_analyzer.analyze_vars(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), print_info=True)
            
            print("All variables in the graph:")
            for variable in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                print(variable.name)
            # tf.summary.FileWriter(self.policy.model_path, self.sess.graph)

    
    def _create_cc_critic(
        self, h_size: int, num_layers: int, vis_encode_type: EncoderType
    ) -> None:
        """
        Creates Continuous control critic (value) network.
        :param h_size: Size of hidden linear layers.
        :param num_layers: Number of hidden linear layers.
        :param vis_encode_type: The type of visual encoder to use.
        """
        if self.separate_value_train:
            input_state = tf.stop_gradient(self.policy.encoder)
        else:
            input_state = self.policy.encoder
        hidden_value = ModelUtils.create_vector_observation_encoder(
            input_state,
            h_size,
            ModelUtils.swish,
            num_layers,
            scope=f"main_graph",
            reuse=False
        )
        self.value_heads, self.value = ModelUtils.create_value_heads(
            self.stream_names, hidden_value
        )
        self.all_old_log_probs = tf.placeholder(
            shape=[None, sum(self.policy.act_size)],
            dtype=tf.float32,
            name="old_probabilities",
        )

        self.old_log_probs = tf.reduce_sum(
            (tf.identity(self.all_old_log_probs)), axis=1, keepdims=True
        )

    def _create_dc_critic(
        self, h_size: int, num_layers: int, vis_encode_type: EncoderType
    ) -> None:
        """
        Creates Discrete control critic (value) network.
        :param h_size: Size of hidden linear layers.
        :param num_layers: Number of hidden linear layers.
        :param vis_encode_type: The type of visual encoder to use.
        """
        if self.separate_value_train:
            input_state = tf.stop_gradient(self.policy.encoder)
        else:
            input_state = self.policy.encoder
        hidden_value = ModelUtils.create_vector_observation_encoder(
            input_state,
            h_size,
            ModelUtils.swish,
            num_layers,
            scope=f"main_graph",
            reuse=False
        )
        self.value_heads, self.value = ModelUtils.create_value_heads(
            self.stream_names, hidden_value
        )

        self.all_old_log_probs = tf.placeholder(
            shape=[None, sum(self.policy.act_size)],
            dtype=tf.float32,
            name="old_probabilities",
        )

        # Break old log probs into separate branches
        old_log_prob_branches = ModelUtils.break_into_branches(
            self.all_old_log_probs, self.policy.act_size
        )

        _, _, old_normalized_logits = ModelUtils.create_discrete_action_masking_layer(
            old_log_prob_branches, self.policy.action_masks, self.policy.act_size
        )

        action_idx = [0] + list(np.cumsum(self.policy.act_size))

        self.old_log_probs = tf.reduce_sum(
            (
                tf.stack(
                    [
                        -tf.nn.softmax_cross_entropy_with_logits_v2(
                            labels=self.policy.selected_actions[
                                :, action_idx[i] : action_idx[i + 1]
                            ],
                            logits=old_normalized_logits[
                                :, action_idx[i] : action_idx[i + 1]
                            ],
                        )
                        for i in range(len(self.policy.act_size))
                    ],
                    axis=1,
                )
            ),
            axis=1,
            keepdims=True,
        )

    def _create_losses(
        self, probs, old_probs, value_heads, entropy, targ_encoder, predict, encoder_distribution,
         beta, epsilon, lr, max_step
    ):
        """
        Creates training-specific Tensorflow ops for PPO models.
        :param probs: Current policy probabilities
        :param old_probs: Past policy probabilities
        :param value_heads: Value estimate tensors from each value stream
        :param beta: Entropy regularization strength
        :param entropy: Current policy entropy
        :param epsilon: Value for policy-divergence threshold
        :param lr: Learning rate
        :param max_step: Total number of training steps.
        """
        self.returns_holders = {}
        self.old_values = {}
        for name in value_heads.keys():
            returns_holder = tf.placeholder(
                shape=[None], dtype=tf.float32, name="{}_returns".format(name)
            )
            old_value = tf.placeholder(
                shape=[None], dtype=tf.float32, name="{}_value_estimate".format(name)
            )
            self.returns_holders[name] = returns_holder
            self.old_values[name] = old_value
        self.advantage = tf.placeholder(
            shape=[None], dtype=tf.float32, name="advantages"
        )
        advantage = tf.expand_dims(self.advantage, -1)

        self.decay_epsilon = ModelUtils.create_schedule(
            self._schedule, epsilon, self.policy.global_step, max_step, min_value=0.1
        )
        self.decay_beta = ModelUtils.create_schedule(
            self._schedule, beta, self.policy.global_step, max_step, min_value=1e-5
        )

        value_losses = []
        for name, head in value_heads.items():
            clipped_value_estimate = self.old_values[name] + tf.clip_by_value(
                tf.reduce_sum(head, axis=1) - self.old_values[name],
                -self.decay_epsilon,
                self.decay_epsilon,
            )
            v_opt_a = tf.squared_difference(
                self.returns_holders[name], tf.reduce_sum(head, axis=1)
            )
            v_opt_b = tf.squared_difference(
                self.returns_holders[name], clipped_value_estimate
            )
            value_loss = tf.reduce_mean(
                tf.dynamic_partition(tf.maximum(v_opt_a, v_opt_b), self.policy.mask, 2)[
                    1
                ]
            )
            value_losses.append(value_loss)
        self.value_loss = tf.reduce_mean(value_losses)

        r_theta = tf.exp(probs - old_probs)
        p_opt_a = r_theta * advantage
        p_opt_b = (
            tf.clip_by_value(
                r_theta, 1.0 - self.decay_epsilon, 1.0 + self.decay_epsilon
            )
            * advantage
        )
        self.policy_loss = -tf.reduce_mean(
            tf.dynamic_partition(tf.minimum(p_opt_a, p_opt_b), self.policy.mask, 2)[1]
        )
        # For cleaner stats reporting
        self.abs_policy_loss = tf.abs(self.policy_loss)

        # encoder and predict loss
        self.dis_returns = tf.placeholder(
            shape=[None], dtype=tf.float32, name="dis_returns"
        )
        target = tf.concat([targ_encoder, tf.expand_dims(self.dis_returns, -1)], axis=1)
        if self.predict_return:
            self.model_loss = tf.reduce_mean(tf.squared_difference(predict, target)) 
        else:
            self.model_loss = tf.reduce_mean(tf.squared_difference(predict, targ_encoder)) 
        if self.with_prior:
            if self.use_var_encoder:
                self.model_loss += encoder_distribution.kl_standard()
            if self.use_var_predict:
                self.model_loss += self.policy.predict_distribution.kl_standard()

        if self.use_inverse_model:
            self.model_loss += self.policy.inverse_loss
        # self.model_loss = 0.2 * self.policy.forward_loss + 0.8 * self.policy.inverse_loss
        self.loss = (
            self.policy_loss
            + self.model_loss
            + 0.5 * self.value_loss
            - self.decay_beta
            * tf.reduce_mean(tf.dynamic_partition(entropy, self.policy.mask, 2)[1])
        )

        self.ppo_loss = (
            self.policy_loss
            + 0.5 * self.value_loss
            - self.decay_beta
            * tf.reduce_mean(tf.dynamic_partition(entropy, self.policy.mask, 2)[1])
        )

    def _create_ppo_optimizer_ops(self):
        if self.use_transfer:
            if self.transfer_type == "dynamics":
                if self.train_type == "all":
                    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                elif self.train_type == "encoding":
                    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "encoding")
                print("trainable", train_vars)
                # train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "encoding")
                # train_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "policy")
                # train_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "value")
                # train_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "policy/mu")
                # train_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "policy/log_std")
                # train_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "value/extrinsic_value")
            elif self.transfer_type == "observation":
                if self.train_type == "all":
                    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                elif self.train_type == "policy":
                    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "policy") \
                        + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "predict") \
                        + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "inverse") \
                        + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "value") 
                        # + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "encoding/latent")
        else:
            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            print("trainable", train_vars)

        self.tf_optimizer = self.create_optimizer_op(self.learning_rate)
        self.grads = self.tf_optimizer.compute_gradients(self.loss, var_list=train_vars)
        self.update_batch = self.tf_optimizer.minimize(self.loss, var_list=train_vars)
        
        
    def _init_alter_update(self):
        
        if self.use_alter:
            policy_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "policy")
            policy_train_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "value")
            model_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "encoding")
            self.ppo_optimizer = self.create_optimizer_op(self.learning_rate)
            self.ppo_grads = self.ppo_optimizer.compute_gradients(self.loss, var_list=policy_train_vars)
            self.ppo_update_batch = self.ppo_optimizer.minimize(self.loss, var_list=policy_train_vars)

            self.model_optimizer = self.create_optimizer_op(self.learning_rate)
            self.model_grads = self.model_optimizer.compute_gradients(self.loss, var_list=model_train_vars)
            self.model_update_batch = self.model_optimizer.minimize(self.loss, var_list=model_train_vars)
        else:
            if self.train_type == "all":
                train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            elif self.train_type == "encoding":
                train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "encoding")
                # train_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "target_enc")
            elif self.train_type == "policy":
                train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "policy") \
                    + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "predict") \
                    + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "inverse") \
                    + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "value") 
            self.ppo_optimizer = self.create_optimizer_op(self.learning_rate)
            self.ppo_grads = self.ppo_optimizer.compute_gradients(self.ppo_loss, var_list=train_vars)
            self.ppo_update_batch = self.ppo_optimizer.minimize(self.ppo_loss, var_list=train_vars)

            self.model_optimizer = self.create_optimizer_op(self.learning_rate)
            self.model_grads = self.model_optimizer.compute_gradients(self.model_loss, var_list=train_vars)
            self.model_update_batch = self.model_optimizer.minimize(self.model_loss, var_list=train_vars)

        self.ppo_update_dict.update(
            {
                "value_loss": self.value_loss,
                "policy_loss": self.abs_policy_loss,
                "update_batch": self.ppo_update_batch,
                "learning_rate": self.learning_rate,
                "decay_epsilon": self.decay_epsilon,
                "decay_beta": self.decay_beta,
            }
        )

        self.model_update_dict.update(
            {
                "model_loss": self.model_loss,
                "update_batch": self.model_update_batch,
                "learning_rate": self.learning_rate,
                "decay_epsilon": self.decay_epsilon,
                "decay_beta": self.decay_beta,
            }
        )

    @timed
    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        """
        Performs update on model.
        :param mini_batch: Batch of experiences.
        :param num_sequences: Number of sequences to process.
        :return: Results of update.
        """
        feed_dict = self._construct_feed_dict(batch, num_sequences)
        stats_needed = self.stats_name_to_update_name
        update_stats = {}
        # Collect feed dicts for all reward signals.
        for _, reward_signal in self.reward_signals.items():
            feed_dict.update(
                reward_signal.prepare_update(self.policy, batch, num_sequences)
            )
            stats_needed.update(reward_signal.stats_name_to_update_name)
        
        if self.use_alter:
            # if self.num_updates / self.alter_every == 0:
            #     update_vals = self._execute_model(feed_dict, self.update_dict)
            #     if self.num_updates % self.alter_every == 0:
            #         print("start update all", self.num_updates)
            if (self.num_updates / self.alter_every) % 2 == 0:
                stats_needed = {
                    "Losses/Model Loss": "model_loss",
                    "Policy/Learning Rate": "learning_rate",
                    "Policy/Epsilon": "decay_epsilon",
                    "Policy/Beta": "decay_beta",
                }
                update_vals = self._execute_model(feed_dict, self.model_update_dict)
                if self.num_updates % self.alter_every == 0:
                    print("start update model", self.num_updates)
            else: # (self.num_updates / self.alter_every) % 2 == 0:
                stats_needed = {
                    "Losses/Value Loss": "value_loss",
                    "Losses/Policy Loss": "policy_loss",
                    "Policy/Learning Rate": "learning_rate",
                    "Policy/Epsilon": "decay_epsilon",
                    "Policy/Beta": "decay_beta",
                }
                update_vals = self._execute_model(feed_dict, self.ppo_update_dict)
                if self.num_updates % self.alter_every == 0:
                    print("start update policy", self.num_updates)
            
        elif self.in_batch_alter:
            update_vals = self._execute_model(feed_dict, self.model_update_dict)
            update_vals.update(self._execute_model(feed_dict, self.ppo_update_dict))
        elif self.use_transfer and self.smart_transfer:
            if self.update_mode == "model":
                update_vals = self._execute_model(feed_dict, self.update_dict)
                cur_loss = update_vals["model_loss"]
                print("model loss:", cur_loss)
                if abs(cur_loss - self.old_loss) < self.conv_thres:
                    self.update_mode = "policy"
                    print("start to train policy")
                else:
                    self.old_loss = cur_loss
            if self.update_mode == "policy":
                update_vals = self._execute_model(feed_dict, self.ppo_update_dict)
        else:
            update_vals = self._execute_model(feed_dict, self.update_dict)

        # update target encoder
        if not self.reuse_encoder and self.num_updates % self.copy_every == 0:
            self.policy.run_hard_copy()
            # print("copy")
            # self.policy.get_encoder_weights()

        for stat_name, update_name in stats_needed.items():
            update_stats[stat_name] = update_vals[update_name]

        self.num_updates += 1
        return update_stats

    def update_part(self, batch: AgentBuffer, num_sequences: int, update_type: str="policy") -> Dict[str, float]:
        """
        Performs update on model.
        :param mini_batch: Batch of experiences.
        :param num_sequences: Number of sequences to process.
        :return: Results of update.
        """
        feed_dict = self._construct_feed_dict(batch, num_sequences)
        if update_type == "model":
            stats_needed = {
                "Losses/Model Loss": "model_loss",
                "Policy/Learning Rate": "learning_rate",
                "Policy/Epsilon": "decay_epsilon",
                "Policy/Beta": "decay_beta",
            }
        elif update_type == "policy":
            stats_needed = {
                "Losses/Value Loss": "value_loss",
                "Losses/Policy Loss": "policy_loss",
                "Policy/Learning Rate": "learning_rate",
                "Policy/Epsilon": "decay_epsilon",
                "Policy/Beta": "decay_beta",
            }
        update_stats = {}
        # Collect feed dicts for all reward signals.
        for _, reward_signal in self.reward_signals.items():
            feed_dict.update(
                reward_signal.prepare_update(self.policy, batch, num_sequences)
            )
            stats_needed.update(reward_signal.stats_name_to_update_name)
        
        if update_type == "model":
            update_vals = self._execute_model(feed_dict, self.model_update_dict)
        elif update_type == "policy":
            update_vals = self._execute_model(feed_dict, self.ppo_update_dict)

        # update target encoder
        if not self.reuse_encoder and self.num_updates % self.copy_every == 0:
            self.policy.run_hard_copy()

        for stat_name, update_name in stats_needed.items():
            update_stats[stat_name] = update_vals[update_name]

        self.num_updates += 1
        return update_stats

    def _construct_feed_dict(
        self, mini_batch: AgentBuffer, num_sequences: int
    ) -> Dict[tf.Tensor, Any]:
        # Do an optional burn-in for memories
        num_burn_in = int(self.burn_in_ratio * self.policy.sequence_length)
        burn_in_mask = np.ones((self.policy.sequence_length), dtype=np.float32)
        burn_in_mask[range(0, num_burn_in)] = 0
        burn_in_mask = np.tile(burn_in_mask, num_sequences)
        feed_dict = {
            self.policy.batch_size_ph: num_sequences,
            self.policy.sequence_length_ph: self.policy.sequence_length,
            self.policy.mask_input: mini_batch["masks"] * burn_in_mask,
            self.advantage: mini_batch["advantages"],
            self.all_old_log_probs: mini_batch["action_probs"],
            self.policy.processed_vector_next: mini_batch["next_vector_in"],
            # self.policy.next_vector_in: mini_batch["next_vector_in"],
            self.policy.current_action: mini_batch["actions"],
            self.dis_returns: mini_batch["discounted_returns"]
        }
        for name in self.reward_signals:
            feed_dict[self.returns_holders[name]] = mini_batch[
                "{}_returns".format(name)
            ]
            feed_dict[self.old_values[name]] = mini_batch[
                "{}_value_estimates".format(name)
            ]

        if self.policy.output_pre is not None and "actions_pre" in mini_batch:
            feed_dict[self.policy.output_pre] = mini_batch["actions_pre"]
        else:
            feed_dict[self.policy.output] = mini_batch["actions"]
            if self.policy.use_recurrent:
                feed_dict[self.policy.prev_action] = mini_batch["prev_action"]
            feed_dict[self.policy.action_masks] = mini_batch["action_mask"]
        if "vector_obs" in mini_batch:
            feed_dict[self.policy.vector_in] = mini_batch["vector_obs"]
        if self.policy.vis_obs_size > 0:
            for i, _ in enumerate(self.policy.visual_in):
                feed_dict[self.policy.visual_in[i]] = mini_batch["visual_obs%d" % i]
        if self.policy.use_recurrent:
            feed_dict[self.policy.memory_in] = [
                mini_batch["memory"][i]
                for i in range(
                    0, len(mini_batch["memory"]), self.policy.sequence_length
                )
            ]
            feed_dict[self.memory_in] = self._make_zero_mem(
                self.m_size, mini_batch.num_experiences
            )
        return feed_dict

    def _create_cc_critic_old(
        self, h_size: int, num_layers: int, vis_encode_type: EncoderType
    ) -> None:
        """
        Creates Continuous control critic (value) network.
        :param h_size: Size of hidden linear layers.
        :param num_layers: Number of hidden linear layers.
        :param vis_encode_type: The type of visual encoder to use.
        """
        hidden_stream = ModelUtils.create_observation_streams(
            self.policy.visual_in,
            self.policy.processed_vector_in,
            1,
            h_size,
            num_layers,
            vis_encode_type,
        )[0]

        if self.policy.use_recurrent:
            hidden_value, memory_value_out = ModelUtils.create_recurrent_encoder(
                hidden_stream,
                self.memory_in,
                self.policy.sequence_length_ph,
                name="lstm_value",
            )
            self.memory_out = memory_value_out
        else:
            hidden_value = hidden_stream

        self.value_heads, self.value = ModelUtils.create_value_heads(
            self.stream_names, hidden_value
        )
        self.all_old_log_probs = tf.placeholder(
            shape=[None, sum(self.policy.act_size)],
            dtype=tf.float32,
            name="old_probabilities",
        )

        self.old_log_probs = tf.reduce_sum(
            (tf.identity(self.all_old_log_probs)), axis=1, keepdims=True
        )

    def _create_dc_critic_old(
        self, h_size: int, num_layers: int, vis_encode_type: EncoderType
    ) -> None:
        """
        Creates Discrete control critic (value) network.
        :param h_size: Size of hidden linear layers.
        :param num_layers: Number of hidden linear layers.
        :param vis_encode_type: The type of visual encoder to use.
        """
        hidden_stream = ModelUtils.create_observation_streams(
            self.policy.visual_in,
            self.policy.processed_vector_in,
            1,
            h_size,
            num_layers,
            vis_encode_type,
        )[0]

        if self.policy.use_recurrent:
            hidden_value, memory_value_out = ModelUtils.create_recurrent_encoder(
                hidden_stream,
                self.memory_in,
                self.policy.sequence_length_ph,
                name="lstm_value",
            )
            self.memory_out = memory_value_out
        else:
            hidden_value = hidden_stream

        self.value_heads, self.value = ModelUtils.create_value_heads(
            self.stream_names, hidden_value
        )

        self.all_old_log_probs = tf.placeholder(
            shape=[None, sum(self.policy.act_size)],
            dtype=tf.float32,
            name="old_probabilities",
        )

        # Break old log probs into separate branches
        old_log_prob_branches = ModelUtils.break_into_branches(
            self.all_old_log_probs, self.policy.act_size
        )

        _, _, old_normalized_logits = ModelUtils.create_discrete_action_masking_layer(
            old_log_prob_branches, self.policy.action_masks, self.policy.act_size
        )

        action_idx = [0] + list(np.cumsum(self.policy.act_size))

        self.old_log_probs = tf.reduce_sum(
            (
                tf.stack(
                    [
                        -tf.nn.softmax_cross_entropy_with_logits_v2(
                            labels=self.policy.selected_actions[
                                :, action_idx[i] : action_idx[i + 1]
                            ],
                            logits=old_normalized_logits[
                                :, action_idx[i] : action_idx[i + 1]
                            ],
                        )
                        for i in range(len(self.policy.act_size))
                    ],
                    axis=1,
                )
            ),
            axis=1,
            keepdims=True,
        )

    
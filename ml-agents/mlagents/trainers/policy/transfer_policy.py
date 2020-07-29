import os
import numpy as np
from typing import Any, Dict, Optional, List, Tuple
from mlagents.tf_utils import tf
from mlagents_envs.timers import timed
from mlagents_envs.base_env import DecisionSteps
from mlagents.trainers.brain import BrainParameters
from mlagents.trainers.models import EncoderType, NormalizerTensors
from mlagents.trainers.models import ModelUtils
from mlagents.trainers.policy.tf_policy import TFPolicy
from mlagents.trainers.settings import TrainerSettings
from mlagents.trainers.distributions import (
    GaussianDistribution,
    MultiCategoricalDistribution,
)

# import tf_slim as slim
EPSILON = 1e-6  # Small value to avoid divide by zero


class GaussianEncoderDistribution:
    def __init__(self, encoded: tf.Tensor, feature_size: int, reuse: bool = False):
        self.mu = tf.layers.dense(
            encoded,
            feature_size,
            activation=None,
            name="mu",
            kernel_initializer=ModelUtils.scaled_init(0.01),
            reuse=reuse,
        )

        self.log_sigma = tf.layers.dense(
            encoded,
            feature_size,
            activation=None,
            name="log_std",
            kernel_initializer=ModelUtils.scaled_init(0.01),
            reuse=reuse,
        )

        self.sigma = tf.exp(self.log_sigma)

    def sample(self):
        epsilon = tf.random_normal(tf.shape(self.mu))
        sampled = self.mu + self.sigma * epsilon

        return sampled

    def kl_standard(self):
        """
        KL divergence with a standard gaussian
        """
        kl = 0.5 * tf.reduce_sum(
            tf.square(self.mu) + tf.square(self.sigma) - 2 * self.log_sigma - 1, 1
        )
        return kl

    def w_distance(self, another):
        return tf.sqrt(
            tf.reduce_sum(tf.squared_difference(self.mu, another.mu), axis=1)
            + tf.reduce_sum(tf.squared_difference(self.sigma, another.sigma), axis=1)
        )


class TransferPolicy(TFPolicy):
    def __init__(
        self,
        seed: int,
        brain: BrainParameters,
        trainer_params: TrainerSettings,
        is_training: bool,
        model_path: str,
        load: bool,
        tanh_squash: bool = False,
        reparameterize: bool = False,
        condition_sigma_on_obs: bool = True,
        create_tf_graph: bool = True,
    ):
        """
        Policy that uses a multilayer perceptron to map the observations to actions. Could
        also use a CNN to encode visual input prior to the MLP. Supports discrete and
        continuous action spaces, as well as recurrent networks.
        :param seed: Random seed.
        :param brain: Assigned BrainParameters object.
        :param trainer_params: Defined training parameters.
        :param is_training: Whether the model should be trained.
        :param load: Whether a pre-trained model will be loaded or a new one created.
        :param model_path: Path where the model should be saved and loaded.
        :param tanh_squash: Whether to use a tanh function on the continuous output, or a clipped output.
        :param reparameterize: Whether we are using the resampling trick to update the policy in continuous output.
        """
        super().__init__(seed, brain, trainer_params, model_path, load)
        self.grads = None
        self.update_batch: Optional[tf.Operation] = None
        num_layers = self.network_settings.num_layers
        self.h_size = self.network_settings.hidden_units
        if num_layers < 1:
            num_layers = 1
        self.num_layers = num_layers
        self.vis_encode_type = self.network_settings.vis_encode_type
        self.tanh_squash = tanh_squash
        self.reparameterize = reparameterize
        self.condition_sigma_on_obs = condition_sigma_on_obs
        self.trainable_variables: List[tf.Variable] = []
        self.next_visual_in: List[tf.Tensor] = []
        self.encoder = None
        self.encoder_distribution = None
        self.targ_encoder = None

        # Non-exposed parameters; these aren't exposed because they don't have a
        # good explanation and usually shouldn't be touched.
        self.log_std_min = -20
        self.log_std_max = 2
        if create_tf_graph:
            self.create_tf_graph()

    def get_trainable_variables(self, 
        train_encoder: bool=True,
        train_action: bool=True,
        train_model: bool=True,
        train_policy: bool=True) -> List[tf.Variable]:
        """
        Returns a List of the trainable variables in this policy. if create_tf_graph hasn't been called,
        returns empty list.
        """
        trainable_variables = []
        if train_encoder:
            trainable_variables += self.encoding_variables
        if train_action:
            trainable_variables += self.action_variables
        if train_model:
            trainable_variables += self.model_variables
        if train_policy:
            trainable_variables += self.policy_variables
        return trainable_variables

    def create_tf_graph(
        self,
        encoder_layers=1,
        action_layers=1,
        policy_layers=1,
        forward_layers=1,
        inverse_layers=1,
        feature_size=16,
        action_feature_size=16,
        transfer=False,
        separate_train=False,
        separate_model_train=False,
        var_encoder=False,
        var_predict=True,
        predict_return=True,
        inverse_model=False,
        reuse_encoder=True,
        use_bisim=True,
        tau=0.1,
    ) -> None:
        """
        Builds the tensorflow graph needed for this policy.
        """
        self.inverse_model = inverse_model
        self.reuse_encoder = reuse_encoder
        self.feature_size = feature_size
        self.action_feature_size = action_feature_size
        self.predict_return = predict_return
        self.use_bisim = use_bisim
        self.transfer = transfer
        self.tau = tau

        with self.graph.as_default():
            tf.set_random_seed(self.seed)
            _vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            if len(_vars) > 0:
                # We assume the first thing created in the graph is the Policy. If
                # already populated, don't create more tensors.
                return
            self.create_input_placeholders()
            self.create_next_inputs()
            self.current_action = tf.placeholder(
                shape=[None, sum(self.act_size)],
                dtype=tf.float32,
                name="current_action",
            )
            self.current_reward = tf.placeholder(
                shape=[None], dtype=tf.float32, name="current_reward"
            )

            self.encoder = self._create_encoder_general(
                self.visual_in,
                self.processed_vector_in,
                self.h_size,
                self.feature_size,
                encoder_layers,
                self.vis_encode_type,
                scope="encoding"
            )

            self.next_encoder = self._create_encoder_general(
                self.visual_next,
                self.processed_vector_next,
                self.h_size,
                self.feature_size,
                encoder_layers,
                self.vis_encode_type,
                scope="encoding",
                reuse=True
            )

            self.targ_encoder = self._create_encoder_general(
                self.visual_in,
                self.processed_vector_in,
                self.h_size,
                self.feature_size,
                encoder_layers,
                self.vis_encode_type,
                scope="target_enc",
                stop_gradient=True,
            )

            self.next_targ_encoder = self._create_encoder_general(
                self.visual_next,
                self.processed_vector_next,
                self.h_size,
                self.feature_size,
                encoder_layers,
                self.vis_encode_type,
                scope="target_enc",
                reuse=True,
                stop_gradient=True,
            )

            self._create_hard_copy()
            self._create_soft_copy()


            # self.encoder = self._create_encoder(
            #     self.visual_in,
            #     self.processed_vector_in,
            #     self.h_size,
            #     self.feature_size,
            #     encoder_layers,
            #     self.vis_encode_type,
            # )

            # self.targ_encoder = self._create_target_encoder(
            #     self.h_size,
            #     self.feature_size,
            #     encoder_layers,
            #     self.vis_encode_type,
            #     reuse_encoder,
            # )

            self.action_encoder = self._create_action_encoder(
                self.current_action,
                self.h_size,
                self.action_feature_size,
                action_layers
            )

            if self.inverse_model:
                with tf.variable_scope("inverse"):
                    self.create_inverse_model(
                        self.encoder, self.targ_encoder, inverse_layers
                    )

            with tf.variable_scope("predict"):

                self.predict, self.predict_distribution = self.create_forward_model(
                    self.encoder,
                    self.action_encoder,
                    forward_layers,
                    var_predict=var_predict,
                    separate_train=separate_model_train
                )

                self.targ_predict, self.targ_predict_distribution = self.create_forward_model(
                    self.targ_encoder,
                    self.action_encoder,
                    forward_layers,
                    var_predict=var_predict,
                    reuse=True,
                    separate_train=separate_model_train
                )

                self.create_forward_loss(self.reuse_encoder, self.transfer)

            if predict_return:
                with tf.variable_scope("reward"):
                    self.create_reward_model(
                        self.encoder, self.action_encoder, forward_layers, separate_train=separate_model_train
                    )

            if self.use_bisim:
                self.create_bisim_model(
                    self.h_size,
                    self.feature_size,
                    encoder_layers,
                    action_layers,
                    self.vis_encode_type,
                    forward_layers,
                    var_predict,
                    predict_return,
                )

            if self.use_continuous_act:
                self._create_cc_actor(
                    self.encoder,
                    self.h_size,
                    policy_layers,
                    self.tanh_squash,
                    self.reparameterize,
                    self.condition_sigma_on_obs,
                    separate_train,
                )
            else:
                self._create_dc_actor(
                    self.encoder, self.h_size, policy_layers, separate_train
                )

            self.policy_variables = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy"
            )
            self.encoding_variables = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="encoding"
            )
            self.action_variables = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="action_enc"
            )
            self.model_variables = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="predict"
            ) + tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="reward"
            )

            self.encoding_variables += tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="lstm"
            )  # LSTMs need to be root scope for Barracuda export
            if self.inverse_model:
                self.model_variables += tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope="inverse"
                )

        self.inference_dict: Dict[str, tf.Tensor] = {
            "action": self.output,
            "log_probs": self.all_log_probs,
            "entropy": self.entropy,
        }
        if self.use_continuous_act:
            self.inference_dict["pre_action"] = self.output_pre
        if self.use_recurrent:
            self.inference_dict["memory_out"] = self.memory_out

        # We do an initialize to make the Policy usable out of the box. If an optimizer is needed,
        # it will re-load the full graph
        self._initialize_graph()

        # slim.model_analyzer.analyze_vars(self.trainable_variables, print_info=True)

    def load_graph_partial(
        self,
        path: str,
        load_model=False,
        load_policy=False,
        load_value=False,
        load_encoder=False,
        load_action=False
    ):
        load_nets = []
        if load_model:
            load_nets.append("predict")
            if self.predict_return:
                load_nets.append("reward")
        if load_policy:
            load_nets.append("policy")
        if load_value:
            load_nets.append("value")
        if load_encoder:
            load_nets.append("encoding")
        if load_action:
            load_nets.append("action_enc")
        # if self.inverse_model:
        #     load_nets.append("inverse")

        with self.graph.as_default():
            for net in load_nets:
                variables_to_restore = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, net
                )
                if net == "value" and len(variables_to_restore) == 0:
                    variables_to_restore = tf.get_collection(
                        tf.GraphKeys.TRAINABLE_VARIABLES, "critic"
                    )
                    net = "critic"
                partial_saver = tf.train.Saver(variables_to_restore)
                partial_model_checkpoint = os.path.join(path, f"{net}.ckpt")
                partial_saver.restore(self.sess, partial_model_checkpoint)
                print("loaded net", net, "from path", path)

        # if load_encoder:
        #     self.run_hard_copy()

    def _create_world_model(
        self,
        encoder: tf.Tensor,
        h_size: int,
        feature_size: int,
        num_layers: int,
        vis_encode_type: EncoderType,
        predict_return: bool = False,
    ) -> tf.Tensor:
        """"
        Builds the world model for state prediction
        """
        with self.graph.as_default():
            with tf.variable_scope("predict"):
                # self.current_action = tf.placeholder(
                #     shape=[None, sum(self.act_size)], dtype=tf.float32, name="current_action"
                # )
                hidden_stream = ModelUtils.create_vector_observation_encoder(
                    tf.concat([encoder, self.current_action], axis=1),
                    h_size,
                    ModelUtils.swish,
                    num_layers,
                    scope=f"main_graph",
                    reuse=False,
                )
                if predict_return:
                    predict = tf.layers.dense(
                        hidden_stream, feature_size + 1, name="next_state"
                    )
                else:
                    predict = tf.layers.dense(
                        hidden_stream, feature_size, name="next_state"
                    )
        return predict

    @timed
    def evaluate(
        self, decision_requests: DecisionSteps, global_agent_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluates policy for the agent experiences provided.
        :param decision_requests: DecisionSteps object containing inputs.
        :param global_agent_ids: The global (with worker ID) agent ids of the data in the batched_step_result.
        :return: Outputs from network as defined by self.inference_dict.
        """
        feed_dict = {
            self.batch_size_ph: len(decision_requests),
            self.sequence_length_ph: 1,
        }
        if self.use_recurrent:
            if not self.use_continuous_act:
                feed_dict[self.prev_action] = self.retrieve_previous_action(
                    global_agent_ids
                )
            feed_dict[self.memory_in] = self.retrieve_memories(global_agent_ids)
        feed_dict = self.fill_eval_dict(feed_dict, decision_requests)
        run_out = self._execute_model(feed_dict, self.inference_dict)
        return run_out

    # def _create_target_encoder(
    #     self,
    #     h_size: int,
    #     feature_size: int,
    #     num_layers: int,
    #     vis_encode_type: EncoderType,
    #     reuse_encoder: bool,
    # ) -> tf.Tensor:
    #     if reuse_encoder:
    #         next_encoder_scope = "encoding"
    #     else:
    #         next_encoder_scope = "target_enc"

    #     self.visual_next = ModelUtils.create_visual_input_placeholders(
    #         self.brain.camera_resolutions
    #     )
    #     self.vector_next = ModelUtils.create_vector_input(self.vec_obs_size)
    #     if self.normalize:
    #         vn_normalization_tensors = self.create_target_normalizer(self.vector_next)
    #         self.vn_update_normalization_op = vn_normalization_tensors.update_op
    #         self.vn_normalization_steps = vn_normalization_tensors.steps
    #         self.vn_running_mean = vn_normalization_tensors.running_mean
    #         self.vn_running_variance = vn_normalization_tensors.running_variance
    #         self.processed_vector_next = ModelUtils.normalize_vector_obs(
    #             self.vector_next,
    #             self.vn_running_mean,
    #             self.vn_running_variance,
    #             self.vn_normalization_steps,
    #         )
    #     else:
    #         self.processed_vector_next = self.vector_next
    #         self.vp_update_normalization_op = None

    #     with tf.variable_scope(next_encoder_scope):
    #         hidden_stream_targ = ModelUtils.create_observation_streams(
    #             self.visual_next,
    #             self.processed_vector_next,
    #             1,
    #             h_size,
    #             num_layers,
    #             vis_encode_type,
    #             reuse=reuse_encoder,
    #         )[0]

    #         latent_targ = tf.layers.dense(
    #             hidden_stream_targ,
    #             feature_size,
    #             name="latent",
    #             reuse=reuse_encoder,
    #             activation=tf.tanh,  # ModelUtils.swish,
    #             kernel_initializer=tf.initializers.variance_scaling(1.0),
    #         )
    #     return latent_targ
        # return tf.stop_gradient(latent_targ)

    # def _create_encoder(
    #     self,
    #     visual_in: List[tf.Tensor],
    #     vector_in: tf.Tensor,
    #     h_size: int,
    #     feature_size: int,
    #     num_layers: int,
    #     vis_encode_type: EncoderType,
    # ) -> tf.Tensor:
    #     """
    #     Creates an encoder for visual and vector observations.
    #     :param h_size: Size of hidden linear layers.
    #     :param num_layers: Number of hidden linear layers.
    #     :param vis_encode_type: Type of visual encoder to use if visual input.
    #     :return: The hidden layer (tf.Tensor) after the encoder.
    #     """
    #     with tf.variable_scope("encoding"):
    #         hidden_stream = ModelUtils.create_observation_streams(
    #             visual_in, vector_in, 1, h_size, num_layers, vis_encode_type, 
    #         )[0]

    #         latent = tf.layers.dense(
    #             hidden_stream,
    #             feature_size,
    #             name="latent",
    #             activation=tf.tanh,  # ModelUtils.swish,
    #             kernel_initializer=tf.initializers.variance_scaling(1.0),
    #         )
    #     return latent
    
    def _create_encoder_general(
        self,
        visual_in: List[tf.Tensor],
        vector_in: tf.Tensor,
        h_size: int,
        feature_size: int,
        num_layers: int,
        vis_encode_type: EncoderType,
        scope: str,
        reuse: bool=False,
        stop_gradient: bool=False
    ) -> tf.Tensor:
        """
        Creates an encoder for visual and vector observations.
        :param h_size: Size of hidden linear layers.
        :param num_layers: Number of hidden linear layers.
        :param vis_encode_type: Type of visual encoder to use if visual input.
        :return: The hidden layer (tf.Tensor) after the encoder.
        """
        with tf.variable_scope(scope):
            hidden_stream = ModelUtils.create_observation_streams(
                visual_in, vector_in, 1, h_size, num_layers, vis_encode_type, reuse=reuse
            )[0]

            latent = tf.layers.dense(
                hidden_stream,
                feature_size,
                name="latent",
                activation=tf.tanh,  # ModelUtils.swish,
                kernel_initializer=tf.initializers.variance_scaling(1.0),
                reuse=reuse
            )
        if stop_gradient:
            latent = tf.stop_gradient(latent)
        return latent

    def _create_action_encoder(
        self,
        action: tf.Tensor,
        h_size: int,
        action_feature_size: int,
        num_layers: int,
        reuse: bool=False
    ) -> tf.Tensor:

        if num_layers < 0:
            return self.current_action
        
        hidden_stream = ModelUtils.create_vector_observation_encoder(
            action, 
            h_size, 
            ModelUtils.swish,
            num_layers, 
            scope="action_enc",
            reuse=reuse
        )

        with tf.variable_scope("action_enc"):
            latent = tf.layers.dense(
                hidden_stream,
                action_feature_size,
                name="latent",
                activation=tf.tanh,  
                kernel_initializer=tf.initializers.variance_scaling(1.0),
                reuse=reuse
            )
        return latent

    def _create_hard_copy(self):
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_enc")
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="encoding")

        with tf.variable_scope("hard_replacement"):
            self.target_hardcp_op = [
                tf.assign(t, e) for t, e in zip(t_params, e_params)
            ]

    def _create_soft_copy(self):
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_enc")
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="encoding")

        with tf.variable_scope("soft_replacement"):
            self.target_softcp_op = [
                tf.assign(t, (1-self.tau) * t + self.tau * e) for t, e in zip(t_params, e_params)
            ]

    def run_hard_copy(self):
        self.sess.run(self.target_hardcp_op)
    
    def run_soft_copy(self):
        self.sess.run(self.target_softcp_op)

    def _create_cc_actor(
        self,
        encoded: tf.Tensor,
        h_size: int,
        num_layers: int,
        tanh_squash: bool = False,
        reparameterize: bool = False,
        condition_sigma_on_obs: bool = True,
        separate_train: bool = False,
    ) -> None:
        """
        Creates Continuous control actor-critic model.
        :param h_size: Size of hidden linear layers.
        :param num_layers: Number of hidden linear layers.
        :param vis_encode_type: Type of visual encoder to use if visual input.
        :param tanh_squash: Whether to use a tanh function, or a clipped output.
        :param reparameterize: Whether we are using the resampling trick to update the policy.
        """
        if self.use_recurrent:
            self.memory_in = tf.placeholder(
                shape=[None, self.m_size], dtype=tf.float32, name="recurrent_in"
            )
            hidden_policy, memory_policy_out = ModelUtils.create_recurrent_encoder(
                encoded, self.memory_in, self.sequence_length_ph, name="lstm_policy"
            )

            self.memory_out = tf.identity(memory_policy_out, name="recurrent_out")
        else:
            hidden_policy = encoded

        if separate_train:
            hidden_policy = tf.stop_gradient(hidden_policy)

        with tf.variable_scope("policy"):
            hidden_policy = ModelUtils.create_vector_observation_encoder(
                hidden_policy,
                h_size,
                ModelUtils.swish,
                num_layers,
                scope=f"main_graph",
                reuse=False,
            )
            distribution = GaussianDistribution(
                hidden_policy,
                self.act_size,
                reparameterize=reparameterize,
                tanh_squash=tanh_squash,
                condition_sigma=condition_sigma_on_obs,
            )

        if tanh_squash:
            self.output_pre = distribution.sample
            self.output = tf.identity(self.output_pre, name="action")
        else:
            self.output_pre = distribution.sample
            # Clip and scale output to ensure actions are always within [-1, 1] range.
            output_post = tf.clip_by_value(self.output_pre, -3, 3) / 3
            self.output = tf.identity(output_post, name="action")

        self.selected_actions = tf.stop_gradient(self.output)

        self.all_log_probs = tf.identity(distribution.log_probs, name="action_probs")
        self.entropy = distribution.entropy

        # We keep these tensors the same name, but use new nodes to keep code parallelism with discrete control.
        self.total_log_probs = distribution.total_log_probs

    def _create_dc_actor(
        self,
        encoded: tf.Tensor,
        h_size: int,
        num_layers: int,
        separate_train: bool = False,
    ) -> None:
        """
        Creates Discrete control actor-critic model.
        :param h_size: Size of hidden linear layers.
        :param num_layers: Number of hidden linear layers.
        :param vis_encode_type: Type of visual encoder to use if visual input.
        """
        if self.use_recurrent:
            self.prev_action = tf.placeholder(
                shape=[None, len(self.act_size)], dtype=tf.int32, name="prev_action"
            )
            prev_action_oh = tf.concat(
                [
                    tf.one_hot(self.prev_action[:, i], self.act_size[i])
                    for i in range(len(self.act_size))
                ],
                axis=1,
            )
            hidden_policy = tf.concat([encoded, prev_action_oh], axis=1)

            self.memory_in = tf.placeholder(
                shape=[None, self.m_size], dtype=tf.float32, name="recurrent_in"
            )
            hidden_policy, memory_policy_out = ModelUtils.create_recurrent_encoder(
                hidden_policy,
                self.memory_in,
                self.sequence_length_ph,
                name="lstm_policy",
            )

            self.memory_out = tf.identity(memory_policy_out, "recurrent_out")
        else:
            hidden_policy = encoded

        if separate_train:
            hidden_policy = tf.stop_gradient(hidden_policy)
        self.action_masks = tf.placeholder(
            shape=[None, sum(self.act_size)], dtype=tf.float32, name="action_masks"
        )

        with tf.variable_scope("policy"):
            hidden_policy = ModelUtils.create_vector_observation_encoder(
                hidden_policy,
                h_size,
                ModelUtils.swish,
                num_layers,
                scope=f"main_graph",
                reuse=False,
            )
            distribution = MultiCategoricalDistribution(
                hidden_policy, self.act_size, self.action_masks
            )
        # It's important that we are able to feed_dict a value into this tensor to get the
        # right one-hot encoding, so we can't do identity on it.
        self.output = distribution.sample
        self.all_log_probs = tf.identity(distribution.log_probs, name="action")
        self.selected_actions = tf.stop_gradient(
            distribution.sample_onehot
        )  # In discrete, these are onehot
        self.entropy = distribution.entropy
        self.total_log_probs = distribution.total_log_probs

    def save_model(self, steps):
        """
        Saves the model
        :param steps: The number of steps the model was trained for
        :return:
        """
        # self.get_policy_weights()
        with self.graph.as_default():
            last_checkpoint = os.path.join(self.model_path, f"model-{steps}.ckpt")
            self.saver.save(self.sess, last_checkpoint)
            tf.train.write_graph(
                self.graph, self.model_path, "raw_graph_def.pb", as_text=False
            )
            # save each net separately
            policy_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "policy")
            policy_saver = tf.train.Saver(policy_vars)
            policy_checkpoint = os.path.join(self.model_path, f"policy.ckpt")
            policy_saver.save(self.sess, policy_checkpoint)

            encoding_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, "encoding"
            )
            encoding_saver = tf.train.Saver(encoding_vars)
            encoding_checkpoint = os.path.join(self.model_path, f"encoding.ckpt")
            encoding_saver.save(self.sess, encoding_checkpoint)

            action_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, "action_enc"
            )
            if len(action_vars) > 0:
                action_saver = tf.train.Saver(action_vars)
                action_checkpoint = os.path.join(self.model_path, f"action_enc.ckpt")
                action_saver.save(self.sess, action_checkpoint)

            latent_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, "encoding/latent"
            )
            latent_saver = tf.train.Saver(latent_vars)
            latent_checkpoint = os.path.join(self.model_path, f"latent.ckpt")
            latent_saver.save(self.sess, latent_checkpoint)

            predict_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, "predict"
            )
            predict_saver = tf.train.Saver(predict_vars)
            predict_checkpoint = os.path.join(self.model_path, f"predict.ckpt")
            predict_saver.save(self.sess, predict_checkpoint)

            value_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "value")
            if len(value_vars) > 0:
                value_saver = tf.train.Saver(value_vars)
                value_checkpoint = os.path.join(self.model_path, f"value.ckpt")
                value_saver.save(self.sess, value_checkpoint)
            
            critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "critic")
            if len(critic_vars) > 0:
                critic_saver = tf.train.Saver(critic_vars)
                critic_checkpoint = os.path.join(self.model_path, f"critic.ckpt")
                critic_saver.save(self.sess, critic_checkpoint)

            if self.inverse_model:
                inverse_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, "inverse"
                )
                inverse_saver = tf.train.Saver(inverse_vars)
                inverse_checkpoint = os.path.join(self.model_path, f"inverse.ckpt")
                inverse_saver.save(self.sess, inverse_checkpoint)

            if self.predict_return:
                reward_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, "reward"
                )
                reward_saver = tf.train.Saver(reward_vars)
                reward_checkpoint = os.path.join(self.model_path, f"reward.ckpt")
                reward_saver.save(self.sess, reward_checkpoint)

    def create_target_normalizer(
        self, vector_obs: tf.Tensor, prefix="vn"
    ) -> NormalizerTensors:
        vec_obs_size = vector_obs.shape[1]
        steps = tf.get_variable(
            prefix + "_normalization_steps",
            [],
            trainable=False,
            dtype=tf.int32,
            initializer=tf.zeros_initializer(),
        )
        running_mean = tf.get_variable(
            prefix + "_running_mean",
            [vec_obs_size],
            trainable=False,
            dtype=tf.float32,
            initializer=tf.zeros_initializer(),
        )
        running_variance = tf.get_variable(
            prefix + "_running_variance",
            [vec_obs_size],
            trainable=False,
            dtype=tf.float32,
            initializer=tf.ones_initializer(),
        )
        update_normalization = ModelUtils.create_normalizer_update(
            vector_obs, steps, running_mean, running_variance
        )
        return NormalizerTensors(
            update_normalization, steps, running_mean, running_variance
        )

    def update_normalization(
        self,
        vector_obs: np.ndarray,
        vector_obs_next: np.ndarray,
        vector_obs_bisim: np.ndarray,
    ) -> None:
        """
        If this policy normalizes vector observations, this will update the norm values in the graph.
        :param vector_obs: The vector observations to add to the running estimate of the distribution.
        """
        if self.use_vec_obs and self.normalize:
            self.sess.run(
                self.update_normalization_op, feed_dict={self.vector_in: vector_obs}
            )
            self.sess.run(
                self.vn_update_normalization_op,
                feed_dict={self.vector_next: vector_obs_next},
            )
            if self.use_bisim:
                self.sess.run(
                    self.bi_update_normalization_op,
                    feed_dict={self.vector_bisim: vector_obs_bisim},
                )

    def get_encoder_weights(self):
        with self.graph.as_default():
            enc = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, "encoding/latent/bias:0"
            )
            targ = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, "target_enc/latent/bias:0"
            )
            print("encoding:", self.sess.run(enc))
            print("target:", self.sess.run(targ))

    def get_policy_weights(self):
        with self.graph.as_default():
            #    pol = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "policy/mu/bias:0")
            #    print("policy:", self.sess.run(pol))
            enc = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "encoding")
            print("encoding:", self.sess.run(enc))
            pred = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "predict")
            print("predict:", self.sess.run(pred))

        #    rew = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "reward")
        #    print("reward:", self.sess.run(rew))

    def create_inverse_model(
        self,
        encoded_state: tf.Tensor,
        encoded_next_state: tf.Tensor,
        inverse_layers: int,
    ) -> None:
        """
        Creates inverse model TensorFlow ops for Curiosity module.
        Predicts action taken given current and future encoded states.
        :param encoded_state: Tensor corresponding to encoded current state.
        :param encoded_next_state: Tensor corresponding to encoded next state.
        """
        combined_input = tf.concat([encoded_state, encoded_next_state], axis=1)
        # hidden = tf.layers.dense(combined_input, 256, activation=ModelUtils.swish)
        hidden = combined_input
        for i in range(inverse_layers - 1):
            hidden = tf.layers.dense(
                hidden,
                self.h_size,
                activation=ModelUtils.swish,
                name="hidden_{}".format(i),
                kernel_initializer=tf.initializers.variance_scaling(1.0),
            )

        if self.brain.vector_action_space_type == "continuous":
            pred_action = tf.layers.dense(
                hidden, self.act_size[0], activation=None, name="pred_action"
            )
            squared_difference = tf.reduce_sum(
                tf.squared_difference(pred_action, self.current_action), axis=1
            )
            self.inverse_loss = tf.reduce_mean(
                tf.dynamic_partition(squared_difference, self.mask, 2)[1]
            )
        else:
            pred_action = tf.concat(
                [
                    tf.layers.dense(
                        hidden,
                        self.act_size[i],
                        activation=tf.nn.softmax,
                        name="pred_action",
                    )
                    for i in range(len(self.act_size))
                ],
                axis=1,
            )
            cross_entropy = tf.reduce_sum(
                -tf.log(pred_action + 1e-10) * self.current_action, axis=1
            )
            self.inverse_loss = tf.reduce_mean(
                tf.dynamic_partition(cross_entropy, self.mask, 2)[1]
            )

    def create_forward_model(
        self,
        encoded_state: tf.Tensor,
        encoded_action: tf.Tensor,
        forward_layers: int,
        var_predict: bool = False,
        reuse: bool = False,
        separate_train: bool = False
    ) -> None:
        """
        Creates forward model TensorFlow ops for Curiosity module.
        Predicts encoded future state based on encoded current state and given action.
        :param encoded_state: Tensor corresponding to encoded current state.
        :param encoded_next_state: Tensor corresponding to encoded next state.
        """
        combined_input = tf.concat([encoded_state, encoded_action], axis=1)
        hidden = combined_input

        if separate_train:
           hidden = tf.stop_gradient(hidden)

        for i in range(forward_layers):
            hidden = tf.layers.dense(
                hidden,
                self.h_size,
                name="hidden_{}".format(i),
                activation=ModelUtils.swish,
                # kernel_initializer=tf.initializers.variance_scaling(1.0),
                reuse=reuse
            )

        if var_predict:
            predict_distribution = GaussianEncoderDistribution(
                hidden, self.feature_size, reuse=reuse
            )
            predict = predict_distribution.sample()
        else:
            predict = tf.layers.dense(
                hidden,
                self.feature_size,
                name="latent",
                # activation=tf.tanh,
                # kernel_initializer=tf.initializers.variance_scaling(1.0),
                reuse=reuse
            )
            predict_distribution = None
        
        return predict, predict_distribution

        # if not self.transfer:
        #     encoded_next_state = tf.stop_gradient(encoded_next_state)
        # squared_difference = 0.5 * tf.reduce_sum(
        #     tf.squared_difference(tf.tanh(self.predict), encoded_next_state), axis=1
        # )

        # # self.forward_loss = tf.reduce_mean(squared_difference)
        # self.next_state = encoded_next_state
        # self.forward_loss = tf.reduce_mean(
        #     tf.dynamic_partition(squared_difference, self.mask, 2)[1]
        # )
    
    def create_forward_loss(self, reuse: bool, transfer: bool):

        if not transfer:
            if reuse:
                encoded_next_state = tf.stop_gradient(self.next_encoder)
            else:
                encoded_next_state = self.next_targ_encoder # gradient of target encode is already stopped

            squared_difference = 0.5 * tf.reduce_sum(
                tf.squared_difference(tf.tanh(self.predict), encoded_next_state), axis=1
            )
            self.forward_loss = tf.reduce_mean(
                tf.dynamic_partition(squared_difference, self.mask, 2)[1]
            )

        else:
            if reuse:
                squared_difference_1 = 0.5 * tf.reduce_sum(
                    tf.squared_difference(tf.tanh(self.predict), tf.stop_gradient(self.next_encoder)), 
                    axis=1
                )
                squared_difference_2 = 0.5 * tf.reduce_sum(
                    tf.squared_difference(tf.tanh(tf.stop_gradient(self.predict)), self.next_encoder), 
                    axis=1
                )
            else:
                squared_difference_1 = 0.5 * tf.reduce_sum(
                    tf.squared_difference(tf.tanh(self.predict), self.next_targ_encoder), 
                    axis=1
                )
                squared_difference_2 = 0.5 * tf.reduce_sum(
                    tf.squared_difference(tf.tanh(self.targ_predict), self.next_encoder), 
                    axis=1
                )
            self.forward_loss = tf.reduce_mean(
                tf.dynamic_partition(0.5 * squared_difference_1 + 0.5 * squared_difference_2, self.mask, 2)[1]
            )


    def create_reward_model(
        self,
        encoded_state: tf.Tensor,
        encoded_action: tf.Tensor,
        forward_layers: int,
        separate_train: bool = False
    ):

        combined_input = tf.concat([encoded_state, encoded_action], axis=1)

        hidden = combined_input
        if separate_train:
           hidden = tf.stop_gradient(hidden)
        for i in range(forward_layers):
            hidden = tf.layers.dense(
                hidden,
                self.h_size * (self.vis_obs_size + int(self.vec_obs_size > 0)),
                name="hidden_{}".format(i),
                activation=ModelUtils.swish,
                # kernel_initializer=tf.initializers.variance_scaling(1.0),
            )
        self.pred_reward = tf.layers.dense(
            hidden,
            1,
            name="reward",
            # activation=ModelUtils.swish,
            # kernel_initializer=tf.initializers.variance_scaling(1.0),
        )

        self.reward_loss = tf.reduce_mean(
            tf.squared_difference(self.pred_reward, self.current_reward)
        )
        # self.reward_loss = tf.clip_by_value(
        #    tf.reduce_mean(
        #        tf.squared_difference(self.pred_reward, self.current_reward)
        #    ),
        #    1e-10,
        #    1.0,
        # )

    def create_bisim_model(
        self,
        h_size: int,
        feature_size: int,
        encoder_layers: int,
        action_layers: int,
        vis_encode_type: EncoderType,
        forward_layers: int,
        var_predict: bool,
        predict_return: bool,
    ) -> None:
        with tf.variable_scope("encoding"):
            self.visual_bisim = ModelUtils.create_visual_input_placeholders(
                self.brain.camera_resolutions
            )
            self.vector_bisim = ModelUtils.create_vector_input(self.vec_obs_size)
            if self.normalize:
                bi_normalization_tensors = self.create_target_normalizer(
                    self.vector_bisim, prefix="bi"
                )
                self.bi_update_normalization_op = bi_normalization_tensors.update_op
                self.bi_normalization_steps = bi_normalization_tensors.steps
                self.bi_running_mean = bi_normalization_tensors.running_mean
                self.bi_running_variance = bi_normalization_tensors.running_variance
                self.processed_vector_bisim = ModelUtils.normalize_vector_obs(
                    self.vector_bisim,
                    self.bi_running_mean,
                    self.bi_running_variance,
                    self.bi_normalization_steps,
                )
            else:
                self.processed_vector_bisim = self.vector_bisim
                self.vp_update_normalization_op = None

            hidden_stream = ModelUtils.create_observation_streams(
                self.visual_bisim,
                self.processed_vector_bisim,
                1,
                h_size,
                encoder_layers,
                vis_encode_type,
                reuse=True,
            )[0]

            self.bisim_encoder = tf.layers.dense(
                hidden_stream,
                feature_size,
                name="latent",
                activation=ModelUtils.swish,
                kernel_initializer=tf.initializers.variance_scaling(1.0),
                reuse=True,
            )
        self.bisim_action = tf.placeholder(
            shape=[None, sum(self.act_size)], dtype=tf.float32, name="bisim_action"
        )
        # self.bisim_action_encoder = self._create_action_encoder(
        #     self.bisim_action,
        #     self.h_size,
        #     self.action_feature_size,
        #     action_layers,
        #     reuse=True,
        # )
        combined_input = tf.concat([self.bisim_encoder, self.bisim_action], axis=1)
        combined_input = tf.stop_gradient(combined_input)

        with tf.variable_scope("predict"):
            hidden = combined_input
            for i in range(forward_layers):
                hidden = tf.layers.dense(
                    hidden,
                    self.h_size,
                    name="hidden_{}".format(i),
                    reuse=True,
                    activation=ModelUtils.swish,
                    # kernel_initializer=tf.initializers.variance_scaling(1.0),
                )

            self.bisim_predict_distribution = GaussianEncoderDistribution(
                hidden, self.feature_size, reuse=True
            )
            self.bisim_predict = self.predict_distribution.sample()
        with tf.variable_scope("reward"):
            hidden = combined_input
            for i in range(forward_layers):
                hidden = tf.layers.dense(
                    hidden,
                    self.h_size * (self.vis_obs_size + int(self.vec_obs_size > 0)),
                    name="hidden_{}".format(i),
                    reuse=True,
                    activation=ModelUtils.swish,
                    # kernel_initializer=tf.initializers.variance_scaling(1.0),
                )
            self.bisim_pred_reward = tf.layers.dense(
                hidden,
                1,
                name="reward",
                reuse=True
                # activation=ModelUtils.swish,
                # kernel_initializer=tf.initializers.variance_scaling(1.0),
            )

    def create_next_inputs(self):
        self.visual_next = ModelUtils.create_visual_input_placeholders(
            self.brain.camera_resolutions
        )
        self.vector_next = ModelUtils.create_vector_input(self.vec_obs_size)
        if self.normalize:
            vn_normalization_tensors = self.create_target_normalizer(self.vector_next)
            self.vn_update_normalization_op = vn_normalization_tensors.update_op
            self.vn_normalization_steps = vn_normalization_tensors.steps
            self.vn_running_mean = vn_normalization_tensors.running_mean
            self.vn_running_variance = vn_normalization_tensors.running_variance
            self.processed_vector_next = ModelUtils.normalize_vector_obs(
                self.vector_next,
                self.vn_running_mean,
                self.vn_running_variance,
                self.vn_normalization_steps,
            )
        else:
            self.processed_vector_next = self.vector_next
            self.vp_update_normalization_op = None
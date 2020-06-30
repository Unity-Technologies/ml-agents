import os
from typing import Any, Dict, Optional, List, Tuple
from mlagents.tf_utils import tf
from mlagents_envs.timers import timed
from mlagents_envs.base_env import DecisionSteps
from mlagents.trainers.brain import BrainParameters
from mlagents.trainers.models import EncoderType
from mlagents.trainers.models import ModelUtils
from mlagents.trainers.policy.tf_policy import TFPolicy
from mlagents.trainers.settings import TrainerSettings
from mlagents.trainers.distributions import (
    GaussianDistribution,
    MultiCategoricalDistribution,
)
import tf_slim as slim
EPSILON = 1e-6  # Small value to avoid divide by zero

class GaussianEncoderDistribution:
    def __init__(
        self,
        encoded: tf.Tensor,
        feature_size: int,
        reuse: bool=False
    ):
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
            reuse=reuse
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
        kl = 0.5 * tf.reduce_sum(tf.square(self.mu) + tf.square(self.sigma) - 2 * self.log_sigma - 1, 1)
        return kl


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
        self.encoder = None
        self.encoder_distribution = None
        self.targ_encoder = None

        # Model-based learning
        self.feature_size = 16  # dimension of latent feature size
        
        # Non-exposed parameters; these aren't exposed because they don't have a
        # good explanation and usually shouldn't be touched.
        self.log_std_min = -20
        self.log_std_max = 2
        if create_tf_graph:
            self.create_tf_graph()

    def get_trainable_variables(self) -> List[tf.Variable]:
        """
        Returns a List of the trainable variables in this policy. if create_tf_graph hasn't been called,
        returns empty list.
        """
        return self.trainable_variables

    def create_tf_graph(self, 
        encoder_layers = 1,
        policy_layers = 1,
        transfer=False, 
        separate_train=False, 
        var_encoder=False,
        var_predict=False,
        predict_return=False,
        inverse_model=False,
        reuse_encoder=False,
    ) -> None:
        """
        Builds the tensorflow graph needed for this policy.
        """
        self.inverse_model = inverse_model
        self.reuse_encoder = transfer

        with self.graph.as_default():
            tf.set_random_seed(self.seed)
            _vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            if len(_vars) > 0:
                # We assume the first thing created in the graph is the Policy. If
                # already populated, don't create more tensors.
                return

            self.create_input_placeholders()
            self.current_action = tf.placeholder(
                    shape=[None, sum(self.act_size)], dtype=tf.float32, name="current_action"
                )

            self.next_visual_in: List[tf.Tensor] = []
            
            # if var_encoder:
            #     self.encoder, self.targ_encoder, self.encoder_distribution, _ = self.create_encoders(var_latent=True, reuse_encoder=reuse_encoder)
            # else:
            #     self.encoder, self.targ_encoder = self.create_encoders(reuse_encoder=reuse_encoder)
            
            # if not reuse_encoder:
            #     self.targ_encoder = tf.stop_gradient(self.targ_encoder)
            #     self._create_hard_copy()
            
            if var_encoder:
                self.encoder_distribution, self.encoder = self._create_var_encoder(
                    self.visual_in,
                    self.processed_vector_in,
                    self.h_size,
                    self.feature_size,
                    encoder_layers,
                    self.vis_encode_type
                )

                _, self.targ_encoder = self._create_var_target_encoder(
                    self.h_size,
                    self.feature_size,
                    encoder_layers,
                    self.vis_encode_type,
                    reuse_encoder
                )
            else:
                self.encoder = self._create_encoder(
                    self.visual_in,
                    self.processed_vector_in,
                    self.h_size,
                    self.feature_size,
                    encoder_layers,
                    self.vis_encode_type
                )

                self.targ_encoder = self._create_target_encoder(
                    self.h_size,
                    self.feature_size,
                    encoder_layers,
                    self.vis_encode_type,
                    reuse_encoder
                )

            if not reuse_encoder:
                self.targ_encoder = tf.stop_gradient(self.targ_encoder)
                self._create_hard_copy()

            with tf.variable_scope("inverse"):
                self.create_inverse_model(self.encoder, self.targ_encoder)
            with tf.variable_scope("predict"):
                self.create_forward_model(self.encoder, self.targ_encoder)

            # if var_predict:
            #     self.predict_distribution, self.predict = self._create_var_world_model(
            #         self.encoder,
            #         self.h_size,
            #         self.feature_size,
            #         self.num_layers,
            #         self.vis_encode_type,
            #         predict_return
            #     )
            # else:
            #     self.predict = self._create_world_model(
            #         self.encoder,
            #         self.h_size,
            #         self.feature_size,
            #         self.num_layers,
            #         self.vis_encode_type,
            #         predict_return
            #     )
            
            # if inverse_model:
            #     self._create_inverse_model(self.encoder, self.targ_encoder)

            if self.use_continuous_act:
                self._create_cc_actor(
                    self.encoder,
                    self.h_size,
                    policy_layers,
                    self.tanh_squash,
                    self.reparameterize,
                    self.condition_sigma_on_obs,
                    separate_train
                )
            else:
                self._create_dc_actor(self.encoder, self.h_size, policy_layers, separate_train)

            self.trainable_variables = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy"
            )
            self.trainable_variables += tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="encoding"
            )
            self.trainable_variables += tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="predict"
            )
            self.trainable_variables += tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="lstm"
            )  # LSTMs need to be root scope for Barracuda export
            if self.inverse_model:
                self.trainable_variables += tf.get_collection(
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
    
    def load_graph_partial(self, path: str, transfer_type="dynamics"):
        load_nets = {"dynamics": ["policy", "predict", "value"], 
            "observation": ["encoding", "inverse"]}
        if self.inverse_model:
            load_nets["dynamics"].append("inverse")
        with self.graph.as_default():
            for net in load_nets[transfer_type]:
                variables_to_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, net)
                partial_saver = tf.train.Saver(variables_to_restore)
                partial_model_checkpoint = os.path.join(path, f"{net}.ckpt")
                partial_saver.restore(self.sess, partial_model_checkpoint)
                print("loaded net", net, "from path", path)
        # variables_to_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "encoding/latent")
        # partial_saver = tf.train.Saver(variables_to_restore)
        # partial_model_checkpoint = os.path.join(path, f"latent.ckpt")
        # partial_saver.restore(self.sess, partial_model_checkpoint)
        # print("loaded net latent from path", path)
        
        if transfer_type == "observation":
            self.run_hard_copy()

    def _create_world_model(
        self,
        encoder: tf.Tensor,
        h_size: int,
        feature_size: int,
        num_layers: int,
        vis_encode_type: EncoderType,
        predict_return: bool=False
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
                    reuse=False
                )
                if predict_return:
                    predict = tf.layers.dense(
                        hidden_stream,
                        feature_size+1,
                        name="next_state"
                    )
                else:
                    predict = tf.layers.dense(
                        hidden_stream,
                        feature_size,
                        name="next_state"
                    )
        return predict

    def _create_var_world_model(
        self,
        encoder: tf.Tensor,
        h_size: int,
        feature_size: int,
        num_layers: int,
        vis_encode_type: EncoderType,
        predict_return: bool=False
    ) -> tf.Tensor:
        """"
        Builds the world model for state prediction
        """
        with self.graph.as_default():
            with tf.variable_scope("predict"):
                
                hidden_stream = ModelUtils.create_vector_observation_encoder(
                    tf.concat([encoder, self.current_action], axis=1),
                    h_size,
                    ModelUtils.swish,
                    num_layers,
                    scope=f"main_graph",
                    reuse=False
                )
                with tf.variable_scope("latent"):
                    if predict_return:
                        predict_distribution = GaussianEncoderDistribution(
                                hidden_stream,
                                feature_size+1
                            )
                        # separate prediction of return
                    else:
                        predict_distribution = GaussianEncoderDistribution(
                                hidden_stream,
                                feature_size
                            )

                    predict = predict_distribution.sample()
        return predict_distribution, predict

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

    def _create_target_encoder(
        self,
        h_size: int,
        feature_size: int,
        num_layers: int,
        vis_encode_type: EncoderType,
        reuse_encoder: bool
    ) -> tf.Tensor:
        if reuse_encoder:
            next_encoder_scope = "encoding"
        else:
            next_encoder_scope = "target_enc"

        self.visual_next = ModelUtils.create_visual_input_placeholders(
            self.brain.camera_resolutions
        )
        self.vector_next = ModelUtils.create_vector_input(self.vec_obs_size)
        if self.normalize:
            self.processed_vector_next = ModelUtils.normalize_vector_obs(
                self.vector_next,
                self.running_mean,
                self.running_variance,
                self.normalization_steps,
            )
        else:
            self.processed_vector_next = self.vector_next

        with tf.variable_scope(next_encoder_scope):
            hidden_stream_targ = ModelUtils.create_observation_streams(
                self.visual_next,
                self.processed_vector_next,
                1,
                h_size,
                num_layers,
                vis_encode_type,
                reuse=reuse_encoder
            )[0]

            latent_targ = tf.layers.dense(
                    hidden_stream_targ,
                    feature_size,
                    name="latent",
                    reuse=reuse_encoder
                )
        return latent_targ
        # return tf.stop_gradient(latent_targ)
    
    def _create_encoder(
        self,
        visual_in: List[tf.Tensor],
        vector_in: tf.Tensor,
        h_size: int,
        feature_size: int,
        num_layers: int,
        vis_encode_type: EncoderType,
    ) -> tf.Tensor:
        """
        Creates an encoder for visual and vector observations.
        :param h_size: Size of hidden linear layers.
        :param num_layers: Number of hidden linear layers.
        :param vis_encode_type: Type of visual encoder to use if visual input.
        :return: The hidden layer (tf.Tensor) after the encoder.
        """
        with tf.variable_scope("encoding"):
            hidden_stream = ModelUtils.create_observation_streams(
                visual_in,
                vector_in,
                1,
                h_size,
                num_layers,
                vis_encode_type,
            )[0]

            latent = tf.layers.dense(
                    hidden_stream,
                    feature_size,
                    name="latent"
                )
        return latent
    
    def _create_var_target_encoder(
        self,
        h_size: int,
        feature_size: int,
        num_layers: int,
        vis_encode_type: EncoderType,
        reuse_encoder: bool
    ) -> tf.Tensor:
        if reuse_encoder:
            next_encoder_scope = "encoding"
        else:
            next_encoder_scope = "target_enc"
        self.visual_next = ModelUtils.create_visual_input_placeholders(
            self.brain.camera_resolutions
        )
        self.vector_next = ModelUtils.create_vector_input(self.vec_obs_size)
        if self.normalize:
            self.processed_vector_next = ModelUtils.normalize_vector_obs(
                self.vector_next,
                self.running_mean,
                self.running_variance,
                self.normalization_steps,
            )
        else:
            self.processed_vector_next = self.vector_next

        with tf.variable_scope(next_encoder_scope):
            hidden_stream_targ = ModelUtils.create_observation_streams(
                self.visual_next,
                self.processed_vector_next,
                1,
                h_size,
                num_layers,
                vis_encode_type,
            )[0]

            with tf.variable_scope("latent"):
                latent_targ_distribution = GaussianEncoderDistribution(
                    hidden_stream_targ,
                    feature_size
                )

                latent_targ = latent_targ_distribution.sample()

        return latent_targ_distribution, latent_targ

    def _create_var_encoder(
        self,
        visual_in: List[tf.Tensor],
        vector_in: tf.Tensor,
        h_size: int,
        feature_size: int,
        num_layers: int,
        vis_encode_type: EncoderType
    ) -> tf.Tensor:
        """
        Creates a variational encoder for visual and vector observations.
        :param h_size: Size of hidden linear layers.
        :param num_layers: Number of hidden linear layers.
        :param vis_encode_type: Type of visual encoder to use if visual input.
        :return: The hidden layer (tf.Tensor) after the encoder.
        """

        with tf.variable_scope("encoding"):
            hidden_stream = ModelUtils.create_observation_streams(
                visual_in,
                vector_in,
                1,
                h_size,
                num_layers,
                vis_encode_type,
            )[0]

            with tf.variable_scope("latent"):
                latent_distribution = GaussianEncoderDistribution(
                    hidden_stream,
                    feature_size
                )

                latent = latent_distribution.sample()

        return latent_distribution, latent
    
    def _create_hard_copy(self):
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_enc')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoding')

        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def run_hard_copy(self):
        self.sess.run(self.target_replace_op)

    def _create_inverse_model(
        self, encoded_state: tf.Tensor, encoded_next_state: tf.Tensor
    ) -> None:
        """
        Creates inverse model TensorFlow ops for Curiosity module.
        Predicts action taken given current and future encoded states.
        :param encoded_state: Tensor corresponding to encoded current state.
        :param encoded_next_state: Tensor corresponding to encoded next state.
        """
        with tf.variable_scope("inverse"):
            combined_input = tf.concat([encoded_state, encoded_next_state], axis=1)
            hidden = tf.layers.dense(combined_input, self.h_size, activation=ModelUtils.swish)
            if self.brain.vector_action_space_type == "continuous":
                pred_action = tf.layers.dense(
                    hidden, self.act_size[0], activation=None
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
                            hidden, self.act_size[i], activation=tf.nn.softmax
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
    
    def _create_cc_actor(
        self,
        encoded: tf.Tensor,
        h_size: int,
        num_layers: int,
        tanh_squash: bool = False,
        reparameterize: bool = False,
        condition_sigma_on_obs: bool = True,
        separate_train: bool = False
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
        separate_train: bool = False
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
        self.get_policy_weights()
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

            encoding_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "encoding")
            encoding_saver = tf.train.Saver(encoding_vars)
            encoding_checkpoint = os.path.join(self.model_path, f"encoding.ckpt")
            encoding_saver.save(self.sess, encoding_checkpoint)

            latent_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "encoding/latent")
            latent_saver = tf.train.Saver(latent_vars)
            latent_checkpoint = os.path.join(self.model_path, f"latent.ckpt")
            latent_saver.save(self.sess, latent_checkpoint)

            predict_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "predict")
            predict_saver = tf.train.Saver(predict_vars)
            predict_checkpoint = os.path.join(self.model_path, f"predict.ckpt")
            predict_saver.save(self.sess, predict_checkpoint)

            value_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "value")
            value_saver = tf.train.Saver(value_vars)
            value_checkpoint = os.path.join(self.model_path, f"value.ckpt")
            value_saver.save(self.sess, value_checkpoint)

            if self.inverse_model:
                inverse_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "inverse")
                inverse_saver = tf.train.Saver(inverse_vars)
                inverse_checkpoint = os.path.join(self.model_path, f"inverse.ckpt")
                inverse_saver.save(self.sess, inverse_checkpoint)

    def get_encoder_weights(self):
        with self.graph.as_default():
            enc = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "encoding/latent/bias:0")
            targ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "target_enc/latent/bias:0")
            print("encoding:", self.sess.run(enc))
            print("target:", self.sess.run(targ))

    def get_policy_weights(self):
        with self.graph.as_default():
            pol = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "policy/mu/bias:0")
            print("policy:", self.sess.run(pol))
    
    def create_encoders(self, var_latent: bool=False, reuse_encoder: bool=False) -> Tuple[tf.Tensor, tf.Tensor]:
        encoded_state_list = []
        encoded_next_state_list = []
        if reuse_encoder:
            next_encoder_scope = "encoding"
        else:
            next_encoder_scope = "target_enc"
        if self.vis_obs_size > 0:
            self.next_visual_in = []
            visual_encoders = []
            next_visual_encoders = []
            for i in range(self.vis_obs_size):
                # Create input ops for next (t+1) visual observations.
                next_visual_input = ModelUtils.create_visual_input(
                    self.brain.camera_resolutions[i],
                    name="next_visual_observation_" + str(i),
                )
                self.next_visual_in.append(next_visual_input)

                # Create the encoder ops for current and next visual input.
                # Note that these encoders are siamese.
                with tf.variable_scope("encoding"):
                    encoded_visual = ModelUtils.create_visual_observation_encoder(
                        self.visual_in[i],
                        self.h_size,
                        ModelUtils.swish,
                        self.num_layers,
                        "stream_{}_visual_obs_encoder".format(i),
                        False,
                    )
                
                with tf.variable_scope(next_encoder_scope):
                    encoded_next_visual = ModelUtils.create_visual_observation_encoder(
                        self.next_visual_in[i],
                        self.h_size,
                        ModelUtils.swish,
                        self.num_layers,
                        "stream_{}_visual_obs_encoder".format(i),
                        reuse_encoder
                    )

                visual_encoders.append(encoded_visual)
                next_visual_encoders.append(encoded_next_visual)

            hidden_visual = tf.concat(visual_encoders, axis=1)
            hidden_next_visual = tf.concat(next_visual_encoders, axis=1)
            encoded_state_list.append(hidden_visual)
            encoded_next_state_list.append(hidden_next_visual)

        if self.vec_obs_size > 0:
            # Create the encoder ops for current and next vector input.
            # Note that these encoders are siamese.
            # Create input op for next (t+1) vector observation.
            self.next_vector_in = tf.placeholder(
                shape=[None, self.vec_obs_size],
                dtype=tf.float32,
                name="next_vector_observation",
            )

            if self.normalize:
                self.processed_vector_next = ModelUtils.normalize_vector_obs(
                    self.next_vector_in,
                    self.running_mean,
                    self.running_variance,
                    self.normalization_steps,
                )
            else:
                self.processed_vector_next = self.next_vector_in

            with tf.variable_scope("encoding"):
                encoded_vector_obs = ModelUtils.create_vector_observation_encoder(
                    self.vector_in,
                    self.h_size,
                    ModelUtils.swish,
                    self.num_layers,
                    "vector_obs_encoder",
                    False,
                )
            with tf.variable_scope(next_encoder_scope):
                encoded_next_vector_obs = ModelUtils.create_vector_observation_encoder(
                    self.processed_vector_next,
                    self.h_size,
                    ModelUtils.swish,
                    self.num_layers,
                    "vector_obs_encoder",
                    reuse_encoder
                )
            encoded_state_list.append(encoded_vector_obs)
            encoded_next_state_list.append(encoded_next_vector_obs)

        encoded_state = tf.concat(encoded_state_list, axis=1)
        encoded_next_state = tf.concat(encoded_next_state_list, axis=1)

        if var_latent:
            with tf.variable_scope("encoding/latent"):
                encoded_state_dist = GaussianEncoderDistribution(
                    encoded_state,
                    self.feature_size,
                )
                encoded_state = encoded_state_dist.sample()
            
            with tf.variable_scope(next_encoder_scope+"/latent"):
                encoded_next_state_dist = GaussianEncoderDistribution(
                    encoded_next_state,
                    self.feature_size,
                    reuse=reuse_encoder
                )
                encoded_next_state = encoded_next_state_dist.sample()
            return encoded_state, encoded_next_state, encoded_state_dist, encoded_next_state_dist
        else:
            with tf.variable_scope("encoding"):
                encoded_state = tf.layers.dense(
                            encoded_state,
                            self.feature_size,
                            name="latent"
                        )
            with tf.variable_scope(next_encoder_scope):
                encoded_next_state = tf.layers.dense(
                            encoded_next_state,
                            self.feature_size,
                            name="latent",
                            reuse=reuse_encoder
                        )

            return encoded_state, encoded_next_state

    def create_inverse_model(
        self, encoded_state: tf.Tensor, encoded_next_state: tf.Tensor
    ) -> None:
        """
        Creates inverse model TensorFlow ops for Curiosity module.
        Predicts action taken given current and future encoded states.
        :param encoded_state: Tensor corresponding to encoded current state.
        :param encoded_next_state: Tensor corresponding to encoded next state.
        """
        combined_input = tf.concat([encoded_state, encoded_next_state], axis=1)
        # hidden = tf.layers.dense(combined_input, 256, activation=ModelUtils.swish)
        if self.brain.vector_action_space_type == "continuous":
            pred_action = tf.layers.dense(
                combined_input, self.act_size[0], activation=None
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
                        combined_input, self.act_size[i], activation=tf.nn.softmax
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
        self, encoded_state: tf.Tensor, encoded_next_state: tf.Tensor
    ) -> None:
        """
        Creates forward model TensorFlow ops for Curiosity module.
        Predicts encoded future state based on encoded current state and given action.
        :param encoded_state: Tensor corresponding to encoded current state.
        :param encoded_next_state: Tensor corresponding to encoded next state.
        """
        combined_input = tf.concat(
            [encoded_state, self.current_action], axis=1
        )
        # hidden = tf.layers.dense(combined_input, 256, activation=ModelUtils.swish)
        predict = tf.layers.dense(
            combined_input,
            self.h_size
            * (self.vis_obs_size + int(self.vec_obs_size > 0)),
            activation=None,
        )
        self.predict = tf.layers.dense(
            predict,
            self.feature_size,
            name="latent"
        )
        squared_difference = 0.5 * tf.reduce_sum(
            tf.squared_difference(self.predict, encoded_next_state), axis=1
        )
        # self.intrinsic_reward = squared_difference
        self.forward_loss = tf.reduce_mean(
            tf.dynamic_partition(squared_difference, self.mask, 2)[1]
        )
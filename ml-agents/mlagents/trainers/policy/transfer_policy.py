import os
from typing import Any, Dict, Optional, List
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

        # Model-based learning
        self.feature_size = 16  # dimension of latent feature size
        self.separate_train = False  # whether to train policy and model separately
        
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

    def create_tf_graph(self, transfer=False) -> None:
        """
        Builds the tensorflow graph needed for this policy.
        """
        with self.graph.as_default():
            tf.set_random_seed(self.seed)
            _vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            if len(_vars) > 0:
                # We assume the first thing created in the graph is the Policy. If
                # already populated, don't create more tensors.
                return

            self.create_input_placeholders()

            # latent feature encoder
            if transfer:
                n_layers = self.num_layers + 1
            else:
                n_layers = self.num_layers
            self.encoder = self._create_encoder(
                self.visual_in,
                self.processed_vector_in,
                self.h_size,
                self.feature_size,
                n_layers,
                self.vis_encode_type
            )

            self.targ_encoder = self._create_target_encoder(
                self.h_size,
                self.feature_size,
                n_layers,
                self.vis_encode_type
            )

            self.hard_copy_encoder()

            self.predict = self._create_world_model(
                self.encoder,
                self.h_size,
                self.feature_size,
                self.num_layers,
                self.vis_encode_type
            )

            if self.use_continuous_act:
                self._create_cc_actor(
                    self.encoder,
                    self.h_size,
                    self.num_layers,
                    self.tanh_squash,
                    self.reparameterize,
                    self.condition_sigma_on_obs,
                )
            else:
                self._create_dc_actor(self.encoder, self.h_size, self.num_layers)
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
        load_nets = {"dynamics": ["policy", "predict", "value"], "observation": ["encoding"]}
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
            self.hard_copy_encoder()

    def _create_world_model(
        self,
        encoder: tf.Tensor,
        h_size: int,
        feature_size: int,
        num_layers: int,
        vis_encode_type: EncoderType,
    ) -> tf.Tensor:
        """"
        Builds the world model for state prediction
        """
        with self.graph.as_default():
            with tf.variable_scope("predict"):
                self.current_action = tf.placeholder(
                    shape=[None, sum(self.act_size)], dtype=tf.float32, name="current_action"
                )
                hidden_stream = ModelUtils.create_vector_observation_encoder(
                    tf.concat([encoder, self.current_action], axis=1),
                    h_size,
                    ModelUtils.swish,
                    num_layers,
                    scope=f"main_graph",
                    reuse=False
                )
                predict = tf.layers.dense(
                    hidden_stream,
                    feature_size,
                    name="next_state"
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

    def _create_target_encoder(
        self,
        h_size: int,
        feature_size: int,
        num_layers: int,
        vis_encode_type: EncoderType,
    ) -> tf.Tensor:
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

        with tf.variable_scope("target_enc"):
            hidden_stream_targ = ModelUtils.create_observation_streams(
                self.visual_next,
                self.processed_vector_next,
                1,
                h_size,
                num_layers,
                vis_encode_type,
            )[0]

            latent_targ = tf.layers.dense(
                    hidden_stream_targ,
                    feature_size,
                    name="latent"
                )
        return tf.stop_gradient(latent_targ)
    
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
                self.visual_in,
                self.processed_vector_in,
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
    
    def hard_copy_encoder(self):
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_enc')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoding')

        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def _create_cc_actor(
        self,
        encoded: tf.Tensor,
        h_size: int,
        num_layers: int,
        tanh_squash: bool = False,
        reparameterize: bool = False,
        condition_sigma_on_obs: bool = True,
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

        if self.separate_train:
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

    def _create_dc_actor(self, encoded: tf.Tensor, h_size: int, num_layers: int) -> None:
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

        if self.separate_train:
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
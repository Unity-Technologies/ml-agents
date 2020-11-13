from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np
from distutils.version import LooseVersion

from mlagents_envs.timers import timed

from mlagents.tf_utils import tf
from mlagents import tf_utils
from mlagents_envs.exception import UnityException
from mlagents_envs.base_env import BehaviorSpec
from mlagents_envs.logging_util import get_logger
from mlagents.trainers.policy import Policy
from mlagents.trainers.action_info import ActionInfo
from mlagents.trainers.trajectory import SplitObservations
from mlagents.trainers.behavior_id_utils import get_global_agent_id
from mlagents_envs.base_env import DecisionSteps
from mlagents.trainers.tf.models import ModelUtils
from mlagents.trainers.settings import TrainerSettings, EncoderType
from mlagents.trainers import __version__
from mlagents.trainers.tf.distributions import (
    GaussianDistribution,
    MultiCategoricalDistribution,
)
from mlagents.tf_utils.globals import get_rank


logger = get_logger(__name__)


# This is the version number of the inputs and outputs of the model, and
# determines compatibility with inference in Barracuda.
MODEL_FORMAT_VERSION = 2

EPSILON = 1e-6  # Small value to avoid divide by zero


class UnityPolicyException(UnityException):
    """
    Related to errors with the Trainer.
    """

    pass


class TFPolicy(Policy):
    """
    Contains a learning model, and the necessary
    functions to save/load models and create the input placeholders.
    """

    # Callback function used at the start of training to synchronize weights.
    # By default, this nothing.
    # If this needs to be used, it should be done from outside ml-agents.
    broadcast_global_variables: Callable[[int], None] = lambda root_rank: None

    def __init__(
        self,
        seed: int,
        behavior_spec: BehaviorSpec,
        trainer_settings: TrainerSettings,
        tanh_squash: bool = False,
        reparameterize: bool = False,
        condition_sigma_on_obs: bool = True,
        create_tf_graph: bool = True,
    ):
        """
        Initialized the policy.
        :param seed: Random seed to use for TensorFlow.
        :param brain: The corresponding Brain for this policy.
        :param trainer_settings: The trainer parameters.
        """
        super().__init__(
            seed,
            behavior_spec,
            trainer_settings,
            tanh_squash,
            reparameterize,
            condition_sigma_on_obs,
        )
        # for ghost trainer save/load snapshots
        self.assign_phs: List[tf.Tensor] = []
        self.assign_ops: List[tf.Operation] = []
        self.update_dict: Dict[str, tf.Tensor] = {}
        self.inference_dict: Dict[str, tf.Tensor] = {}
        self.first_normalization_update: bool = False

        self.graph = tf.Graph()
        self.sess = tf.Session(
            config=tf_utils.generate_session_config(), graph=self.graph
        )
        self._initialize_tensorflow_references()
        self.grads = None
        self.update_batch: Optional[tf.Operation] = None
        self.trainable_variables: List[tf.Variable] = []
        self.rank = get_rank()
        if create_tf_graph:
            self.create_tf_graph()

    def get_trainable_variables(self) -> List[tf.Variable]:
        """
        Returns a List of the trainable variables in this policy. if create_tf_graph hasn't been called,
        returns empty list.
        """
        return self.trainable_variables

    def create_tf_graph(self) -> None:
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
            encoded = self._create_encoder(
                self.visual_in,
                self.processed_vector_in,
                self.h_size,
                self.num_layers,
                self.vis_encode_type,
            )
            if self.use_continuous_act:
                self._create_cc_actor(
                    encoded,
                    self.tanh_squash,
                    self.reparameterize,
                    self.condition_sigma_on_obs,
                )
            else:
                self._create_dc_actor(encoded)
            self.trainable_variables = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy"
            )
            self.trainable_variables += tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="lstm"
            )  # LSTMs need to be root scope for Barracuda export

        self.inference_dict = {
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
        self.initialize()
        # Create assignment ops for Ghost Trainer
        self.init_load_weights()

    def _create_encoder(
        self,
        visual_in: List[tf.Tensor],
        vector_in: tf.Tensor,
        h_size: int,
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
        with tf.variable_scope("policy"):
            encoded = ModelUtils.create_observation_streams(
                self.visual_in,
                self.processed_vector_in,
                1,
                h_size,
                num_layers,
                vis_encode_type,
            )[0]
        return encoded

    @staticmethod
    def _convert_version_string(version_string: str) -> Tuple[int, ...]:
        """
        Converts the version string into a Tuple of ints (major_ver, minor_ver, patch_ver).
        :param version_string: The semantic-versioned version string (X.Y.Z).
        :return: A Tuple containing (major_ver, minor_ver, patch_ver).
        """
        ver = LooseVersion(version_string)
        return tuple(map(int, ver.version[0:3]))

    def initialize(self):
        with self.graph.as_default():
            init = tf.global_variables_initializer()
            self.sess.run(init)

    def get_weights(self):
        with self.graph.as_default():
            _vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            values = [v.eval(session=self.sess) for v in _vars]
            return values

    def init_load_weights(self):
        with self.graph.as_default():
            _vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            values = [v.eval(session=self.sess) for v in _vars]
            for var, value in zip(_vars, values):
                assign_ph = tf.placeholder(var.dtype, shape=value.shape)
                self.assign_phs.append(assign_ph)
                self.assign_ops.append(tf.assign(var, assign_ph))

    def load_weights(self, values):
        if len(self.assign_ops) == 0:
            logger.warning(
                "Calling load_weights in tf_policy but assign_ops is empty. Did you forget to call init_load_weights?"
            )
        with self.graph.as_default():
            feed_dict = {}
            for assign_ph, value in zip(self.assign_phs, values):
                feed_dict[assign_ph] = value
            self.sess.run(self.assign_ops, feed_dict=feed_dict)

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

    def get_action(
        self, decision_requests: DecisionSteps, worker_id: int = 0
    ) -> ActionInfo:
        """
        Decides actions given observations information, and takes them in environment.
        :param decision_requests: A dictionary of brain names and DecisionSteps from environment.
        :param worker_id: In parallel environment training, the unique id of the environment worker that
            the DecisionSteps came from. Used to construct a globally unique id for each agent.
        :return: an ActionInfo containing action, memories, values and an object
        to be passed to add experiences
        """
        if len(decision_requests) == 0:
            return ActionInfo.empty()

        global_agent_ids = [
            get_global_agent_id(worker_id, int(agent_id))
            for agent_id in decision_requests.agent_id
        ]  # For 1-D array, the iterator order is correct.

        run_out = self.evaluate(  # pylint: disable=assignment-from-no-return
            decision_requests, global_agent_ids
        )

        self.save_memories(global_agent_ids, run_out.get("memory_out"))
        self.check_nan_action(run_out.get("action"))

        return ActionInfo(
            action=run_out.get("action"),
            value=run_out.get("value"),
            outputs=run_out,
            agent_ids=decision_requests.agent_id,
        )

    def update(self, mini_batch, num_sequences):
        """
        Performs update of the policy.
        :param num_sequences: Number of experience trajectories in batch.
        :param mini_batch: Batch of experiences.
        :return: Results of update.
        """
        raise UnityPolicyException("The update function was not implemented.")

    def _execute_model(self, feed_dict, out_dict):
        """
        Executes model.
        :param feed_dict: Input dictionary mapping nodes to input data.
        :param out_dict: Output dictionary mapping names to nodes.
        :return: Dictionary mapping names to input data.
        """
        network_out = self.sess.run(list(out_dict.values()), feed_dict=feed_dict)
        run_out = dict(zip(list(out_dict.keys()), network_out))
        return run_out

    def fill_eval_dict(self, feed_dict, batched_step_result):
        vec_vis_obs = SplitObservations.from_observations(batched_step_result.obs)
        for i, _ in enumerate(vec_vis_obs.visual_observations):
            feed_dict[self.visual_in[i]] = vec_vis_obs.visual_observations[i]
        if self.use_vec_obs:
            feed_dict[self.vector_in] = vec_vis_obs.vector_observations
        if not self.use_continuous_act:
            mask = np.ones(
                (
                    len(batched_step_result),
                    sum(self.behavior_spec.action_spec.discrete_branches),
                ),
                dtype=np.float32,
            )
            if batched_step_result.action_mask is not None:
                mask = 1 - np.concatenate(batched_step_result.action_mask, axis=1)
            feed_dict[self.action_masks] = mask
        return feed_dict

    def get_current_step(self):
        """
        Gets current model step.
        :return: current model step.
        """
        step = self.sess.run(self.global_step)
        return step

    def set_step(self, step: int) -> int:
        """
        Sets current model step to step without creating additional ops.
        :param step: Step to set the current model step to.
        :return: The step the model was set to.
        """
        current_step = self.get_current_step()
        # Increment a positive or negative number of steps.
        return self.increment_step(step - current_step)

    def increment_step(self, n_steps):
        """
        Increments model step.
        """
        out_dict = {
            "global_step": self.global_step,
            "increment_step": self.increment_step_op,
        }
        feed_dict = {self.steps_to_increment: n_steps}
        return self.sess.run(out_dict, feed_dict=feed_dict)["global_step"]

    def get_inference_vars(self):
        """
        :return:list of inference var names
        """
        return list(self.inference_dict.keys())

    def get_update_vars(self):
        """
        :return:list of update var names
        """
        return list(self.update_dict.keys())

    def update_normalization(self, vector_obs: np.ndarray) -> None:
        """
        If this policy normalizes vector observations, this will update the norm values in the graph.
        :param vector_obs: The vector observations to add to the running estimate of the distribution.
        """
        if self.use_vec_obs and self.normalize:
            if self.first_normalization_update:
                self.sess.run(
                    self.init_normalization_op, feed_dict={self.vector_in: vector_obs}
                )
                self.first_normalization_update = False
            else:
                self.sess.run(
                    self.update_normalization_op, feed_dict={self.vector_in: vector_obs}
                )

    @property
    def use_vis_obs(self):
        return self.vis_obs_size > 0

    @property
    def use_vec_obs(self):
        return self.vec_obs_size > 0

    def _initialize_tensorflow_references(self):
        self.value_heads: Dict[str, tf.Tensor] = {}
        self.normalization_steps: Optional[tf.Variable] = None
        self.running_mean: Optional[tf.Variable] = None
        self.running_variance: Optional[tf.Variable] = None
        self.init_normalization_op: Optional[tf.Operation] = None
        self.update_normalization_op: Optional[tf.Operation] = None
        self.value: Optional[tf.Tensor] = None
        self.all_log_probs: tf.Tensor = None
        self.total_log_probs: Optional[tf.Tensor] = None
        self.entropy: Optional[tf.Tensor] = None
        self.output_pre: Optional[tf.Tensor] = None
        self.output: Optional[tf.Tensor] = None
        self.selected_actions: tf.Tensor = None
        self.action_masks: Optional[tf.Tensor] = None
        self.prev_action: Optional[tf.Tensor] = None
        self.memory_in: Optional[tf.Tensor] = None
        self.memory_out: Optional[tf.Tensor] = None
        self.version_tensors: Optional[Tuple[tf.Tensor, tf.Tensor, tf.Tensor]] = None

    def create_input_placeholders(self):
        with self.graph.as_default():
            (
                self.global_step,
                self.increment_step_op,
                self.steps_to_increment,
            ) = ModelUtils.create_global_steps()
            self.vector_in, self.visual_in = ModelUtils.create_input_placeholders(
                self.behavior_spec.observation_shapes
            )
            if self.normalize:
                self.first_normalization_update = True
                normalization_tensors = ModelUtils.create_normalizer(self.vector_in)
                self.update_normalization_op = normalization_tensors.update_op
                self.init_normalization_op = normalization_tensors.init_op
                self.normalization_steps = normalization_tensors.steps
                self.running_mean = normalization_tensors.running_mean
                self.running_variance = normalization_tensors.running_variance
                self.processed_vector_in = ModelUtils.normalize_vector_obs(
                    self.vector_in,
                    self.running_mean,
                    self.running_variance,
                    self.normalization_steps,
                )
            else:
                self.processed_vector_in = self.vector_in
                self.update_normalization_op = None

            self.batch_size_ph = tf.placeholder(
                shape=None, dtype=tf.int32, name="batch_size"
            )
            self.sequence_length_ph = tf.placeholder(
                shape=None, dtype=tf.int32, name="sequence_length"
            )
            self.mask_input = tf.placeholder(
                shape=[None], dtype=tf.float32, name="masks"
            )
            # Only needed for PPO, but needed for BC module
            self.epsilon = tf.placeholder(
                shape=[None, self.act_size[0]], dtype=tf.float32, name="epsilon"
            )
            self.mask = tf.cast(self.mask_input, tf.int32)

            tf.Variable(
                int(self.behavior_spec.action_spec.is_continuous()),
                name="is_continuous_control",
                trainable=False,
                dtype=tf.int32,
            )
            int_version = TFPolicy._convert_version_string(__version__)
            major_ver_t = tf.Variable(
                int_version[0],
                name="trainer_major_version",
                trainable=False,
                dtype=tf.int32,
            )
            minor_ver_t = tf.Variable(
                int_version[1],
                name="trainer_minor_version",
                trainable=False,
                dtype=tf.int32,
            )
            patch_ver_t = tf.Variable(
                int_version[2],
                name="trainer_patch_version",
                trainable=False,
                dtype=tf.int32,
            )
            self.version_tensors = (major_ver_t, minor_ver_t, patch_ver_t)
            tf.Variable(
                MODEL_FORMAT_VERSION,
                name="version_number",
                trainable=False,
                dtype=tf.int32,
            )
            tf.Variable(
                self.m_size, name="memory_size", trainable=False, dtype=tf.int32
            )
            if self.behavior_spec.action_spec.is_continuous():
                tf.Variable(
                    self.act_size[0],
                    name="action_output_shape",
                    trainable=False,
                    dtype=tf.int32,
                )
            else:
                tf.Variable(
                    sum(self.act_size),
                    name="action_output_shape",
                    trainable=False,
                    dtype=tf.int32,
                )

    def _create_cc_actor(
        self,
        encoded: tf.Tensor,
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

        with tf.variable_scope("policy"):
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

    def _create_dc_actor(self, encoded: tf.Tensor) -> None:
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

        self.action_masks = tf.placeholder(
            shape=[None, sum(self.act_size)], dtype=tf.float32, name="action_masks"
        )

        with tf.variable_scope("policy"):
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

from typing import Any, Dict, List, Optional
import abc
import numpy as np
from mlagents.tf_utils import tf
from mlagents import tf_utils
from mlagents_envs.exception import UnityException
from mlagents_envs.logging_util import get_logger
from mlagents.trainers.policy import Policy
from mlagents.trainers.action_info import ActionInfo
from mlagents.trainers.trajectory import SplitObservations
from mlagents.trainers.brain_conversion_utils import get_global_agent_id
from mlagents_envs.base_env import DecisionSteps
from mlagents.trainers.models import ModelUtils


logger = get_logger(__name__)


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

    def __init__(self, seed, brain, trainer_parameters, load=False):
        """
        Initialized the policy.
        :param seed: Random seed to use for TensorFlow.
        :param brain: The corresponding Brain for this policy.
        :param trainer_parameters: The trainer parameters.
        """
        self._version_number_ = 2
        self.m_size = 0

        # for ghost trainer save/load snapshots
        self.assign_phs = []
        self.assign_ops = []

        self.inference_dict = {}
        self.update_dict = {}
        self.sequence_length = 1
        self.seed = seed
        self.brain = brain

        self.act_size = brain.vector_action_space_size
        self.vec_obs_size = brain.vector_observation_space_size
        self.vis_obs_size = brain.number_visual_observations

        self.use_recurrent = trainer_parameters["use_recurrent"]
        self.memory_dict: Dict[str, np.ndarray] = {}
        self.num_branches = len(self.brain.vector_action_space_size)
        self.previous_action_dict: Dict[str, np.array] = {}
        self.normalize = trainer_parameters.get("normalize", False)
        self.use_continuous_act = brain.vector_action_space_type == "continuous"
        if self.use_continuous_act:
            self.num_branches = self.brain.vector_action_space_size[0]
        self.model_path = trainer_parameters["model_path"]
        self.initialize_path = trainer_parameters.get("init_path", None)
        self.keep_checkpoints = trainer_parameters.get("keep_checkpoints", 5)
        self.graph = tf.Graph()
        self.sess = tf.Session(
            config=tf_utils.generate_session_config(), graph=self.graph
        )
        self.saver = None
        self.seed = seed
        if self.use_recurrent:
            self.m_size = trainer_parameters["memory_size"]
            self.sequence_length = trainer_parameters["sequence_length"]
            if self.m_size == 0:
                raise UnityPolicyException(
                    "The memory size for brain {0} is 0 even "
                    "though the trainer uses recurrent.".format(brain.brain_name)
                )
            elif self.m_size % 2 != 0:
                raise UnityPolicyException(
                    "The memory size for brain {0} is {1} "
                    "but it must be divisible by 2.".format(
                        brain.brain_name, self.m_size
                    )
                )
        self._initialize_tensorflow_references()
        self.load = load

    @abc.abstractmethod
    def get_trainable_variables(self) -> List[tf.Variable]:
        """
        Returns a List of the trainable variables in this policy. if create_tf_graph hasn't been called,
        returns empty list.
        """
        pass

    @abc.abstractmethod
    def create_tf_graph(self):
        """
        Builds the tensorflow graph needed for this policy.
        """
        pass

    def _initialize_graph(self):
        with self.graph.as_default():
            self.saver = tf.train.Saver(max_to_keep=self.keep_checkpoints)
            init = tf.global_variables_initializer()
            self.sess.run(init)

    def _load_graph(self, model_path: str, reset_global_steps: bool = False) -> None:
        with self.graph.as_default():
            self.saver = tf.train.Saver(max_to_keep=self.keep_checkpoints)
            logger.info(
                "Loading model for brain {} from {}.".format(
                    self.brain.brain_name, model_path
                )
            )
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt is None:
                raise UnityPolicyException(
                    "The model {0} could not be loaded. Make "
                    "sure you specified the right "
                    "--run-id and that the previous run you are loading from had the same "
                    "behavior names.".format(model_path)
                )
            try:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            except tf.errors.NotFoundError:
                raise UnityPolicyException(
                    "The model {0} was found but could not be loaded. Make "
                    "sure the model is from the same version of ML-Agents, has the same behavior parameters, "
                    "and is using the same trainer configuration as the current run.".format(
                        model_path
                    )
                )
            if reset_global_steps:
                logger.info(
                    "Starting training from step 0 and saving to {}.".format(
                        self.model_path
                    )
                )
            else:
                logger.info(
                    "Resuming training from step {}.".format(self.get_current_step())
                )

    def initialize_or_load(self):
        # If there is an initialize path, load from that. Else, load from the set model path.
        # If load is set to True, don't reset steps to 0. Else, do. This allows a user to,
        # e.g., resume from an initialize path.
        reset_steps = not self.load
        if self.initialize_path is not None:
            self._load_graph(self.initialize_path, reset_global_steps=reset_steps)
        elif self.load:
            self._load_graph(self.model_path, reset_global_steps=reset_steps)
        else:
            self._initialize_graph()

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

    def evaluate(
        self, decision_requests: DecisionSteps, global_agent_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluates policy for the agent experiences provided.
        :param decision_requests: DecisionSteps input to network.
        :return: Output from policy based on self.inference_dict.
        """
        raise UnityPolicyException("The evaluate function was not implemented.")

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
                (len(batched_step_result), np.sum(self.brain.vector_action_space_size)),
                dtype=np.float32,
            )
            if batched_step_result.action_mask is not None:
                mask = 1 - np.concatenate(batched_step_result.action_mask, axis=1)
            feed_dict[self.action_masks] = mask
        return feed_dict

    def make_empty_memory(self, num_agents):
        """
        Creates empty memory for use with RNNs
        :param num_agents: Number of agents.
        :return: Numpy array of zeros.
        """
        return np.zeros((num_agents, self.m_size), dtype=np.float32)

    def save_memories(
        self, agent_ids: List[str], memory_matrix: Optional[np.ndarray]
    ) -> None:
        if memory_matrix is None:
            return
        for index, agent_id in enumerate(agent_ids):
            self.memory_dict[agent_id] = memory_matrix[index, :]

    def retrieve_memories(self, agent_ids: List[str]) -> np.ndarray:
        memory_matrix = np.zeros((len(agent_ids), self.m_size), dtype=np.float32)
        for index, agent_id in enumerate(agent_ids):
            if agent_id in self.memory_dict:
                memory_matrix[index, :] = self.memory_dict[agent_id]
        return memory_matrix

    def remove_memories(self, agent_ids):
        for agent_id in agent_ids:
            if agent_id in self.memory_dict:
                self.memory_dict.pop(agent_id)

    def make_empty_previous_action(self, num_agents):
        """
        Creates empty previous action for use with RNNs and discrete control
        :param num_agents: Number of agents.
        :return: Numpy array of zeros.
        """
        return np.zeros((num_agents, self.num_branches), dtype=np.int)

    def save_previous_action(
        self, agent_ids: List[str], action_matrix: Optional[np.ndarray]
    ) -> None:
        if action_matrix is None:
            return
        for index, agent_id in enumerate(agent_ids):
            self.previous_action_dict[agent_id] = action_matrix[index, :]

    def retrieve_previous_action(self, agent_ids: List[str]) -> np.ndarray:
        action_matrix = np.zeros((len(agent_ids), self.num_branches), dtype=np.int)
        for index, agent_id in enumerate(agent_ids):
            if agent_id in self.previous_action_dict:
                action_matrix[index, :] = self.previous_action_dict[agent_id]
        return action_matrix

    def remove_previous_action(self, agent_ids):
        for agent_id in agent_ids:
            if agent_id in self.previous_action_dict:
                self.previous_action_dict.pop(agent_id)

    def get_current_step(self):
        """
        Gets current model step.
        :return: current model step.
        """
        step = self.sess.run(self.global_step)
        return step

    def _set_step(self, step: int) -> int:
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

    def save_model(self, steps):
        """
        Saves the model
        :param steps: The number of steps the model was trained for
        :return:
        """
        with self.graph.as_default():
            last_checkpoint = self.model_path + "/model-" + str(steps) + ".ckpt"
            self.saver.save(self.sess, last_checkpoint)
            tf.train.write_graph(
                self.graph, self.model_path, "raw_graph_def.pb", as_text=False
            )

    def update_normalization(self, vector_obs: np.ndarray) -> None:
        """
        If this policy normalizes vector observations, this will update the norm values in the graph.
        :param vector_obs: The vector observations to add to the running estimate of the distribution.
        """
        if self.use_vec_obs and self.normalize:
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

    def create_input_placeholders(self):
        with self.graph.as_default():
            (
                self.global_step,
                self.increment_step_op,
                self.steps_to_increment,
            ) = ModelUtils.create_global_steps()
            self.visual_in = ModelUtils.create_visual_input_placeholders(
                self.brain.camera_resolutions
            )
            self.vector_in = ModelUtils.create_vector_input(self.vec_obs_size)
            if self.normalize:
                normalization_tensors = ModelUtils.create_normalizer(self.vector_in)
                self.update_normalization_op = normalization_tensors.update_op
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
                int(self.brain.vector_action_space_type == "continuous"),
                name="is_continuous_control",
                trainable=False,
                dtype=tf.int32,
            )
            tf.Variable(
                self._version_number_,
                name="version_number",
                trainable=False,
                dtype=tf.int32,
            )
            tf.Variable(
                self.m_size, name="memory_size", trainable=False, dtype=tf.int32
            )
            if self.brain.vector_action_space_type == "continuous":
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

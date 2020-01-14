import logging
from typing import Any, Dict, List, Optional

import numpy as np
from mlagents.tf_utils import tf
from mlagents import tf_utils

from mlagents_envs.exception import UnityException
from mlagents.trainers.policy import Policy
from mlagents.trainers.action_info import ActionInfo
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
from mlagents.trainers import tensorflow_to_barracuda as tf2bc
from mlagents.trainers.trajectory import SplitObservations
from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.brain import BrainInfo


logger = logging.getLogger("mlagents.trainers")


class UnityPolicyException(UnityException):
    """
    Related to errors with the Trainer.
    """

    pass


class TFPolicy(Policy):
    """
    Contains a learning model, and the necessary
    functions to interact with it to perform evaluate and updating.
    """

    possible_output_nodes = [
        "action",
        "value_estimate",
        "action_probs",
        "recurrent_out",
        "memory_size",
        "version_number",
        "is_continuous_control",
        "action_output_shape",
    ]

    def __init__(self, seed, brain, trainer_parameters):
        """
        Initialized the policy.
        :param seed: Random seed to use for TensorFlow.
        :param brain: The corresponding Brain for this policy.
        :param trainer_parameters: The trainer parameters.
        """
        self.m_size = None
        self.model = None
        self.inference_dict = {}
        self.update_dict = {}
        self.sequence_length = 1
        self.seed = seed
        self.brain = brain
        self.use_recurrent = trainer_parameters["use_recurrent"]
        self.memory_dict: Dict[str, np.ndarray] = {}
        self.reward_signals: Dict[str, "RewardSignal"] = {}
        self.num_branches = len(self.brain.vector_action_space_size)
        self.previous_action_dict: Dict[str, np.array] = {}
        self.normalize = trainer_parameters.get("normalize", False)
        self.use_continuous_act = brain.vector_action_space_type == "continuous"
        if self.use_continuous_act:
            self.num_branches = self.brain.vector_action_space_size[0]
        self.model_path = trainer_parameters["model_path"]
        self.keep_checkpoints = trainer_parameters.get("keep_checkpoints", 5)
        self.graph = tf.Graph()
        self.sess = tf.Session(
            config=tf_utils.generate_session_config(), graph=self.graph
        )
        self.saver = None
        if self.use_recurrent:
            self.m_size = trainer_parameters["memory_size"]
            self.sequence_length = trainer_parameters["sequence_length"]
            if self.m_size == 0:
                raise UnityPolicyException(
                    "The memory size for brain {0} is 0 even "
                    "though the trainer uses recurrent.".format(brain.brain_name)
                )
            elif self.m_size % 4 != 0:
                raise UnityPolicyException(
                    "The memory size for brain {0} is {1} "
                    "but it must be divisible by 4.".format(
                        brain.brain_name, self.m_size
                    )
                )

    def _initialize_graph(self):
        with self.graph.as_default():
            self.saver = tf.train.Saver(max_to_keep=self.keep_checkpoints)
            init = tf.global_variables_initializer()
            self.sess.run(init)

    def _load_graph(self):
        with self.graph.as_default():
            self.saver = tf.train.Saver(max_to_keep=self.keep_checkpoints)
            logger.info("Loading Model for brain {}".format(self.brain.brain_name))
            ckpt = tf.train.get_checkpoint_state(self.model_path)
            if ckpt is None:
                logger.info(
                    "The model {0} could not be found. Make "
                    "sure you specified the right "
                    "--run-id".format(self.model_path)
                )
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def evaluate(self, brain_info: BrainInfo) -> Dict[str, Any]:
        """
        Evaluates policy for the agent experiences provided.
        :param brain_info: BrainInfo input to network.
        :return: Output from policy based on self.inference_dict.
        """
        raise UnityPolicyException("The evaluate function was not implemented.")

    def get_action(self, brain_info: BrainInfo) -> ActionInfo:
        """
        Decides actions given observations information, and takes them in environment.
        :param brain_info: A dictionary of brain names and BrainInfo from environment.
        :return: an ActionInfo containing action, memories, values and an object
        to be passed to add experiences
        """
        if len(brain_info.agents) == 0:
            return ActionInfo([], [], {})

        agents_done = [
            agent
            for agent, done in zip(brain_info.agents, brain_info.local_done)
            if done
        ]

        self.remove_memories(agents_done)
        self.remove_previous_action(agents_done)

        run_out = self.evaluate(brain_info)  # pylint: disable=assignment-from-no-return
        self.save_memories(brain_info.agents, run_out.get("memory_out"))
        return ActionInfo(
            action=run_out.get("action"), value=run_out.get("value"), outputs=run_out
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

    def fill_eval_dict(self, feed_dict, brain_info):
        for i, _ in enumerate(brain_info.visual_observations):
            feed_dict[self.model.visual_in[i]] = brain_info.visual_observations[i]
        if self.use_vec_obs:
            feed_dict[self.model.vector_in] = brain_info.vector_observations
        if not self.use_continuous_act:
            feed_dict[self.model.action_masks] = brain_info.action_masks
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
        step = self.sess.run(self.model.global_step)
        return step

    def increment_step(self, n_steps):
        """
        Increments model step.
        """
        out_dict = {
            "global_step": self.model.global_step,
            "increment_step": self.model.increment_step,
        }
        feed_dict = {self.model.steps_to_increment: n_steps}
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
            last_checkpoint = self.model_path + "/model-" + str(steps) + ".cptk"
            self.saver.save(self.sess, last_checkpoint)
            tf.train.write_graph(
                self.graph, self.model_path, "raw_graph_def.pb", as_text=False
            )

    def export_model(self):
        """
        Exports latest saved model to .nn format for Unity embedding.
        """

        with self.graph.as_default():
            target_nodes = ",".join(self._process_graph())
            graph_def = self.graph.as_graph_def()
            output_graph_def = graph_util.convert_variables_to_constants(
                self.sess, graph_def, target_nodes.replace(" ", "").split(",")
            )
            frozen_graph_def_path = self.model_path + "/frozen_graph_def.pb"
            with gfile.GFile(frozen_graph_def_path, "wb") as f:
                f.write(output_graph_def.SerializeToString())
            tf2bc.convert(frozen_graph_def_path, self.model_path + ".nn")
            logger.info("Exported " + self.model_path + ".nn file")

    def _process_graph(self):
        """
        Gets the list of the output nodes present in the graph for inference
        :return: list of node names
        """
        all_nodes = [x.name for x in self.graph.as_graph_def().node]
        nodes = [x for x in all_nodes if x in self.possible_output_nodes]
        logger.info("List of nodes to export for brain :" + self.brain.brain_name)
        for n in nodes:
            logger.info("\t" + n)
        return nodes

    def update_normalization(self, vector_obs: np.ndarray) -> None:
        """
        If this policy normalizes vector observations, this will update the norm values in the graph.
        :param vector_obs: The vector observations to add to the running estimate of the distribution.
        """
        if self.use_vec_obs and self.normalize:
            self.sess.run(
                self.model.update_normalization,
                feed_dict={self.model.vector_in: vector_obs},
            )

    def get_batched_value_estimates(self, batch: AgentBuffer) -> Dict[str, np.ndarray]:
        feed_dict: Dict[tf.Tensor, Any] = {
            self.model.batch_size: batch.num_experiences,
            self.model.sequence_length: 1,  # We want to feed data in batch-wise, not time-wise.
        }

        if self.use_vec_obs:
            feed_dict[self.model.vector_in] = batch["vector_obs"]
        if self.model.vis_obs_size > 0:
            for i in range(len(self.model.visual_in)):
                _obs = batch["visual_obs%d" % i]
                feed_dict[self.model.visual_in[i]] = _obs
        if self.use_recurrent:
            feed_dict[self.model.memory_in] = batch["memory"]
        if not self.use_continuous_act and self.use_recurrent:
            feed_dict[self.model.prev_action] = batch["prev_action"]
        value_estimates = self.sess.run(self.model.value_heads, feed_dict)
        value_estimates = {k: np.squeeze(v, axis=1) for k, v in value_estimates.items()}

        return value_estimates

    def get_value_estimates(
        self, next_obs: List[np.ndarray], agent_id: str, done: bool
    ) -> Dict[str, float]:
        """
        Generates value estimates for bootstrapping.
        :param experience: AgentExperience to be used for bootstrapping.
        :param done: Whether or not this is the last element of the episode, in which case the value estimate will be 0.
        :return: The value estimate dictionary with key being the name of the reward signal and the value the
        corresponding value estimate.
        """

        feed_dict: Dict[tf.Tensor, Any] = {
            self.model.batch_size: 1,
            self.model.sequence_length: 1,
        }
        vec_vis_obs = SplitObservations.from_observations(next_obs)
        for i in range(len(vec_vis_obs.visual_observations)):
            feed_dict[self.model.visual_in[i]] = [vec_vis_obs.visual_observations[i]]

        if self.use_vec_obs:
            feed_dict[self.model.vector_in] = [vec_vis_obs.vector_observations]
        if self.use_recurrent:
            feed_dict[self.model.memory_in] = self.retrieve_memories([agent_id])
        if not self.use_continuous_act and self.use_recurrent:
            feed_dict[self.model.prev_action] = self.retrieve_previous_action(
                [agent_id]
            )
        value_estimates = self.sess.run(self.model.value_heads, feed_dict)

        value_estimates = {k: float(v) for k, v in value_estimates.items()}

        # If we're done, reassign all of the value estimates that need terminal states.
        if done:
            for k in value_estimates:
                if self.reward_signals[k].use_terminal_states:
                    value_estimates[k] = 0.0

        return value_estimates

    @property
    def vis_obs_size(self):
        return self.model.vis_obs_size

    @property
    def vec_obs_size(self):
        return self.model.vec_obs_size

    @property
    def use_vis_obs(self):
        return self.model.vis_obs_size > 0

    @property
    def use_vec_obs(self):
        return self.model.vec_obs_size > 0

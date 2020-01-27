import logging
import numpy as np
from typing import Any, Dict, Optional, List

from mlagents.tf_utils import tf

from mlagents_envs.timers import timed
from mlagents_envs.base_env import BatchedStepResult
from mlagents.trainers.brain import BrainParameters
from mlagents.trainers.models import EncoderType
from mlagents.trainers.models import LearningModel
from mlagents.trainers.tf_policy import TFPolicy
from mlagents.trainers.components.bc.module import BCModule

logger = logging.getLogger("mlagents.trainers")


class NNPolicy(TFPolicy):
    def __init__(
        self,
        seed: int,
        brain: BrainParameters,
        trainer_params: Dict[str, Any],
        is_training: bool,
        load: bool,
    ):
        """
        Policy for Proximal Policy Optimization Networks.
        :param seed: Random seed.
        :param brain: Assigned Brain object.
        :param trainer_params: Defined training parameters.
        :param is_training: Whether the model should be trained.
        :param load: Whether a pre-trained model will be loaded or a new one created.
        """
        with tf.variable_scope("policy"):
            super().__init__(seed, brain, trainer_params)

            self.stats_name_to_update_name = {
                "Losses/Value Loss": "value_loss",
                "Losses/Policy Loss": "policy_loss",
            }

            self.optimizer: Optional[tf.train.AdamOptimizer] = None
            self.grads = None
            self.update_batch: Optional[tf.Operation] = None
            num_layers = trainer_params["num_layers"]
            h_size = trainer_params["hidden_units"]
            if num_layers < 1:
                num_layers = 1
            vis_encode_type = EncoderType(
                trainer_params.get("vis_encode_type", "simple")
            )

            with self.graph.as_default():
                if self.use_continuous_act:
                    self.create_cc_actor(h_size, num_layers, vis_encode_type)
                else:
                    self.create_dc_actor(h_size, num_layers, vis_encode_type)
                self.bc_module: Optional[BCModule] = None
                # Create pretrainer if needed
                if "behavioral_cloning" in trainer_params:
                    BCModule.check_config(trainer_params["behavioral_cloning"])
                    self.bc_module = BCModule(
                        self,
                        policy_learning_rate=trainer_params["learning_rate"],
                        default_batch_size=trainer_params["batch_size"],
                        default_num_epoch=3,
                        **trainer_params["behavioral_cloning"],
                    )

        self.inference_dict: Dict[str, tf.Tensor] = {
            "action": self.output,
            "log_probs": self.all_log_probs,
            "entropy": self.entropy,
        }
        if self.use_continuous_act:
            self.inference_dict["pre_action"] = self.output_pre
        if self.use_recurrent:
            self.inference_dict["policy_memory_out"] = self.memory_out
        self.load = load

    def initialize_or_load(self):
        if self.load:
            self._load_graph()
        else:
            self._initialize_graph()

    @timed
    def evaluate(
        self, batched_step_result: BatchedStepResult, global_agent_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluates policy for the agent experiences provided.
        :param batched_step_result: BatchedStepResult object containing inputs.
        :param global_agent_ids: The global (with worker ID) agent ids of the data in the batched_step_result.
        :return: Outputs from network as defined by self.inference_dict.
        """
        feed_dict = {
            self.batch_size_ph: batched_step_result.n_agents(),
            self.sequence_length_ph: 1,
        }
        if self.use_recurrent:
            if not self.use_continuous_act:
                feed_dict[self.prev_action] = self.retrieve_previous_action(
                    global_agent_ids
                )
            feed_dict[self.memory_in] = self.retrieve_memories(global_agent_ids)
        feed_dict = self.fill_eval_dict(feed_dict, batched_step_result)
        run_out = self._execute_model(feed_dict, self.inference_dict)
        return run_out

    def create_cc_actor(
        self, h_size: int, num_layers: int, vis_encode_type: EncoderType
    ) -> None:
        """
        Creates Continuous control actor-critic model.
        :param h_size: Size of hidden linear layers.
        :param num_layers: Number of hidden linear layers.
        """
        hidden_stream = LearningModel.create_observation_streams(
            self.visual_in,
            self.processed_vector_in,
            1,
            h_size,
            num_layers,
            vis_encode_type,
            stream_scopes=["policy"],
        )[0]

        if self.use_recurrent:
            self.memory_in = tf.placeholder(
                shape=[None, self.m_size], dtype=tf.float32, name="recurrent_in"
            )
            _half_point = int(self.m_size / 2)
            hidden_policy, memory_policy_out = LearningModel.create_recurrent_encoder(
                hidden_stream,
                self.memory_in[:, :_half_point],
                self.sequence_length_ph,
                name="lstm_policy",
            )

            self.memory_out = memory_policy_out
        else:
            hidden_policy = hidden_stream

        mu = tf.layers.dense(
            hidden_policy,
            self.act_size[0],
            activation=None,
            kernel_initializer=LearningModel.scaled_init(0.01),
            reuse=tf.AUTO_REUSE,
        )

        self.log_sigma_sq = tf.get_variable(
            "log_sigma_squared",
            [self.act_size[0]],
            dtype=tf.float32,
            initializer=tf.zeros_initializer(),
        )

        sigma_sq = tf.exp(self.log_sigma_sq)

        self.epsilon = tf.random_normal(tf.shape(mu))
        # Clip and scale output to ensure actions are always within [-1, 1] range.
        self.output_pre = mu + tf.sqrt(sigma_sq) * self.epsilon
        output_post = tf.clip_by_value(self.output_pre, -3, 3) / 3
        self.output = tf.identity(output_post, name="action")
        self.selected_actions = tf.stop_gradient(output_post)

        # Compute probability of model output.
        all_probs = (
            -0.5 * tf.square(tf.stop_gradient(self.output_pre) - mu) / sigma_sq
            - 0.5 * tf.log(2.0 * np.pi)
            - 0.5 * self.log_sigma_sq
        )

        self.all_log_probs = tf.identity(all_probs, name="action_probs")

        single_dim_entropy = 0.5 * tf.reduce_mean(
            tf.log(2 * np.pi * np.e) + self.log_sigma_sq
        )
        # Make entropy the right shape
        self.entropy = tf.ones_like(tf.reshape(mu[:, 0], [-1])) * single_dim_entropy

        # We keep these tensors the same name, but use new nodes to keep code parallelism with discrete control.
        self.log_probs = tf.reduce_sum(
            (tf.identity(self.all_log_probs)), axis=1, keepdims=True
        )

    def create_dc_actor(
        self, h_size: int, num_layers: int, vis_encode_type: EncoderType
    ) -> None:
        """
        Creates Discrete control actor-critic model.
        :param h_size: Size of hidden linear layers.
        :param num_layers: Number of hidden linear layers.
        """
        hidden_stream = LearningModel.create_observation_streams(
            self.visual_in,
            self.processed_vector_in,
            1,
            h_size,
            num_layers,
            vis_encode_type,
            stream_scopes=["policy"],
        )[0]

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
            hidden_policy = tf.concat([hidden_stream, prev_action_oh], axis=1)

            self.memory_in = tf.placeholder(
                shape=[None, self.m_size], dtype=tf.float32, name="recurrent_in"
            )
            _half_point = int(self.m_size / 2)
            hidden_policy, memory_policy_out = LearningModel.create_recurrent_encoder(
                hidden_policy,
                self.memory_in[:, :_half_point],
                self.sequence_length_ph,
                name="lstm_policy",
            )

            self.memory_out = memory_policy_out
        else:
            hidden_policy = hidden_stream

        policy_branches = []
        for size in self.act_size:
            policy_branches.append(
                tf.layers.dense(
                    hidden_policy,
                    size,
                    activation=None,
                    use_bias=False,
                    kernel_initializer=LearningModel.scaled_init(0.01),
                )
            )

        self.all_log_probs = tf.concat(policy_branches, axis=1, name="action_probs")

        self.action_masks = tf.placeholder(
            shape=[None, sum(self.act_size)], dtype=tf.float32, name="action_masks"
        )
        output, _, normalized_logits = LearningModel.create_discrete_action_masking_layer(
            self.all_log_probs, self.action_masks, self.act_size
        )

        self.output = tf.identity(output)
        self.normalized_logits = tf.identity(normalized_logits, name="action")

        self.action_holder = tf.placeholder(
            shape=[None, len(policy_branches)], dtype=tf.int32, name="action_holder"
        )
        self.action_oh = tf.concat(
            [
                tf.one_hot(self.action_holder[:, i], self.act_size[i])
                for i in range(len(self.act_size))
            ],
            axis=1,
        )
        self.selected_actions = tf.stop_gradient(self.action_oh)

        action_idx = [0] + list(np.cumsum(self.act_size))

        self.entropy = tf.reduce_sum(
            (
                tf.stack(
                    [
                        tf.nn.softmax_cross_entropy_with_logits_v2(
                            labels=tf.nn.softmax(
                                self.all_log_probs[:, action_idx[i] : action_idx[i + 1]]
                            ),
                            logits=self.all_log_probs[
                                :, action_idx[i] : action_idx[i + 1]
                            ],
                        )
                        for i in range(len(self.act_size))
                    ],
                    axis=1,
                )
            ),
            axis=1,
        )

        self.log_probs = tf.reduce_sum(
            (
                tf.stack(
                    [
                        -tf.nn.softmax_cross_entropy_with_logits_v2(
                            labels=self.action_oh[:, action_idx[i] : action_idx[i + 1]],
                            logits=normalized_logits[
                                :, action_idx[i] : action_idx[i + 1]
                            ],
                        )
                        for i in range(len(self.act_size))
                    ],
                    axis=1,
                )
            ),
            axis=1,
            keepdims=True,
        )

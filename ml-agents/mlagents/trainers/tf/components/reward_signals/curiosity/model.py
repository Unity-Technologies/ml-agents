from typing import List, Tuple
from mlagents.tf_utils import tf

from mlagents.trainers.tf.models import ModelUtils
from mlagents.trainers.policy.tf_policy import TFPolicy


class CuriosityModel:
    def __init__(
        self, policy: TFPolicy, encoding_size: int = 128, learning_rate: float = 3e-4
    ):
        """
        Creates the curiosity model for the Curiosity reward Generator
        :param policy: The policy being trained
        :param encoding_size: The size of the encoding for the Curiosity module
        :param learning_rate: The learning rate for the curiosity module
        """
        self.encoding_size = encoding_size
        self.policy = policy
        self.next_visual_in: List[tf.Tensor] = []
        encoded_state, encoded_next_state = self.create_curiosity_encoders()
        self.create_inverse_model(encoded_state, encoded_next_state)
        self.create_forward_model(encoded_state, encoded_next_state)
        self.create_loss(learning_rate)

    def create_curiosity_encoders(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Creates state encoders for current and future observations.
        Used for implementation of ﻿Curiosity-driven Exploration by Self-supervised Prediction
        See https://arxiv.org/abs/1705.05363 for more details.
        :return: current and future state encoder tensors.
        """
        encoded_state_list = []
        encoded_next_state_list = []

        # Create input ops for next (t+1) visual observations.
        self.next_vector_in, self.next_visual_in = ModelUtils.create_input_placeholders(
            self.policy.behavior_spec.observation_shapes, name_prefix="curiosity_next_"
        )

        if self.next_visual_in:
            visual_encoders = []
            next_visual_encoders = []
            for i, (vis_in, next_vis_in) in enumerate(
                zip(self.policy.visual_in, self.next_visual_in)
            ):
                # Create the encoder ops for current and next visual input.
                # Note that these encoders are siamese.
                encoded_visual = ModelUtils.create_visual_observation_encoder(
                    vis_in,
                    self.encoding_size,
                    ModelUtils.swish,
                    1,
                    f"curiosity_stream_{i}_visual_obs_encoder",
                    False,
                )

                encoded_next_visual = ModelUtils.create_visual_observation_encoder(
                    next_vis_in,
                    self.encoding_size,
                    ModelUtils.swish,
                    1,
                    f"curiosity_stream_{i}_visual_obs_encoder",
                    True,
                )
                visual_encoders.append(encoded_visual)
                next_visual_encoders.append(encoded_next_visual)

            hidden_visual = tf.concat(visual_encoders, axis=1)
            hidden_next_visual = tf.concat(next_visual_encoders, axis=1)
            encoded_state_list.append(hidden_visual)
            encoded_next_state_list.append(hidden_next_visual)

        if self.policy.vec_obs_size > 0:
            encoded_vector_obs = ModelUtils.create_vector_observation_encoder(
                self.policy.vector_in,
                self.encoding_size,
                ModelUtils.swish,
                2,
                "curiosity_vector_obs_encoder",
                False,
            )
            encoded_next_vector_obs = ModelUtils.create_vector_observation_encoder(
                self.next_vector_in,
                self.encoding_size,
                ModelUtils.swish,
                2,
                "curiosity_vector_obs_encoder",
                True,
            )
            encoded_state_list.append(encoded_vector_obs)
            encoded_next_state_list.append(encoded_next_vector_obs)
        encoded_state = tf.concat(encoded_state_list, axis=1)
        encoded_next_state = tf.concat(encoded_next_state_list, axis=1)
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
        hidden = tf.layers.dense(combined_input, 256, activation=ModelUtils.swish)
        if self.policy.behavior_spec.action_spec.is_continuous():
            pred_action = tf.layers.dense(
                hidden, self.policy.act_size[0], activation=None
            )
            squared_difference = tf.reduce_sum(
                tf.squared_difference(pred_action, self.policy.selected_actions), axis=1
            )
            self.inverse_loss = tf.reduce_mean(
                tf.dynamic_partition(squared_difference, self.policy.mask, 2)[1]
            )
        else:
            pred_action = tf.concat(
                [
                    tf.layers.dense(
                        hidden, self.policy.act_size[i], activation=tf.nn.softmax
                    )
                    for i in range(len(self.policy.act_size))
                ],
                axis=1,
            )
            cross_entropy = tf.reduce_sum(
                -tf.log(pred_action + 1e-10) * self.policy.selected_actions, axis=1
            )
            self.inverse_loss = tf.reduce_mean(
                tf.dynamic_partition(cross_entropy, self.policy.mask, 2)[1]
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
            [encoded_state, self.policy.selected_actions], axis=1
        )
        hidden = tf.layers.dense(combined_input, 256, activation=ModelUtils.swish)
        pred_next_state = tf.layers.dense(
            hidden,
            self.encoding_size
            * (self.policy.vis_obs_size + int(self.policy.vec_obs_size > 0)),
            activation=None,
        )
        squared_difference = 0.5 * tf.reduce_sum(
            tf.squared_difference(pred_next_state, encoded_next_state), axis=1
        )
        self.intrinsic_reward = squared_difference
        self.forward_loss = tf.reduce_mean(
            tf.dynamic_partition(squared_difference, self.policy.mask, 2)[1]
        )

    def create_loss(self, learning_rate: float) -> None:
        """
        Creates the loss node of the model as well as the update_batch optimizer to update the model.
        :param learning_rate: The learning rate for the optimizer.
        """
        self.loss = 10 * (0.2 * self.forward_loss + 0.8 * self.inverse_loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.update_batch = optimizer.minimize(self.loss)

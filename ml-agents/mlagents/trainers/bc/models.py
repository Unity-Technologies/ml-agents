import tensorflow as tf
import tensorflow.contrib.layers as c_layers
from mlagents.trainers.models import LearningModel


class BehavioralCloningModel(LearningModel):
    def __init__(
        self,
        brain,
        h_size=128,
        lr=1e-4,
        n_layers=2,
        m_size=128,
        normalize=False,
        use_recurrent=False,
        seed=0,
    ):
        LearningModel.__init__(self, m_size, normalize, use_recurrent, brain, seed)
        num_streams = 1
        hidden_streams = self.create_observation_streams(num_streams, h_size, n_layers)
        hidden = hidden_streams[0]
        self.dropout_rate = tf.placeholder(
            dtype=tf.float32, shape=[], name="dropout_rate"
        )
        hidden_reg = tf.layers.dropout(hidden, self.dropout_rate)
        if self.use_recurrent:
            tf.Variable(
                self.m_size, name="memory_size", trainable=False, dtype=tf.int32
            )
            self.memory_in = tf.placeholder(
                shape=[None, self.m_size], dtype=tf.float32, name="recurrent_in"
            )
            hidden_reg, self.memory_out = self.create_recurrent_encoder(
                hidden_reg, self.memory_in, self.sequence_length
            )
            self.memory_out = tf.identity(self.memory_out, name="recurrent_out")

        if brain.vector_action_space_type == "discrete":
            policy_branches = []
            for size in self.act_size:
                policy_branches.append(
                    tf.layers.dense(
                        hidden_reg,
                        size,
                        activation=None,
                        use_bias=False,
                        kernel_initializer=c_layers.variance_scaling_initializer(
                            factor=0.01
                        ),
                    )
                )
            self.action_probs = tf.concat(
                [tf.nn.softmax(branch) for branch in policy_branches],
                axis=1,
                name="action_probs",
            )
            self.action_masks = tf.placeholder(
                shape=[None, sum(self.act_size)], dtype=tf.float32, name="action_masks"
            )
            self.sample_action_float, _, normalized_logits = self.create_discrete_action_masking_layer(
                tf.concat(policy_branches, axis=1), self.action_masks, self.act_size
            )
            tf.identity(normalized_logits, name="action")
            self.sample_action = tf.cast(self.sample_action_float, tf.int32)
            self.true_action = tf.placeholder(
                shape=[None, len(policy_branches)],
                dtype=tf.int32,
                name="teacher_action",
            )
            self.action_oh = tf.concat(
                [
                    tf.one_hot(self.true_action[:, i], self.act_size[i])
                    for i in range(len(self.act_size))
                ],
                axis=1,
            )
            self.loss = tf.reduce_sum(
                -tf.log(self.action_probs + 1e-10) * self.action_oh
            )
            self.action_percent = tf.reduce_mean(
                tf.cast(
                    tf.equal(
                        tf.cast(tf.argmax(self.action_probs, axis=1), tf.int32),
                        self.sample_action,
                    ),
                    tf.float32,
                )
            )
        else:
            self.policy = tf.layers.dense(
                hidden_reg,
                self.act_size[0],
                activation=None,
                use_bias=False,
                name="pre_action",
                kernel_initializer=c_layers.variance_scaling_initializer(factor=0.01),
            )
            self.clipped_sample_action = tf.clip_by_value(self.policy, -1, 1)
            self.sample_action = tf.identity(self.clipped_sample_action, name="action")
            self.true_action = tf.placeholder(
                shape=[None, self.act_size[0]], dtype=tf.float32, name="teacher_action"
            )
            self.clipped_true_action = tf.clip_by_value(self.true_action, -1, 1)
            self.loss = tf.reduce_sum(
                tf.squared_difference(self.clipped_true_action, self.sample_action)
            )

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update = optimizer.minimize(self.loss)

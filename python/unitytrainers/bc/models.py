import tensorflow as tf
import tensorflow.contrib.layers as c_layers
from unitytrainers.models import LearningModel


class BehavioralCloningModel(LearningModel):
    def __init__(self, brain, h_size=128, lr=1e-4, n_layers=2, m_size=128,
                 normalize=False, use_recurrent=False):
        LearningModel.__init__(self, m_size, normalize, use_recurrent, brain)

        num_streams = 1
        hidden_streams = self.create_observation_streams(num_streams, h_size, n_layers)
        hidden = hidden_streams[0]
        self.dropout_rate = tf.placeholder(dtype=tf.float32, shape=[], name="dropout_rate")
        hidden_reg = tf.layers.dropout(hidden, self.dropout_rate)
        if self.use_recurrent:
            tf.Variable(self.m_size, name="memory_size", trainable=False, dtype=tf.int32)
            self.memory_in = tf.placeholder(shape=[None, self.m_size], dtype=tf.float32, name='recurrent_in')
            hidden_reg, self.memory_out = self.create_recurrent_encoder(hidden_reg, self.memory_in,
                                                                        self.sequence_length)
            self.memory_out = tf.identity(self.memory_out, name='recurrent_out')
        self.policy = tf.layers.dense(hidden_reg, self.a_size, activation=None, use_bias=False, name='pre_action',
                                      kernel_initializer=c_layers.variance_scaling_initializer(factor=0.01))

        if brain.vector_action_space_type == "discrete":
            self.action_probs = tf.nn.softmax(self.policy)
            self.sample_action_float = tf.multinomial(self.policy, 1)
            self.sample_action_float = tf.identity(self.sample_action_float, name="action")
            self.sample_action = tf.cast(self.sample_action_float, tf.int32)
            self.true_action = tf.placeholder(shape=[None], dtype=tf.int32, name="teacher_action")
            self.action_oh = tf.one_hot(self.true_action, self.a_size)
            self.loss = tf.reduce_sum(-tf.log(self.action_probs + 1e-10) * self.action_oh)
            self.action_percent = tf.reduce_mean(tf.cast(
                tf.equal(tf.cast(tf.argmax(self.action_probs, axis=1), tf.int32), self.sample_action), tf.float32))
        else:
            self.clipped_sample_action = tf.clip_by_value(self.policy, -1, 1)
            self.sample_action = tf.identity(self.clipped_sample_action, name="action")
            self.true_action = tf.placeholder(shape=[None, self.a_size], dtype=tf.float32, name="teacher_action")
            self.clipped_true_action = tf.clip_by_value(self.true_action, -1, 1)
            self.loss = tf.reduce_sum(tf.squared_difference(self.clipped_true_action, self.sample_action))

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update = optimizer.minimize(self.loss)

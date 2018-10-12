import tensorflow as tf


class Discriminator(object):
    def __init__(self, o_size, a_size, h_size, lr):
        self.h_size = h_size
        self.make_inputs(o_size, a_size)
        self.make_network()
        self.make_loss(lr)

    def make_inputs(self, o_size, a_size):
        self.obs_in_expert = tf.placeholder(shape=[None, o_size], dtype=tf.float32)
        self.action_in_expert = tf.placeholder(shape=[None, a_size], dtype=tf.float32)
        self.obs_in_policy = tf.placeholder(shape=[None, o_size], dtype=tf.float32)
        self.action_in_policy = tf.placeholder(shape=[None, a_size], dtype=tf.float32)

    def make_discriminator(self, state_in, action_in, reuse):
        with tf.variable_scope("discriminator"):
            concat_input = tf.concat([state_in, action_in], axis=1)

            hidden_1 = tf.layers.dense(
                concat_input, self.h_size, activation=tf.nn.elu,
                name="d_hidden_1", reuse=reuse)

            hidden_2 = tf.layers.dense(
                hidden_1, self.h_size, activation=tf.nn.elu,
                name="d_hidden_2", reuse=reuse)

            d_value = tf.layers.dense(hidden_2, 1, activation=tf.nn.sigmoid,
                                      name="d_value", reuse=reuse)
            return d_value

    def make_loss(self, learning_rate):
        self.de = tf.reduce_mean(self.d_expert)
        self.dp = tf.reduce_mean(self.d_policy)
        self.d_loss = -tf.reduce_mean(
            tf.log(self.d_expert + 1e-10) + tf.log(1.0 - self.d_policy + 1e-10))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.update_batch = optimizer.minimize(self.d_loss)

    def make_network(self):
        self.d_expert = self.make_discriminator(self.obs_in_expert, self.action_in_expert, False)
        self.d_policy = self.make_discriminator(self.obs_in_policy, self.action_in_policy, True)

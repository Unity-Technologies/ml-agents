import logging
from mlagents.trainers.trainer import UnityTrainerException
import tensorflow as tf

logger = logging.getLogger("mlagents.trainers")


class RewardModel(object):
    def create_normalizer(self, reward_input, name):
        mean_reward_input = tf.reduce_mean(reward_input, axis=0)
        self.normalization_steps = tf.get_variable("{}_normalization_steps".format(name), [],
                                            trainable=False, dtype=tf.int32,
                                            initializer=tf.ones_initializer())
        self.running_mean = tf.get_variable("{}_running_mean".format(name), [],
                                            trainable=False, dtype=tf.float32,
                                            initializer=tf.zeros_initializer())
        self.running_variance = tf.get_variable("{}_running_variance".format(name), [],
                                                trainable=False, dtype=tf.float32,
                                                initializer=tf.ones_initializer())
        self.update_normalization = self.create_normalizer_update(mean_reward_input)
        normalized_reward = tf.clip_by_value((reward_input - self.running_mean) / tf.sqrt(
            self.running_variance / (tf.cast(self.normalization_steps, tf.float32) + 1)), -5, 5,
                                                 name="{}_normalized_reward".format(name))
        return normalized_reward

    def create_normalizer_update(self, reward_input):
        new_mean = self.running_mean + (reward_input - self.running_mean) / \
                   tf.cast(tf.add(self.normalization_steps, 1), tf.float32)
        new_variance = self.running_variance + (reward_input - new_mean) * \
                       (reward_input - self.running_mean)
        update_mean = tf.assign(self.running_mean, new_mean)
        update_variance = tf.assign(self.running_variance, new_variance)
        update_norm_step = tf.assign(self.normalization_steps, self.normalization_steps + 1)
        return tf.group([update_mean, update_variance, update_norm_step])
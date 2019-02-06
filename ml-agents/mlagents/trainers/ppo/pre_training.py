import logging
import tensorflow as tf
import numpy as np

from mlagents.trainers.models import LearningModel
from mlagents.trainers.demo_loader import demo_to_buffer

logger = logging.getLogger("mlagents.trainers")


class PreTraining(object):
    sequence_length_name = 'sequence_length'
    use_recurrent_name = 'use_recurrent'
    memory_size = 'memory_size'
    # TODO : Implement recurrent
    # TODO : Implement Discrete Control
    # TODO : Pretrain the critic ? All of the critics ? Use a gradient gate ?
    # TODO : tune lambdas during the training process (start at 1 and slowly go to right value)
    #  TODO : check the equality between brain and brain_params

    def __init__(self, sess, policy_model: LearningModel, brain, parameters):
        self.n_sequences = 128
        self.n_epoch = 500
        self.batches_per_epoch = 10

        self.use_recurrent = parameters[self.use_recurrent_name]
        self.sequence_length = 1
        self.m_size = 0
        if self.use_recurrent:
            self.sequence_length = parameters[self.sequence_length_name]
            self.m_size = parameters[self.memory_size]

        self.sess = sess
        self.policy_model = policy_model
        self.brain = brain

        buffer_name = parameters["demo_path"]
        brain_params, self.demonstration_buffer = demo_to_buffer(
            buffer_name,
            self.sequence_length)

        self.n_sequences = min(self.n_sequences, len(self.demonstration_buffer.update_buffer['actions']))
        if self.use_recurrent and not self.brain.vector_action_space_type == "continuous":
            # self.demonstration_buffer.update_buffer['prev_action'].set(
            #     []+self.demonstration_buffer.update_buffer['actions'][:-1])
            print(self.demonstration_buffer.update_buffer['actions'])
            print(np.shape(self.demonstration_buffer.update_buffer['actions']))

        self._add_loss(0.005)
        self.out_dict = {
            "loss": self.loss,
            "update": self.update_batch
        }
        if parameters["normalize"]:
            self.out_dict['update_normalization'] = self.policy_model.update_normalization
            self.out_dict['increment_step'] = self.policy_model.increment_step

    def _add_loss(self, learning_rate):
        selected_action = self.policy_model.output
        action_size = self.policy_model.act_size
        if self.brain.vector_action_space_type == "continuous":
            self.expert_action = tf.placeholder(shape=[None, action_size[0]],
                                                dtype=tf.float32,
                                                name="teacher_action")
            entropy = 0.5 * tf.reduce_mean(tf.log(2 * np.pi * np.e) + self.policy_model.log_sigma_sq)
            self.loss = tf.reduce_sum(tf.squared_difference(selected_action, self.expert_action)) / self.n_sequences \
                        - tf.reduce_sum(entropy)
        else:
            log_probs = self.policy_model.all_log_probs
            self.expert_action = tf.placeholder(shape=[None, len(action_size)], dtype=tf.int32)
            expert_action_oh = tf.concat([
                tf.one_hot(self.expert_action[:, i], action_size[i]) for i in range(len(action_size))], axis=1)
            action_idx = [0] + list(np.cumsum(action_size))
            entropy = tf.reduce_sum((tf.stack([
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=tf.nn.softmax(log_probs[:, action_idx[i]:action_idx[i + 1]]),
                    logits=log_probs[:, action_idx[i]:action_idx[i + 1]])
                for i in range(len(action_size))], axis=1)), axis=1)
            self.loss = -tf.reduce_sum(tf.log(tf.nn.softmax(log_probs)+ 0*1e-10) * expert_action_oh) - 1 * tf.reduce_sum(entropy)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.update_batch = optimizer.minimize(self.loss)

    def update_policy(self):
        """
        Updates the policy.
        """
        for iteration in range(self.n_epoch):
            self.demonstration_buffer.update_buffer.shuffle()
            batch_losses = []
            num_batches = min(len(self.demonstration_buffer.update_buffer['actions']) //
                              self.n_sequences, self.batches_per_epoch)
            for i in range(num_batches):
                update_buffer = self.demonstration_buffer.update_buffer
                start = i * self.n_sequences
                end = (i + 1) * self.n_sequences
                mini_batch = update_buffer.make_mini_batch(start, end)
                loss = self._update(mini_batch, self.n_sequences)
                batch_losses.append(loss)
            logger.info("Pre-Training loss at iteration "+str(iteration)+" : "+str(np.mean(batch_losses)))

    def _update(self, mini_batch, num_sequences):
        """
        Performs update on model.
        :param mini_batch: Batch of experiences.
        :param num_sequences: Number of sequences to process.
        :return: Results of update.
        """

        feed_dict = {#self.policy.model.dropout_rate: self.update_rate,
                     self.policy_model.batch_size: num_sequences,
                     self.policy_model.sequence_length: self.sequence_length,
                     # self.true_return : 10*np.ones(num_sequences)
                        }
        if self.brain.vector_action_space_type == "continuous":
            feed_dict[self.expert_action] = mini_batch['actions']. \
                reshape([-1, self.brain.vector_action_space_size[0]])
            feed_dict[self.policy_model.epsilon] = np.random.normal(
                size=(1, self.policy_model.act_size[0]))
        else:
            feed_dict[self.expert_action] = mini_batch['actions'].reshape(
                [-1, len(self.brain.vector_action_space_size)])
            feed_dict[self.policy_model.action_masks] = np.ones(
                (num_sequences, sum(self.brain.vector_action_space_size)))
        if self.brain.vector_observation_space_size > 0:
            apparent_obs_size = self.brain.vector_observation_space_size * \
                                self.brain.num_stacked_vector_observations
            feed_dict[self.policy_model.vector_in] = mini_batch['vector_obs'] \
                .reshape([-1, apparent_obs_size])
        for i, _ in enumerate(self.policy_model.visual_in):
            visual_obs = mini_batch['visual_obs%d' % i]
            feed_dict[self.policy_model.visual_in[i]] = visual_obs
        if self.use_recurrent:
            feed_dict[self.policy_model.memory_in] = np.zeros([num_sequences, self.m_size])
            if not self.brain.vector_action_space_type == "continuous":
                print(mini_batch.keys())
                # feed_dict[self.policy_model.prev_action] = mini_batch['prev_action'] \
                #     .reshape([-1, len(self.policy_model.act_size)])
                feed_dict[self.policy_model.prev_action] = np.zeros((num_sequences* self.sequence_length,
                                                                     len(self.policy_model.act_size)))

        run_out = self.execute_model(feed_dict, self.out_dict)
        return run_out['loss']

    def execute_model(self, feed_dict, out_dict):
        """
        Executes model.
        :param feed_dict: Input dictionary mapping nodes to input data.
        :param out_dict: Output dictionary mapping names to nodes.
        :return: Dictionary mapping names to input data.
        """
        network_out = self.sess.run(list(out_dict.values()), feed_dict=feed_dict)
        run_out = dict(zip(list(out_dict.keys()), network_out))
        return run_out

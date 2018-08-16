import logging

import numpy as np
import tensorflow as tf
from unitytrainers.ppo.models import PPOModel
from unitytrainers.trainer import UnityTrainerException

logger = logging.getLogger("unityagents")


class PPOPolicy(object):
    def __init__(self, seed, env, brain_name, trainer_parameters, sess, is_training):
        self.m_size = None
        self.has_updated = False
        self.sequence_length = 1
        self.brain = env.brains[brain_name]
        self.use_curiosity = bool(trainer_parameters['use_curiosity'])
        self.variable_scope = trainer_parameters['graph_scope']
        self.use_recurrent = trainer_parameters["use_recurrent"]
        self.is_continuous_action = (env.brains[brain_name].vector_action_space_type == "continuous")
        self.use_visual_obs = (env.brains[brain_name].number_visual_observations > 0)
        self.use_vector_obs = (env.brains[brain_name].vector_observation_space_size > 0)
        self.sess = sess
        if self.use_recurrent:
            self.m_size = trainer_parameters["memory_size"]
            self.sequence_length = trainer_parameters["sequence_length"]
            if self.m_size == 0:
                raise UnityTrainerException("The memory size for brain {0} is 0 even though the trainer uses recurrent."
                                            .format(brain_name))
            elif self.m_size % 4 != 0:
                raise UnityTrainerException("The memory size for brain {0} is {1} but it must be divisible by 4."
                                            .format(brain_name, self.m_size))

        with tf.variable_scope(self.variable_scope):
            tf.set_random_seed(seed)
            self.model = PPOModel(env.brains[brain_name],
                                  lr=float(trainer_parameters['learning_rate']),
                                  h_size=int(trainer_parameters['hidden_units']),
                                  epsilon=float(trainer_parameters['epsilon']),
                                  beta=float(trainer_parameters['beta']),
                                  max_step=float(trainer_parameters['max_steps']),
                                  normalize=trainer_parameters['normalize'],
                                  use_recurrent=trainer_parameters['use_recurrent'],
                                  num_layers=int(trainer_parameters['num_layers']),
                                  m_size=self.m_size,
                                  use_curiosity=bool(trainer_parameters['use_curiosity']),
                                  curiosity_strength=float(trainer_parameters['curiosity_strength']),
                                  curiosity_enc_size=float(trainer_parameters['curiosity_enc_size']))

        self.inference_run_list = [self.model.output, self.model.all_log_probs, self.model.value,
                                   self.model.entropy, self.model.learning_rate]
        if self.is_continuous_action:
            self.inference_run_list.append(self.model.output_pre)
        if self.use_recurrent:
            self.inference_run_list.extend([self.model.memory_out])
        if is_training and self.use_vector_obs and trainer_parameters['normalize']:
            self.inference_run_list.extend([self.model.update_mean, self.model.update_variance])

    def act(self, curr_brain_info):
        feed_dict = {self.model.batch_size: len(curr_brain_info.vector_observations),
                     self.model.sequence_length: 1}
        if self.use_recurrent:
            if not self.is_continuous_action:
                feed_dict[self.model.prev_action] = curr_brain_info.previous_vector_actions.reshape(
                    [-1, len(self.model.a_size)])
            if curr_brain_info.memories.shape[1] == 0:
                curr_brain_info.memories = np.zeros((len(curr_brain_info.agents), self.m_size))
            feed_dict[self.model.memory_in] = curr_brain_info.memories
        if self.use_visual_obs:
            for i, _ in enumerate(curr_brain_info.visual_observations):
                feed_dict[self.model.visual_in[i]] = curr_brain_info.visual_observations[i]
        if self.use_vector_obs:
            feed_dict[self.model.vector_in] = curr_brain_info.vector_observations

        values = self.sess.run(self.inference_run_list, feed_dict=feed_dict)
        run_out = dict(zip(self.inference_run_list, values))
        return run_out

    def update(self, buffer, n_sequences, i):
        start = i * n_sequences
        end = (i + 1) * n_sequences
        feed_dict = {self.model.batch_size: n_sequences,
                     self.model.sequence_length: self.sequence_length,
                     self.model.mask_input: np.array(buffer['masks'][start:end]).flatten(),
                     self.model.returns_holder: np.array(buffer['discounted_returns'][start:end]).flatten(),
                     self.model.old_value: np.array(buffer['value_estimates'][start:end]).flatten(),
                     self.model.advantage: np.array(buffer['advantages'][start:end]).reshape([-1, 1]),
                     self.model.all_old_log_probs: np.array(buffer['action_probs'][start:end]).reshape(
                         [-1, sum(self.model.a_size)])}
        if self.is_continuous_action:
            feed_dict[self.model.output_pre] = np.array(buffer['actions_pre'][start:end]).reshape(
                [-1, self.model.a_size[0]])
        else:
            feed_dict[self.model.action_holder] = np.array(buffer['actions'][start:end]).reshape(
                [-1, len(self.model.a_size)])
            if self.use_recurrent:
                feed_dict[self.model.prev_action] = np.array(buffer['prev_action'][start:end]).reshape(
                    [-1, len(self.model.a_size)])
        if self.use_vector_obs:
            total_observation_length = self.model.o_size
            feed_dict[self.model.vector_in] = np.array(buffer['vector_obs'][start:end]).reshape(
                [-1, total_observation_length])
            if self.use_curiosity:
                feed_dict[self.model.next_vector_in] = np.array(buffer['next_vector_in'][start:end]) \
                    .reshape([-1, total_observation_length])
        if self.use_visual_obs:
            for i, _ in enumerate(self.model.visual_in):
                _obs = np.array(buffer['visual_obs%d' % i][start:end])
                if self.sequence_length > 1 and self.use_recurrent:
                    (_batch, _seq, _w, _h, _c) = _obs.shape
                    feed_dict[self.model.visual_in[i]] = _obs.reshape([-1, _w, _h, _c])
                else:
                    feed_dict[self.model.visual_in[i]] = _obs
            if self.use_curiosity:
                for i, _ in enumerate(self.model.visual_in):
                    _obs = np.array(buffer['next_visual_obs%d' % i][start:end])
                    if self.sequence_length > 1 and self.use_recurrent:
                        (_batch, _seq, _w, _h, _c) = _obs.shape
                        feed_dict[self.model.next_visual_in[i]] = _obs.reshape([-1, _w, _h, _c])
                    else:
                        feed_dict[self.model.next_visual_in[i]] = _obs
        if self.use_recurrent:
            mem_in = np.array(buffer['memory'][start:end])[:, 0, :]
            feed_dict[self.model.memory_in] = mem_in
        self.has_updated = True
        run_list = [self.model.value_loss, self.model.policy_loss, self.model.update_batch]
        if self.use_curiosity:
            run_list.extend([self.model.forward_loss, self.model.inverse_loss])
        values = self.sess.run(run_list, feed_dict=feed_dict)
        run_out = dict(zip(run_list, values))
        return run_out

    def get_intrinsic_rewards(self, curr_info, next_info):
        """
        Generates intrinsic reward used for Curiosity-based training.
        :BrainInfo curr_info: Current BrainInfo.
        :BrainInfo next_info: Next BrainInfo.
        :return: Intrinsic rewards for all agents.
        """
        if self.use_curiosity:
            feed_dict = {self.model.batch_size: len(next_info.vector_observations), self.model.sequence_length: 1}
            if self.is_continuous_action:
                feed_dict[self.model.output] = next_info.previous_vector_actions
            else:
                feed_dict[self.model.action_holder] = next_info.previous_vector_actions

            if self.use_visual_obs:
                for i in range(len(curr_info.visual_observations)):
                    feed_dict[self.model.visual_in[i]] = curr_info.visual_observations[i]
                    feed_dict[self.model.next_visual_in[i]] = next_info.visual_observations[i]
            if self.use_vector_obs:
                feed_dict[self.model.vector_in] = curr_info.vector_observations
                feed_dict[self.model.next_vector_in] = next_info.vector_observations
            if self.use_recurrent:
                if curr_info.memories.shape[1] == 0:
                    curr_info.memories = np.zeros((len(curr_info.agents), self.m_size))
                feed_dict[self.model.memory_in] = curr_info.memories
            intrinsic_rewards = self.sess.run(self.model.intrinsic_reward,
                                              feed_dict=feed_dict) * float(self.has_updated)
            return intrinsic_rewards
        else:
            return None

    def get_value_estimate(self, brain_info, idx):
        """
        Generates value estimates for bootstrapping.
        :param brain_info: BrainInfo to be used for bootstrapping.
        :param idx: Index in BrainInfo of agent.
        :return: Value estimate.
        """
        feed_dict = {self.model.batch_size: 1, self.model.sequence_length: 1}
        if self.use_visual_obs:
            for i in range(len(brain_info.visual_observations)):
                feed_dict[self.model.visual_in[i]] = [brain_info.visual_observations[i][idx]]
        if self.use_vector_obs:
            feed_dict[self.model.vector_in] = [brain_info.vector_observations[idx]]
        if self.use_recurrent:
            if brain_info.memories.shape[1] == 0:
                brain_info.memories = np.zeros(
                    (len(brain_info.vector_observations), self.m_size))
            feed_dict[self.model.memory_in] = [brain_info.memories[idx]]
        if not self.is_continuous_action and self.use_recurrent:
            feed_dict[self.model.prev_action] = brain_info.previous_vector_actions[idx].reshape(
                [-1, len(self.model.a_size)])
        value_estimate = self.sess.run(self.model.value, feed_dict)
        return value_estimate

    @property
    def graph_scope(self):
        """
        Returns the graph scope of the trainer.
        """
        return self.variable_scope

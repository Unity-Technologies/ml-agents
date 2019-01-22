import numpy as np
from mlagents.trainers.ppo.reward_signals import RewardSignal
from mlagents.trainers.ppo.reward_signals.curiosity.model import CuriosityModel
from mlagents.trainers.policy import Policy


class CuriositySignal(RewardSignal):
    def __init__(self, policy: Policy, encoding_size, signal_strength):
        """
        Creates the Curiosity reward generator
        :param policy: The Learning Policy
        :param encoding_size: The size of the Curiosity encoding
        :param signal_strength: The scaling parameter for the reward. The scaled reward will be the unscaled
        reward multiplied by the strength parameter
        """
        self.policy = policy
        self.strength = signal_strength
        self.stat_name = 'Policy/Curiosity Reward'
        self.value_name = 'Policy/Curiosity Value Estimate'
        self.model = CuriosityModel(policy.model, encoding_size=encoding_size)
        self.update_dict = {'forward_loss': self.model.forward_loss,
                            'inverse_loss': self.model.inverse_loss,
                            'update': self.model.update_batch}
        self.has_updated = False

    def evaluate(self, current_info, next_info):
        if len(current_info.agents) == 0:
            return []

        feed_dict = {self.policy.model.batch_size: len(next_info.vector_observations),
                     self.policy.model.sequence_length: 1}
        feed_dict = self.policy.fill_eval_dict(feed_dict, brain_info=current_info)
        if self.policy.use_continuous_act:
            feed_dict[self.policy.model.selected_actions] = next_info.previous_vector_actions
        else:
            feed_dict[self.policy.model.action_holder] = next_info.previous_vector_actions
        for i in range(self.policy.model.vis_obs_size):
            feed_dict[self.model.next_visual_in[i]] = next_info.visual_observations[i]
        if self.policy.use_vec_obs:
            feed_dict[self.model.next_vector_in] = next_info.vector_observations
        if self.policy.use_recurrent:
            if current_info.memories.shape[1] == 0:
                current_info.memories = self.policy.make_empty_memory(len(current_info.agents))
            feed_dict[self.policy.model.memory_in] = current_info.memories
        unscaled_reward = self.policy.sess.run(self.model.intrinsic_reward,
                                               feed_dict=feed_dict)
        scaled_reward = np.clip(
            unscaled_reward * float(self.has_updated) * self.strength, 0, 1)
        return scaled_reward, unscaled_reward

    def update(self, mini_batch, num_sequences):
        """
        Updates model using buffer.
        :param num_sequences: Number of trajectories in batch.
        :param mini_batch: Experience batch.
        :return: Output from update process.
        """
        feed_dict = {self.policy.model.batch_size: num_sequences,
                     self.policy.model.sequence_length: self.policy.sequence_length,
                     self.policy.model.mask_input: mini_batch['masks'].flatten(),
                     self.policy.model.advantage: mini_batch['advantages'].reshape([-1, 1]),
                     self.policy.model.all_old_log_probs: mini_batch['action_probs'].reshape(
                         [-1, sum(self.policy.model.act_size)])}
        if self.policy.use_continuous_act:
            feed_dict[self.policy.model.output_pre] = mini_batch['actions_pre'].reshape(
                [-1, self.policy.model.act_size[0]])
            feed_dict[self.policy.model.epsilon] = mini_batch['random_normal_epsilon'].reshape(
                [-1, self.policy.model.act_size[0]])
        else:
            feed_dict[self.policy.model.action_holder] = mini_batch['actions'].reshape(
                [-1, len(self.policy.model.act_size)])
            if self.policy.use_recurrent:
                feed_dict[self.policy.model.prev_action] = mini_batch['prev_action'].reshape(
                    [-1, len(self.policy.model.act_size)])
            feed_dict[self.policy.model.action_masks] = mini_batch['action_mask'].reshape(
                [-1, sum(self.policy.brain.vector_action_space_size)])
        if self.policy.use_vec_obs:
            feed_dict[self.policy.model.vector_in] = mini_batch['vector_obs'].reshape(
                [-1, self.policy.vec_obs_size])
            feed_dict[self.model.next_vector_in] = mini_batch['next_vector_in'].reshape(
                [-1, self.policy.vec_obs_size])
        if self.policy.model.vis_obs_size > 0:
            for i, _ in enumerate(self.policy.model.visual_in):
                _obs = mini_batch['visual_obs%d' % i]
                if self.policy.sequence_length > 1 and self.policy.use_recurrent:
                    (_batch, _seq, _w, _h, _c) = _obs.shape
                    feed_dict[self.policy.model.visual_in[i]] = _obs.reshape([-1, _w, _h, _c])
                else:
                    feed_dict[self.policy.model.visual_in[i]] = _obs
            for i, _ in enumerate(self.policy.model.visual_in):
                _obs = mini_batch['next_visual_obs%d' % i]
                if self.policy.sequence_length > 1 and self.policy.use_recurrent:
                    (_batch, _seq, _w, _h, _c) = _obs.shape
                    feed_dict[self.model.next_visual_in[i]] = _obs.reshape(
                        [-1, _w, _h, _c])
                else:
                    feed_dict[self.model.next_visual_in[i]] = _obs
        if self.policy.use_recurrent:
            mem_in = mini_batch['memory'][:, 0, :]
            feed_dict[self.policy.model.memory_in] = mem_in
        self.has_updated = True
        run_out = self.policy._execute_model(feed_dict, self.update_dict)
        return run_out

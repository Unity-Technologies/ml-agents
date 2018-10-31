import numpy as np

from mlagents.trainers.ppo.reward_signals import RewardSignal
from .model import GAILModel
from mlagents.trainers.demo_loader import demo_to_buffer


class GAILSignal(RewardSignal):
    def __init__(self, policy, h_size, lr, demo_path, signal_strength):
        super().__init__()
        self.policy = policy
        self.strength = signal_strength
        self.stat_name = 'Policy/GAIL Reward'
        self.model = GAILModel(policy.model, h_size, lr, 64)
        _, self.demonstration_buffer = demo_to_buffer(demo_path, 1)

    def evaluate(self, current_info, next_info):
        if len(current_info.agents) == 0:
            return []

        feed_dict = {self.policy.model.batch_size: len(next_info.vector_observations),
                     self.policy.model.sequence_length: 1}
        feed_dict = self.policy.fill_eval_dict(feed_dict, brain_info=current_info)
        feed_dict[self.model.done_policy] = np.reshape(next_info.local_done, [-1, 1])
        if self.policy.use_continuous_act:
            feed_dict[self.policy.model.selected_actions] = next_info.previous_vector_actions
        else:
            feed_dict[self.policy.model.action_holder] = next_info.previous_vector_actions
        if self.policy.use_recurrent:
            if current_info.memories.shape[1] == 0:
                current_info.memories = self.policy.make_empty_memory(len(current_info.agents))
            feed_dict[self.policy.model.memory_in] = current_info.memories
        unscaled_reward = self.policy.sess.run(self.model.intrinsic_reward,
                                               feed_dict=feed_dict)
        scaled_reward = unscaled_reward * float(self.policy.has_updated) * self.strength
        return scaled_reward, unscaled_reward

    def update(self, policy_buffer, n_sequences, max_batches):
        self.demonstration_buffer.update_buffer.shuffle()
        policy_buffer.update_buffer.shuffle()
        batch_losses = []
        possible_batches = len(self.demonstration_buffer.update_buffer['actions']) // n_sequences
        if max_batches == 0:
            num_batches = possible_batches
        else:
            num_batches = min(possible_batches, max_batches)
        for i in range(num_batches):
            demo_update_buffer = self.demonstration_buffer.update_buffer
            policy_update_buffer = policy_buffer.update_buffer
            start = i * n_sequences
            end = (i + 1) * n_sequences
            mini_batch_demo = demo_update_buffer.make_mini_batch(start, end)
            mini_batch_policy = policy_update_buffer.make_mini_batch(start, end)
            run_out = self._update_batch(mini_batch_demo, mini_batch_policy)
            loss = run_out['gail_loss']
            batch_losses.append(loss)
        return np.mean(batch_losses)

    def _update_batch(self, mini_batch_demo, mini_batch_policy):
        feed_dict = {}
        feed_dict[self.model.done_expert] = mini_batch_demo['done'].reshape([-1, 1])
        feed_dict[self.model.done_policy] = mini_batch_policy['done'].reshape([-1, 1])

        if self.policy.use_continuous_act:
            feed_dict[self.policy.model.selected_actions] = mini_batch_policy['actions'].reshape(
                [-1, self.policy.model.act_size[0]])
            feed_dict[self.model.action_in_expert] = mini_batch_demo['actions'].reshape(
                [-1, self.policy.model.act_size[0]])
        else:
            feed_dict[self.policy.model.action_holder] = mini_batch_policy['actions'].reshape(
                [-1, len(self.policy.model.act_size)])
            feed_dict[self.model.action_in_expert] = mini_batch_demo['actions'].reshape(
                [-1, len(self.policy.model.act_size)])

        if self.policy.use_vec_obs:
            feed_dict[self.policy.model.vector_in] = mini_batch_policy['vector_obs'].reshape(
                [-1, self.policy.vec_obs_size])
            feed_dict[self.model.obs_in_expert] = mini_batch_demo['vector_obs'].reshape(
                [-1, self.policy.vec_obs_size])
        loss, _ = self.policy.sess.run([self.model.loss, self.model.update_batch],
                                       feed_dict=feed_dict)
        run_out = {'gail_loss': loss}
        return run_out

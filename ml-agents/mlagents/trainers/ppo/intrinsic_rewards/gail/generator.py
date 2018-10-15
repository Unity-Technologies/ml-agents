import numpy as np

from mlagents.trainers.ppo.intrinsic_rewards.intrinsic_reward import IntrinsicReward
from .model import Discriminator
from mlagents.trainers.demo_loader import demo_to_buffer


class GAIL(IntrinsicReward):
    def __init__(self, policy, h_size, lr, demo_path):
        super().__init__()
        self.name = "GAIL"
        self.policy = policy
        self.discriminator = Discriminator(policy.model, h_size, lr)
        _, self.demonstration_buffer = demo_to_buffer(demo_path, 1)

    def evaluate(self, current_info, next_info):
        if len(current_info.agents) == 0:
            return []

        feed_dict = {self.policy.model.batch_size: len(next_info.vector_observations),
                     self.policy.model.sequence_length: 1}
        if self.policy.use_continuous_act:
            feed_dict[self.policy.model.selected_actions] = next_info.previous_vector_actions
        else:
            feed_dict[self.policy.model.action_holder] = next_info.previous_vector_actions
        for i in range(self.policy.model.vis_obs_size):
            feed_dict[self.policy.model.visual_in[i]] = current_info.visual_observations[i]
        if self.policy.use_vec_obs:
            feed_dict[self.policy.model.vector_in] = current_info.vector_observations
        if self.policy.use_recurrent:
            if current_info.memories.shape[1] == 0:
                current_info.memories = self.policy.make_empty_memory(len(current_info.agents))
            feed_dict[self.policy.model.memory_in] = current_info.memories
        raw_intrinsic_rewards = self.policy.sess.run(self.discriminator.intrinsic_reward,
                                                     feed_dict=feed_dict)
        intrinsic_rewards = raw_intrinsic_rewards * float(self.policy.has_updated)
        return intrinsic_rewards

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
        if self.policy.use_continuous_act:
            feed_dict[self.policy.model.selected_actions] = mini_batch_policy['actions'].reshape(
                [-1, self.policy.model.act_size[0]])
            feed_dict[self.discriminator.action_in_expert] = mini_batch_demo['actions'].reshape(
                [-1, self.policy.model.act_size[0]])
        else:
            feed_dict[self.policy.model.action_holder] = mini_batch_policy['actions'].reshape(
                [-1, len(self.policy.model.act_size)])
            feed_dict[self.discriminator.action_in_expert] = mini_batch_demo['actions'].reshape(
                [-1, len(self.policy.model.act_size)])

        if self.policy.use_vec_obs:
            feed_dict[self.policy.model.vector_in] = mini_batch_policy['vector_obs'].reshape(
                [-1, self.policy.vec_obs_size])
            feed_dict[self.discriminator.obs_in_expert] = mini_batch_demo['vector_obs'].reshape(
                [-1, self.policy.vec_obs_size])
        loss, _ = self.policy.sess.run([self.discriminator.loss, self.discriminator.update_batch],
                                       feed_dict=feed_dict)
        run_out = {'gail_loss': loss}
        return run_out

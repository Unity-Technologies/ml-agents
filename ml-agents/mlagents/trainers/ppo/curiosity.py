from .intrinsic_reward import IntrinsicReward
from .curiosity_model import ICM


class Curiosity(IntrinsicReward):
    def __init__(self, policy, encoding_size, strength):
        super().__init__()
        self.icm = ICM(policy.model, encoding_size=encoding_size, strength=strength)
        self.policy = policy

    def get_intrinsic_rewards(self, curr_info, next_info):
        """
        Generates intrinsic reward used for Curiosity-based training.
        :BrainInfo curr_info: Current BrainInfo.
        :BrainInfo next_info: Next BrainInfo.
        :return: Intrinsic rewards for all agents.
        """
        if len(curr_info.agents) == 0:
            return []

        feed_dict = {self.policy.model.batch_size: len(next_info.vector_observations),
                     self.policy.model.sequence_length: 1}
        if self.policy.use_continuous_act:
            feed_dict[self.policy.model.output] = next_info.previous_vector_actions
        else:
            feed_dict[self.policy.model.action_holder] = next_info.previous_vector_actions
        for i in range(self.policy.model.vis_obs_size):
            feed_dict[self.policy.model.visual_in[i]] = curr_info.visual_observations[i]
            feed_dict[self.icm.next_visual_in[i]] = next_info.visual_observations[i]
        if self.policy.use_vec_obs:
            feed_dict[self.policy.model.vector_in] = curr_info.vector_observations
            feed_dict[self.icm.next_vector_in] = next_info.vector_observations
        if self.policy.use_recurrent:
            if curr_info.memories.shape[1] == 0:
                curr_info.memories = self.policy.make_empty_memory(len(curr_info.agents))
            feed_dict[self.policy.model.memory_in] = curr_info.memories
        raw_intrinsic_rewards = self.policy.sess.run(self.icm.intrinsic_reward, feed_dict=feed_dict)
        intrinsic_rewards = raw_intrinsic_rewards * float(self.policy.has_updated)
        return intrinsic_rewards

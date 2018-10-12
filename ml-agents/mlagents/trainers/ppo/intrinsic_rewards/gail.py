from .intrinsic_reward import IntrinsicReward
from .gail_discriminator import Discriminator
from mlagents.trainers.demo_loader import demo_to_buffer


class GAIL(IntrinsicReward):
    def __init__(self, policy, h_size, lr, demo_path):
        super().__init__()
        self.name = "GAIL"
        self.policy = policy
        self.discriminator = Discriminator(policy.model, h_size, lr)
        self.expert_demos, _ = demo_to_buffer(demo_path, 1)

    def get_intrinsic_rewards(self, current_info, next_info):
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

    def update_generator(self, mini_batch):
        return None

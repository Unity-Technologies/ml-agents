import numpy as np

from mlagents.trainers.ppo.components import RewardSignal
from mlagents.trainers.policy import Policy
from .model import GAILModel
from mlagents.trainers.demo_loader import demo_to_buffer


class GAILSignal(RewardSignal):
    def __init__(self, policy: Policy, h_size, lr, demo_path, signal_strength):
        """
        The Gail Reward signal generator.
        :param policy: The policy of the learning model
        :param h_size: The size of the the hidden layers of the discriminator
        :param lr: The Learning Rate
        :param demo_path: The path to the demonstration file
        :param signal_strength: The scaling parameter for the reward. The scaled reward will be the unscaled
        reward multiplied by the strength parameter
        """
        super().__init__()
        self.policy = policy
        self.strength = signal_strength
        self.stat_name = 'Policy/GAIL Reward'
        self.value_name = 'Policy/GAIL Value Estimate'
        self.model = GAILModel(policy.model, h_size, lr, 64)
        _, self.demonstration_buffer = demo_to_buffer(demo_path, policy.sequence_length)
        self.has_updated = False

    def evaluate(self, current_info, next_info):
        if len(current_info.agents) == 0:
            return []

        feed_dict = {self.policy.model.batch_size: len(next_info.vector_observations),
                     self.policy.model.sequence_length: 1,
                     self.model.use_noise: [0]}
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
        scaled_reward = unscaled_reward * float(self.has_updated) * self.strength
        return scaled_reward, unscaled_reward

    def update(self, policy_buffer, n_sequences=32, max_batches=100):
        """
        Updates model using buffer.
        :param policy_buffer: The policy buffer containing the trajectories for the current policy.
        :param n_sequences: The number of sequences used in each mini batch.
        :param max_batches: The maximum number of batches to use per update.
        :return: The loss of the update.
        """
        batch_losses = []
        n_sequences = n_sequences // 2
        possible_demo_batches = len(
            self.demonstration_buffer.update_buffer['actions']) // n_sequences
        possible_policy_batches = len(policy_buffer.update_buffer['actions']) // n_sequences
        possible_batches = min(possible_policy_batches, possible_demo_batches)

        # for reporting
        kl_loss = []
        pos=[]
        pes=[]
        zlss = []
        zme=[]
        zmp = []
        # end for reporting
        n_epoch = 3
        for epoch in range(n_epoch):
            self.demonstration_buffer.update_buffer.shuffle()
            policy_buffer.update_buffer.shuffle()
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
                # for reporting
                kl_loss.append(run_out['kl'])
                pos.append(run_out['po'])
                pes.append(run_out['pe'])
                zlss.append(run_out['zlss'])
                zmp.append(run_out['zmp'])
                zme.append(run_out['zme'])
                # end for reporting
                batch_losses.append(loss)
        self.has_updated = True

        # for reporting

        print('n_epoch','beta', 'kl_loss', 'policy_estimate', 'expert_estimate', 'z_mean_expert', 'z_mean_policy', 'z_log_sig_sq')
        print(n_epoch, self.policy.sess.run(self.model.beta), np.mean(kl_loss), np.mean(pos), np.mean(pes), np.mean(zme), np.mean(zmp), np.mean(zlss))
        # end for reporting
        return np.mean(batch_losses)

    def _update_batch(self, mini_batch_demo, mini_batch_policy):
        """
        Helper method for update.
        :param mini_batch_demo: A mini batch of expert trajectories
        :param mini_batch_policy: A mini batch of trajectories sampled from the current policy
        :return: Output from update process.
        """
        feed_dict = {self.model.done_expert: mini_batch_demo['done'].reshape([-1, 1]),
                     self.model.done_policy: mini_batch_policy['done'].reshape([-1, 1]),
                     self.model.use_noise: [1.0]}

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

        if self.policy.use_vis_obs > 0:
            for i in range(len(self.policy.model.visual_in)):
                policy_obs = mini_batch_policy['visual_obs%d' % i]
                if self.policy.sequence_length > 1 and self.policy.use_recurrent:
                    (_batch, _seq, _w, _h, _c) = policy_obs.shape
                    feed_dict[self.policy.model.visual_in[i]] = policy_obs.reshape([-1, _w, _h, _c])
                else:
                    feed_dict[self.policy.model.visual_in[i]] = policy_obs

                demo_obs = mini_batch_demo['visual_obs%d' % i]
                if self.policy.sequence_length > 1 and self.policy.use_recurrent:
                    (_batch, _seq, _w, _h, _c) = demo_obs.shape
                    feed_dict[self.model.expert_visual_in[i]] = demo_obs.reshape(
                        [-1, _w, _h, _c])
                else:
                    feed_dict[self.model.expert_visual_in[i]] = demo_obs
        if self.policy.use_vec_obs:
            feed_dict[self.policy.model.vector_in] = mini_batch_policy['vector_obs'].reshape(
                [-1, self.policy.vec_obs_size])
            feed_dict[self.model.obs_in_expert] = mini_batch_demo['vector_obs'].reshape(
                [-1, self.policy.vec_obs_size])

        # for reporting
        po, pe, zlss, zme, zmp = None, None, None, None, None
        kl_loss = None
        # end for reporting
        if self.model.use_vail:
            loss, _, kl_loss, po, pe, zlss, zme, zmp = self.policy.sess.run([self.model.loss, self.model.update_batch,
                                                     self.model.kl_loss
                                                        , self.model.policy_estimate, self.model.expert_estimate,
                                                             self.model.z_log_sigma_sq, self.model.z_mean_expert, self.model.z_mean_policy],
                                                    feed_dict=feed_dict)
            self.update_beta(kl_loss)
        else:
            loss, _, po, pe = self.policy.sess.run([self.model.loss, self.model.update_batch
                                            , self.model.policy_estimate, self.model.expert_estimate],
                                           feed_dict=feed_dict)
        run_out = {'gail_loss': loss, 'po': po,'pe': pe, 'kl':kl_loss, "zlss":zlss, "zme":zme, 'zmp':zmp}
        return run_out

    def update_beta(self, kl_div):
        """
        Updates the Beta parameter with the latest kl_divergence value.
        The larger Beta, the stronger the importance of the kl divergence in the loss function.
        :param kl_div: The KL divergence
        """
        self.policy.sess.run(self.model.update_beta, feed_dict={self.model.kl_div_input: kl_div})


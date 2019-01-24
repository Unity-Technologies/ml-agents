import logging
import numpy as np

from mlagents.trainers.ppo.models import PPOModel
from mlagents.trainers.policy import Policy
from mlagents.trainers.ppo.reward_signals.gail import GAILSignal
from mlagents.trainers.ppo.reward_signals.curiosity import CuriositySignal
from mlagents.trainers.ppo.reward_signals.extrinsic import ExtrinsicSignal
from mlagents.trainers.ppo.reward_signals.entropy import EntropySignal
from mlagents.trainers.ppo.pre_training import PreTraining


logger = logging.getLogger("mlagents.trainers")


class PPOPolicy(Policy):
    def __init__(self, seed, brain, trainer_params, is_training, load):
        """
        Policy for Proximal Policy Optimization Networks.
        :param seed: Random seed.
        :param brain: Assigned Brain object.
        :param trainer_params: Defined training parameters.
        :param is_training: Whether the model should be trained.
        :param load: Whether a pre-trained model will be loaded or a new one created.
        """
        super().__init__(seed, brain, trainer_params)

        reward_strengths = dict(zip(trainer_params['reward_signals'],
                                    trainer_params['reward_strength']))
        self.reward_signals = {}
        with self.graph.as_default():
            self.model = PPOModel(brain,
                                  lr=float(trainer_params['learning_rate']),
                                  h_size=int(trainer_params['hidden_units']),
                                  epsilon=float(trainer_params['epsilon']),
                                  beta=float(trainer_params['beta']),
                                  max_step=float(trainer_params['max_steps']),
                                  normalize=trainer_params['normalize'],
                                  use_recurrent=trainer_params['use_recurrent'],
                                  num_layers=int(trainer_params['num_layers']),
                                  m_size=self.m_size,
                                  seed=seed,
                                  stream_names=list(reward_strengths.keys()))
            self.model.create_ppo_optimizer()

            if 'extrinsic' in reward_strengths.keys():
                self.reward_signals['extrinsic'] = ExtrinsicSignal(reward_strengths['extrinsic'])
            if 'curiosity' in reward_strengths.keys():
                encoding_size = float(trainer_params['curiosity_enc_size'])
                curiosity_signal = CuriositySignal(policy=self,
                                                   signal_strength=reward_strengths['curiosity'],
                                                   encoding_size=encoding_size)
                self.reward_signals['curiosity'] = curiosity_signal
            if 'gail' in reward_strengths.keys():
                gail_signal = GAILSignal(self, int(trainer_params['hidden_units']),
                                         float(trainer_params['learning_rate']),
                                         trainer_params['demo_path'], reward_strengths['gail'])
                self.reward_signals['gail'] = gail_signal
            if 'entropy' in reward_strengths.keys():
                self.reward_signals['entropy'] = EntropySignal(self, reward_strengths['entropy'])

        pre_trainer = None
        if load:
            self._load_graph()
        else:
            if trainer_params['pre_training']:
                with self.graph.as_default():
                    pre_trainer = PreTraining(self.sess,
                                              self.model,
                                              self.brain,
                                              trainer_params)
            self._initialize_graph()
        if pre_trainer:
                pre_trainer.update_policy()

        self.inference_dict = {
            "action": self.model.output,
            "log_probs": self.model.all_log_probs,
            "value": self.model.value,
            "entropy": self.model.entropy,
            "learning_rate": self.model.learning_rate,
        }
        if self.use_continuous_act:
            self.inference_dict["pre_action"] = self.model.output_pre
        if self.use_recurrent:
            self.inference_dict['memory_out'] = self.model.memory_out
        if is_training and self.use_vec_obs and trainer_params['normalize'] and pre_trainer is None and not load:
            self.inference_dict['update_mean'] = self.model.update_normalization

        self.update_dict = {'value_loss': self.model.value_loss,
                            'policy_loss': self.model.policy_loss,
                            'update_batch': self.model.update_batch}

    def evaluate(self, brain_info):
        """
        Evaluates policy for the agent experiences provided.
        :param brain_info: BrainInfo object containing inputs.
        :return: Outputs from network as defined by self.inference_dict.
        """
        feed_dict = {
            self.model.batch_size: len(brain_info.vector_observations),
            self.model.sequence_length: 1,
        }
        epsilon = None
        if self.use_recurrent:
            if not self.use_continuous_act:
                feed_dict[
                    self.model.prev_action
                ] = brain_info.previous_vector_actions.reshape(
                    [-1, len(self.model.act_size)]
                )
            if brain_info.memories.shape[1] == 0:
                brain_info.memories = self.make_empty_memory(len(brain_info.agents))
            feed_dict[self.model.memory_in] = brain_info.memories
        if self.use_continuous_act:
            epsilon = np.random.normal(
                size=(len(brain_info.vector_observations), self.model.act_size[0])
            )
            feed_dict[self.model.epsilon] = epsilon
        feed_dict = self.fill_eval_dict(feed_dict, brain_info)
        run_out = self._execute_model(feed_dict, self.inference_dict)
        if self.use_continuous_act:
            run_out["random_normal_epsilon"] = epsilon
        return run_out

    def update(self, mini_batch, num_sequences):
        """
        Updates model using buffer.
        :param num_sequences: Number of trajectories in batch.
        :param mini_batch: Experience batch.
        :return: Output from update process.
        """
        feed_dict = {self.model.batch_size: num_sequences,
                     self.model.sequence_length: self.sequence_length,
                     self.model.mask_input: mini_batch['masks'].flatten(),
                     self.model.advantage: mini_batch['advantages'].reshape([-1, 1]),
                     self.model.all_old_log_probs: mini_batch['action_probs'].reshape(
                         [-1, sum(self.model.act_size)])}
        for i, name in enumerate(self.reward_signals.keys()):
            feed_dict[self.model.returns_holders[i]] = mini_batch['{}_returns'.format(name)].flatten()
            feed_dict[self.model.old_values[i]] = mini_batch['{}_value_estimates'.format(name)].flatten()

        if self.use_continuous_act:
            feed_dict[self.model.output_pre] = mini_batch["actions_pre"].reshape(
                [-1, self.model.act_size[0]]
            )
            feed_dict[self.model.epsilon] = mini_batch["random_normal_epsilon"].reshape(
                [-1, self.model.act_size[0]]
            )
        else:
            feed_dict[self.model.action_holder] = mini_batch["actions"].reshape(
                [-1, len(self.model.act_size)]
            )
            if self.use_recurrent:
                feed_dict[self.model.prev_action] = mini_batch["prev_action"].reshape(
                    [-1, len(self.model.act_size)]
                )
            feed_dict[self.model.action_masks] = mini_batch["action_mask"].reshape(
                [-1, sum(self.brain.vector_action_space_size)]
            )
        if self.use_vec_obs:
            feed_dict[self.model.vector_in] = mini_batch['vector_obs'].reshape(
                [-1, self.vec_obs_size])
        if self.model.vis_obs_size > 0:
            for i, _ in enumerate(self.model.visual_in):
                _obs = mini_batch["visual_obs%d" % i]
                if self.sequence_length > 1 and self.use_recurrent:
                    (_batch, _seq, _w, _h, _c) = _obs.shape
                    feed_dict[self.model.visual_in[i]] = _obs.reshape([-1, _w, _h, _c])
                else:
                    feed_dict[self.model.visual_in[i]] = _obs
        if self.use_recurrent:
            mem_in = mini_batch["memory"][:, 0, :]
            feed_dict[self.model.memory_in] = mem_in
        run_out = self._execute_model(feed_dict, self.update_dict)
        return run_out

    def get_value_estimates(self, brain_info, idx):
        """
        Generates value estimates for bootstrapping.
        :param brain_info: BrainInfo to be used for bootstrapping.
        :param idx: Index in BrainInfo of agent.
        :return: The value estimate dictionary with key being the name of the reward signal and the value the
        corresponding value estimate.
        """
        feed_dict = {self.model.batch_size: 1, self.model.sequence_length: 1}
        for i in range(len(brain_info.visual_observations)):
            feed_dict[self.model.visual_in[i]] = [
                brain_info.visual_observations[i][idx]
            ]
        if self.use_vec_obs:
            feed_dict[self.model.vector_in] = [brain_info.vector_observations[idx]]
        if self.use_recurrent:
            if brain_info.memories.shape[1] == 0:
                brain_info.memories = self.make_empty_memory(len(brain_info.agents))
            feed_dict[self.model.memory_in] = [brain_info.memories[idx]]
        if not self.use_continuous_act and self.use_recurrent:
            feed_dict[self.model.prev_action] = brain_info.previous_vector_actions[
                idx
            ].reshape([-1, len(self.model.act_size)])
        value_estimate = self.sess.run(self.model.value, feed_dict)
        return value_estimate

    def get_last_reward(self):
        """
        Returns the last reward the trainer has had
        :return: the new last reward
        """
        return self.sess.run(self.model.last_reward)

    def update_reward(self, new_reward):
        """
        Updates reward value for policy.
        :param new_reward: New reward to save.
        """
        self.sess.run(
            self.model.update_reward, feed_dict={self.model.new_reward: new_reward}
        )

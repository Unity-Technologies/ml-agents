import logging
import numpy as np
import tensorflow as tf

from mlagents.trainers import BrainInfo, ActionInfo
from mlagents.trainers.sac.models import SACModel
from mlagents.trainers.tf_policy import TFPolicy
from mlagents.trainers.components.reward_signals.reward_signal_factory import (
    create_reward_signal,
)
from mlagents.trainers.components.bc import BCModule

logger = logging.getLogger("mlagents.trainers")


class SACPolicy(TFPolicy):
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

        reward_signal_configs = {}
        for key, rsignal in trainer_params["reward_signals"].items():
            if type(rsignal) is dict:
                reward_signal_configs[key] = rsignal

        self.reward_signals = {}
        with self.graph.as_default():
            self.model = SACModel(
                brain,
                lr=float(trainer_params["learning_rate"]),
                h_size=int(trainer_params["hidden_units"]),
                init_entcoef=float(trainer_params["init_entcoef"]),
                max_step=float(trainer_params["max_steps"]),
                normalize=trainer_params["normalize"],
                use_recurrent=trainer_params["use_recurrent"],
                num_layers=int(trainer_params["num_layers"]),
                m_size=self.m_size,
                seed=seed,
                stream_names=list(reward_signal_configs.keys()),
                tau=float(trainer_params["tau"]),
                gammas=list(_val["gamma"] for _val in reward_signal_configs.values()),
                vis_encode_type=trainer_params["vis_encode_type"],
            )
            self.model.create_sac_optimizers()

            # Initialize Components
            for key, rsignal in reward_signal_configs.items():
                if type(rsignal) is dict:
                    self.reward_signals[key] = create_reward_signal(self, key, rsignal)

            # BC trainer is not a reward signal
            # if "demo_aided" in trainer_params:
            #     self.bc_trainer = BCTrainer(
            #         self,
            #         float(
            #             trainer_params["demo_aided"]["demo_strength"]
            #             * trainer_params["learning_rate"]
            #         ),
            #         trainer_params["demo_aided"]["demo_path"],
            #         trainer_params["demo_aided"]["demo_steps"],
            #         trainer_params["batch_size"],
            #     )

        if load:
            self._load_graph()
        else:
            self._initialize_graph()
            self.sess.run(self.model.target_init_op)

        self.inference_dict = {
            "action": self.model.output,
            "log_probs": self.model.all_log_probs,
            "value": self.model.value_heads,
            "entropy": self.model.entropy,
            "learning_rate": self.model.learning_rate,
        }
        if self.use_continuous_act:
            self.inference_dict["pre_action"] = self.model.output_pre
        if self.use_recurrent:
            self.inference_dict["memory_out"] = self.model.memory_out
        if (
            is_training
            and self.use_vec_obs
            and trainer_params["normalize"]
            and not load
        ):
            self.inference_dict["update_mean"] = self.model.update_normalization

        self.update_dict = {
            "value_loss": self.model.total_value_loss,
            "policy_loss": self.model.policy_loss,
            "q1_loss": self.model.q1_loss,
            "q2_loss": self.model.q2_loss,
            "entropy_coef": self.model.ent_coef,
            "entropy": self.model.entropy,
            "update_batch": self.model.update_batch_policy,
            "update_value": self.model.update_batch_value,
            "update_entropy": self.model.update_batch_entropy,
        }

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
        # epsilon = None
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
        # if self.use_continuous_act:
        #     epsilon = np.random.normal(
        #         size=(len(brain_info.vector_observations), self.model.act_size[0])
        #     )

        feed_dict = self.fill_eval_dict(feed_dict, brain_info)
        run_out = self._execute_model(feed_dict, self.inference_dict)
        return run_out

    def update(self, mini_batch, num_sequences, update_target=True):
        """
        Updates model using buffer.
        :param num_sequences: Number of trajectories in batch.
        :param mini_batch: Experience batch.
        :return: Output from update process.
        """
        feed_dict = {
            self.model.batch_size: num_sequences,
            self.model.sequence_length: self.sequence_length,
            self.model.next_sequence_length: self.sequence_length,
            self.model.mask_input: mini_batch["masks"].flatten(),
        }
        for i, name in enumerate(self.reward_signals.keys()):
            feed_dict[self.model.rewards_holders[name]] = mini_batch[
                "{}_rewards".format(name)
            ].flatten()

        if self.use_continuous_act:
            feed_dict[self.model.action_holder] = mini_batch["actions"].reshape(
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
            feed_dict[self.model.vector_in] = mini_batch["vector_obs"].reshape(
                [-1, self.vec_obs_size]
            )
            feed_dict[self.model.next_vector_in] = mini_batch["next_vector_in"].reshape(
                [-1, self.vec_obs_size]
            )
        if self.model.vis_obs_size > 0:
            for i, _ in enumerate(self.model.visual_in):
                _obs = mini_batch["visual_obs%d" % i]
                if self.sequence_length > 1 and self.use_recurrent:
                    (_batch, _seq, _w, _h, _c) = _obs.shape
                    feed_dict[self.model.visual_in[i]] = _obs.reshape([-1, _w, _h, _c])
                else:
                    feed_dict[self.model.visual_in[i]] = _obs

            for i, _ in enumerate(self.model.next_visual_in):
                _obs = mini_batch["next_visual_obs%d" % i]
                if self.sequence_length > 1 and self.use_recurrent:
                    (_batch, _seq, _w, _h, _c) = _obs.shape
                    feed_dict[self.model.next_visual_in[i]] = _obs.reshape(
                        [-1, _w, _h, _c]
                    )
                else:
                    feed_dict[self.model.next_visual_in[i]] = _obs
        if self.use_recurrent:
            mem_in = mini_batch["memory"][:, 0, :]
            next_mem_in = (
                mini_batch["memory"][:, 1, :]
                if self.sequence_length > 1
                else mini_batch["memory"][:, 0, :]
            )
            feed_dict[self.model.memory_in] = mem_in
            feed_dict[self.model.next_memory_in] = next_mem_in[:, : self.m_size // 4]
        feed_dict[self.model.dones_holder] = mini_batch["done"].flatten()
        run_out = self._execute_model(feed_dict, self.update_dict)
        # for key in feed_dict.keys():
        #     print(np.isnan(feed_dict[key]).any())
        #     print(key)
        if update_target:
            self.sess.run(self.model.target_update_op)
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

    def get_action(self, brain_info: BrainInfo) -> ActionInfo:
        """
        Decides actions given observations information, and takes them in environment.
        :param brain_info: A dictionary of brain names and BrainInfo from environment.
        :return: an ActionInfo containing action, memories, values and an object
        to be passed to add experiences
        """
        if len(brain_info.agents) == 0:
            return ActionInfo([], [], [], None, None)

        run_out = self.evaluate(brain_info)
        mean_values = np.mean(
            np.array(list(run_out.get("value").values())), axis=0
        ).flatten()

        return ActionInfo(
            action=run_out.get("action"),
            memory=run_out.get("memory_out"),
            text=None,
            value=mean_values,
            outputs=run_out,
        )

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

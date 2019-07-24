import logging
import numpy as np

from mlagents.envs.timers import timed
from mlagents.trainers.ppo.policy import PPOPolicy
from mlagents.trainers.components.reward_signals.reward_signal_factory import (
    create_reward_signal,
)
from mlagents.trainers.components.bc.module import BCModule

logger = logging.getLogger("mlagents.trainers")


class MultiGPUPPOPolicy(PPOPolicy):
    def __init__(self, seed, brain, trainer_params, is_training, load):
        reward_signal_configs = trainer_params["reward_signals"]
        self.reward_signals = {}

        self.devices = get_devices()
        self.towers = []
        
        with self.graph.as_default():
            for device in devices:
                with tf.device(device):
                    with tf.variable_scope(TOWER_SCOPE_NAME, reuse=tf.AUTO_REUSE):
                        self.towers.append(PPOModel(
                                brain=brain,
                                lr=float(trainer_params["learning_rate"]),
                                h_size=int(trainer_params["hidden_units"]),
                                epsilon=float(trainer_params["epsilon"]),
                                beta=float(trainer_params["beta"]),
                                max_step=float(trainer_params["max_steps"]),
                                normalize=trainer_params["normalize"],
                                use_recurrent=trainer_params["use_recurrent"],
                                num_layers=int(trainer_params["num_layers"]),
                                m_size=self.m_size,
                                seed=seed,
                                stream_names=list(reward_signal_configs.keys()),
                                vis_encode_type=trainer_params["vis_encode_type"],
                            )
                        )
                        self.towers[-1].create_ppo_optimizer()

            # Create reward signals
            for reward_signal, config in reward_signal_configs.items():
                self.reward_signals[reward_signal] = create_reward_signal(
                    self, reward_signal, config
                )

            # Create pretrainer if needed
            if "pretraining" in trainer_params:
                BCModule.check_config(trainer_params["pretraining"])
                self.bc_module = BCModule(
                    self,
                    policy_learning_rate=trainer_params["learning_rate"],
                    default_batch_size=trainer_params["batch_size"],
                    default_num_epoch=trainer_params["num_epoch"],
                    **trainer_params["pretraining"],
                )
            else:
                self.bc_module = None

        self.model = self.towers[0]

        if load:
            self._load_graph()
        else:
            self._initialize_graph()

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

        self.update_batch = self.average_gradients([t.grads for t in self.towers])
        self.update_dict = {"update_batch": self.update_batch}
        self.update_dict.update({"value_loss_%d".format(i):self.towers[i].value_loss for i in range(len(self.towers)) })
        self.update_dict.update({"policy_loss_%d".format(i):self.towers[i].policy_loss for i in range(len(self.towers)) })


    @timed
    def update(self, mini_batch, n_sequences):
        """
        Updates model using buffer.
        :param n_sequences: Number of trajectories in batch.
        :param mini_batch: Experience batch.
        :return: Output from update process.
        """
        feed_dict = {}
        towers = self.model.towers if len(self.devices) > 1 else [self.model]

        device_batch_size = n_sequences // len(self.devices)
        device_batches = []
        for i in range(len(self.devices)):
            device_batches.append(
                {k: v[i : i + device_batch_size] for k, v in mini_batch.items()}
            )

        assert len(device_batches) == len(towers)
        for batch, tower in zip(device_batches, towers):
            feed_dict = {
                self.model.batch_size: num_sequences,
                self.model.sequence_length: self.sequence_length,
                self.model.mask_input: mini_batch["masks"].flatten(),
                self.model.advantage: mini_batch["advantages"].reshape([-1, 1]),
                self.model.all_old_log_probs: mini_batch["action_probs"].reshape(
                    [-1, sum(self.model.act_size)]
                ),
            }
            for name in self.reward_signals:
                feed_dict[self.model.returns_holders[name]] = mini_batch[
                    "{}_returns".format(name)
                ].flatten()
                feed_dict[self.model.old_values[name]] = mini_batch[
                    "{}_value_estimates".format(name)
                ].flatten()

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
                feed_dict[self.model.vector_in] = mini_batch["vector_obs"].reshape(
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
            if self.use_recurrent:
                mem_in = mini_batch["memory"][:, 0, :]
                feed_dict[self.model.memory_in] = mem_in

        out = self._execute_model(feed_dict, self.update_dict)
        run_out["value_loss"] = np.mean([out["value_loss_%d".format(i)] for i in range(len(self.towers))])
        run_out["policy_loss"] = np.mean([out["policy_loss_%d".format(i)] for i in range(len(self.towers))])
        run_out["update_batch"] = out["update_batch"]
        return run_out

    def average_gradients(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = [g for g, _ in grad_and_vars if g is not None]
            if not grads:
                continue
            avg_grad = tf.reduce_mean(tf.stack(grads), 0)
            var = grad_and_vars[0][1]
            average_grads.append((avg_grad, var))
        return average_grads


def get_devices():
    local_device_protos = device_lib.list_local_devices()
    devices = [x.name for x in local_device_protos if x.device_type == "GPU"]
    return devices

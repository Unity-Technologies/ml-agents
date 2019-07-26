import logging
import numpy as np

import tensorflow as tf
from tensorflow.python.client import device_lib
from mlagents.envs.timers import timed
from mlagents.trainers.ppo.policy import PPOPolicy
from mlagents.trainers.ppo.models import PPOModel
from mlagents.trainers.components.reward_signals.reward_signal_factory import (
    create_reward_signal,
)
from mlagents.trainers.components.bc.module import BCModule

# Variable scope in which created variables will be placed under
TOWER_SCOPE_NAME = "tower"

logger = logging.getLogger("mlagents.trainers")


class MultiGpuPPOPolicy(PPOPolicy):
    def __init__(self, seed, brain, trainer_params, is_training, load):
        """
        Policy for Proximal Policy Optimization Networks with multi-GPU training
        :param seed: Random seed.
        :param brain: Assigned Brain object.
        :param trainer_params: Defined training parameters.
        :param is_training: Whether the model should be trained.
        :param load: Whether a pre-trained model will be loaded or a new one created.
        """
        super().__init__(seed, brain, trainer_params, is_training, load)

        self.update_batch = self.average_gradients([t.grads for t in self.towers])
        self.update_dict = {"update_batch": self.update_batch}
        self.update_dict.update(
            {
                "value_loss_%d".format(i): self.towers[i].value_loss
                for i in range(len(self.towers))
            }
        )
        self.update_dict.update(
            {
                "policy_loss_%d".format(i): self.towers[i].policy_loss
                for i in range(len(self.towers))
            }
        )

    def create_model(self, brain, trainer_params, reward_signal_configs, seed):
        """
        Create PPO model, one on each device
        :param brain: Assigned Brain object.
        :param trainer_params: Defined training parameters.
        :param reward_signal_configs: Reward signal config
        :param seed: Random seed.
        """
        self.devices = get_devices()
        self.towers = []
        with self.graph.as_default():
            for device in self.devices:
                with tf.device(device):
                    with tf.variable_scope(TOWER_SCOPE_NAME, reuse=tf.AUTO_REUSE):
                        self.towers.append(
                            PPOModel(
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
            self.model = self.towers[0]

    @timed
    def update(self, mini_batch, num_sequences):
        """
        Updates model using buffer.
        :param n_sequences: Number of trajectories in batch.
        :param mini_batch: Experience batch.
        :return: Output from update process.
        """
        feed_dict = {}

        device_batch_size = num_sequences // len(self.devices)
        device_batches = []
        for i in range(len(self.devices)):
            device_batches.append(
                {k: v[i : i + device_batch_size] for k, v in mini_batch.items()}
            )

        assert len(device_batches) == len(self.towers)
        for batch, tower in zip(device_batches, self.towers):
            feed_dict.update(self.construct_feed_dict(tower, batch, num_sequences))

        out = self._execute_model(feed_dict, self.update_dict)
        run_out = {}
        run_out["value_loss"] = np.mean(
            [out["value_loss_%d".format(i)] for i in range(len(self.towers))]
        )
        run_out["policy_loss"] = np.mean(
            [out["policy_loss_%d".format(i)] for i in range(len(self.towers))]
        )
        run_out["update_batch"] = out["update_batch"]
        return run_out

    def average_gradients(self, tower_grads):
        """
        Average gradients from all towers
        :param tower_grads: Gradients from all towers
        """
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
    """
    Get all available GPU devices
    """
    local_device_protos = device_lib.list_local_devices()
    devices = [x.name for x in local_device_protos if x.device_type == "GPU"]
    return devices

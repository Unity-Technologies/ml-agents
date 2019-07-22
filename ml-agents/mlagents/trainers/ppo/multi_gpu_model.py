import logging

import tensorflow as tf
from tensorflow.python.client import device_lib
from mlagents.trainers.ppo.models import PPOModel

# Variable scope in which created variables will be placed under
TOWER_SCOPE_NAME = "tower"

logger = logging.getLogger("mlagents.trainers")


class PPOMultiGPUModel:
    def __init__(self, devices, *args, **kwargs):
        """
        A multi-GPU wrapper for PPO model
        """
        self.towers = []
        for device in devices:
            with tf.device(device):
                with tf.variable_scope(TOWER_SCOPE_NAME, reuse=tf.AUTO_REUSE) as scope:
                    self.towers.append(PPOModel(*args, **kwargs))

        self.value_loss = tf.reduce_mean(tf.stack([t.value_loss for t in self.towers]))
        self.policy_loss = tf.reduce_mean(
            tf.stack([t.policy_loss for t in self.towers])
        )
        self.optimizer = self.towers[0].optimizer
        avg_grad = self.average_gradients([t.grads for t in self.towers])
        self.update_batch = self.optimizer.apply_gradients(avg_grad)

    def __getattr__(self, name):
        return getattr(self.towers[0], name)

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


def get_devices(multi_gpu):
    local_device_protos = device_lib.list_local_devices()
    devices = [x.name for x in local_device_protos if x.device_type == "GPU"]
    if len(devices) < 1:
        devices = ["/cpu:0"]
    if not multi_gpu:
        devices = devices[:1]
    return devices

import logging
from typing import Any, Dict, List, Optional

from mlagents.tf_utils import tf

from tensorflow.python.client import device_lib
from mlagents.trainers.brain import BrainParameters
from mlagents_envs.timers import timed
from mlagents.trainers.models import EncoderType, LearningRateSchedule
from mlagents.trainers.ppo.policy import PPOPolicy
from mlagents.trainers.ppo.models import PPOModel
from mlagents.trainers.components.reward_signals import RewardSignal
from mlagents.trainers.components.reward_signals.reward_signal_factory import (
    create_reward_signal,
)

# Variable scope in which created variables will be placed under
TOWER_SCOPE_NAME = "tower"

logger = logging.getLogger("mlagents.trainers")


class MultiGpuPPOPolicy(PPOPolicy):
    def __init__(
        self,
        seed: int,
        brain: BrainParameters,
        trainer_params: Dict[str, Any],
        is_training: bool,
        load: bool,
    ):
        self.towers: List[PPOModel] = []
        self.devices: List[str] = []
        self.model: Optional[PPOModel] = None
        self.total_policy_loss: Optional[tf.Tensor] = None
        self.reward_signal_towers: List[Dict[str, RewardSignal]] = []
        self.reward_signals: Dict[str, RewardSignal] = {}

        super().__init__(seed, brain, trainer_params, is_training, load)

    def create_model(
        self, brain, trainer_params, reward_signal_configs, is_training, load, seed
    ):
        """
        Create PPO models, one on each device
        :param brain: Assigned Brain object.
        :param trainer_params: Defined training parameters.
        :param reward_signal_configs: Reward signal config
        :param seed: Random seed.
        """
        self.devices = get_devices()

        with self.graph.as_default():
            with tf.variable_scope("", reuse=tf.AUTO_REUSE):
                for device in self.devices:
                    with tf.device(device):
                        self.towers.append(
                            PPOModel(
                                brain=brain,
                                lr=float(trainer_params["learning_rate"]),
                                lr_schedule=LearningRateSchedule(
                                    trainer_params.get(
                                        "learning_rate_schedule", "linear"
                                    )
                                ),
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
                                vis_encode_type=EncoderType(
                                    trainer_params.get("vis_encode_type", "simple")
                                ),
                            )
                        )
                        self.towers[-1].create_ppo_optimizer()
            self.model = self.towers[0]
            avg_grads = self.average_gradients([t.grads for t in self.towers])
            update_batch = self.model.optimizer.apply_gradients(avg_grads)

            avg_value_loss = tf.reduce_mean(
                tf.stack([model.value_loss for model in self.towers]), 0
            )
            avg_policy_loss = tf.reduce_mean(
                tf.stack([model.policy_loss for model in self.towers]), 0
            )

        self.inference_dict.update(
            {
                "action": self.model.output,
                "log_probs": self.model.all_log_probs,
                "value_heads": self.model.value_heads,
                "value": self.model.value,
                "entropy": self.model.entropy,
                "learning_rate": self.model.learning_rate,
            }
        )
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

        self.total_policy_loss = self.model.abs_policy_loss
        self.update_dict.update(
            {
                "value_loss": avg_value_loss,
                "policy_loss": avg_policy_loss,
                "update_batch": update_batch,
            }
        )

    def create_reward_signals(self, reward_signal_configs):
        """
        Create reward signals
        :param reward_signal_configs: Reward signal config.
        """
        with self.graph.as_default():
            with tf.variable_scope(TOWER_SCOPE_NAME, reuse=tf.AUTO_REUSE):
                for device_id, device in enumerate(self.devices):
                    with tf.device(device):
                        reward_tower = {}
                        for reward_signal, config in reward_signal_configs.items():
                            reward_tower[reward_signal] = create_reward_signal(
                                self, self.towers[device_id], reward_signal, config
                            )
                            for k, v in reward_tower[reward_signal].update_dict.items():
                                self.update_dict[k + "_" + str(device_id)] = v
                        self.reward_signal_towers.append(reward_tower)
                for _, reward_tower in self.reward_signal_towers[0].items():
                    for _, update_key in reward_tower.stats_name_to_update_name.items():
                        all_reward_signal_stats = tf.stack(
                            [
                                self.update_dict[update_key + "_" + str(i)]
                                for i in range(len(self.towers))
                            ]
                        )
                        mean_reward_signal_stats = tf.reduce_mean(
                            all_reward_signal_stats, 0
                        )
                        self.update_dict.update({update_key: mean_reward_signal_stats})

            self.reward_signals = self.reward_signal_towers[0]

    @timed
    def update(self, mini_batch, num_sequences):
        """
        Updates model using buffer.
        :param n_sequences: Number of trajectories in batch.
        :param mini_batch: Experience batch.
        :return: Output from update process.
        """
        feed_dict = {}
        stats_needed = self.stats_name_to_update_name

        device_batch_size = num_sequences // len(self.devices)
        device_batches = []
        for i in range(len(self.devices)):
            device_batches.append(
                {
                    k: v[
                        i * device_batch_size : i * device_batch_size
                        + device_batch_size
                    ]
                    for (k, v) in mini_batch.items()
                }
            )

        for batch, tower, reward_tower in zip(
            device_batches, self.towers, self.reward_signal_towers
        ):
            feed_dict.update(self.construct_feed_dict(tower, batch, num_sequences))
            stats_needed.update(self.stats_name_to_update_name)
            for _, reward_signal in reward_tower.items():
                feed_dict.update(
                    reward_signal.prepare_update(tower, batch, num_sequences)
                )
                stats_needed.update(reward_signal.stats_name_to_update_name)

        update_vals = self._execute_model(feed_dict, self.update_dict)
        update_stats = {}
        for stat_name, update_name in stats_needed.items():
            update_stats[stat_name] = update_vals[update_name]
        return update_stats

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


def get_devices() -> List[str]:
    """
    Get all available GPU devices
    """
    local_device_protos = device_lib.list_local_devices()
    devices = [x.name for x in local_device_protos if x.device_type == "GPU"]
    return devices

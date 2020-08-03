import os

import torch
from mlagents_envs.logging_util import get_logger
from mlagents.trainers.saver.saver import Saver
from mlagents.model_serialization import SerializationSettings
from mlagents_envs.base_env import BehaviorSpec
from mlagents.trainers.settings import TrainerSettings
from mlagents.trainers.policy.torch_policy import TorchPolicy


logger = get_logger(__name__)


class TorchSaver(Saver):
    """
    Saver class for PyTorch
    """
    def __init__(
        self,
        policy: TorchPolicy,
        trainer_settings: TrainerSettings,
        model_path: str,
        load: bool = False,
    ):
        super().__init__()
        self.policy = policy
        self.model_path = model_path
        self.initialize_path = trainer_settings.init_path
        self._keep_checkpoints = trainer_settings.keep_checkpoints
        self.load = load

        self.modules = {}

    def register(self, module):
        self.modules.update(module.get_modules())
    
    def save_checkpoint(self, checkpoint_path: str, settings: SerializationSettings) -> None:
        """
        Checkpoints the policy on disk.

        :param checkpoint_path: filepath to write the checkpoint
        :param settings: SerializationSettings for exporting the model.
        """
        print('save checkpoint_path:', checkpoint_path)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        state_dict = {name: module.state_dict() for name, module in self.modules.items()}
        torch.save(state_dict, f"{checkpoint_path}.pt")
        torch.save(state_dict, os.path.join(self.model_path, "checkpoint.pt"))

    def maybe_load(self):
        # If there is an initialize path, load from that. Else, load from the set model path.
        # If load is set to True, don't reset steps to 0. Else, do. This allows a user to,
        # e.g., resume from an initialize path.
        reset_steps = not self.load
        if self.initialize_path is not None:
            self._load_model(self.initialize_path, reset_global_steps=reset_steps)
        elif self.load:
            self._load_model(self.model_path, reset_global_steps=reset_steps)

    def export(self, output_filepath: str, settings: SerializationSettings) -> None:
        print('export output_filepath:', output_filepath)
        fake_vec_obs = [torch.zeros([1] + [self.policy.vec_obs_size])]
        fake_vis_obs = [torch.zeros([1] + [84, 84, 3])]
        fake_masks = torch.ones([1] + self.policy.actor_critic.act_size)
        # print(fake_vec_obs[0].shape, fake_vis_obs[0].shape, fake_masks.shape)
        # fake_memories = torch.zeros([1] + [self.m_size])
        output_names = ["action", "action_probs", "is_continuous_control", \
            "version_number", "memory_size", "action_output_shape"]
        input_names = ["vector_observation", "action_mask"]
        dynamic_axes = {"vector_observation": [0], "action": [0], "action_probs": [0]}
        torch.onnx.export(
            self.policy.actor_critic,
            (fake_vec_obs, fake_vis_obs, fake_masks),
            f"{output_filepath}.onnx",
            verbose=False,
            opset_version=9,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    def _load_model(self, load_path: str, reset_global_steps: bool = False) -> None:
        model_path = os.path.join(load_path, "checkpoint.pt")
        print('load model_path:', model_path)
        saved_state_dict = torch.load(model_path)
        for name, state_dict in saved_state_dict.items():
            self.modules[name].load_state_dict(state_dict)
        if reset_global_steps:
            self.policy._set_step(0)
            logger.info(
                "Starting training from step 0 and saving to {}.".format(
                    self.model_path
                )
            )
        else:
            logger.info(f"Resuming training from step {self.policy.get_current_step()}.")

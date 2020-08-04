import os

import torch
from mlagents_envs.logging_util import get_logger
from mlagents.trainers.saver.saver import BaseSaver
from mlagents_envs.base_env import BehaviorSpec
from mlagents.trainers.settings import TrainerSettings
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.torch.model_serialization import ModelSerializer


logger = get_logger(__name__)


class TorchSaver(BaseSaver):
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
        self.exporter = ModelSerializer(self.policy)

        self.modules = {}

    def register(self, module):
        self.modules.update(module.get_modules())

    def save_checkpoint(self, checkpoint_path: str, brain_name: str) -> None:
        """
        Checkpoints the policy on disk.

        :param checkpoint_path: filepath to write the checkpoint
        :param brain_name: Brain name of brain to be trained
        """
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        state_dict = {name: module.state_dict() for name, module in self.modules.items()}
        torch.save(state_dict, f"{checkpoint_path}.pt")
        torch.save(state_dict, os.path.join(self.model_path, "checkpoint.pt"))
        self.export(checkpoint_path, brain_name)

    def maybe_load(self):
        # If there is an initialize path, load from that. Else, load from the set model path.
        # If load is set to True, don't reset steps to 0. Else, do. This allows a user to,
        # e.g., resume from an initialize path.
        reset_steps = not self.load
        if self.initialize_path is not None:
            self._load_model(self.initialize_path, reset_global_steps=reset_steps)
        elif self.load:
            self._load_model(self.model_path, reset_global_steps=reset_steps)

    def export(self, output_filepath: str, brain_name: str) -> None:
        self.exporter.export_policy_model(output_filepath)

    def _load_model(self, load_path: str, reset_global_steps: bool = False) -> None:
        model_path = os.path.join(load_path, "checkpoint.pt")
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

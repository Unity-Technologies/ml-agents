import os
import shutil
from mlagents.torch_utils import torch
from typing import Dict, Union, Optional, cast
from mlagents_envs.exception import UnityPolicyException
from mlagents_envs.logging_util import get_logger
from mlagents.trainers.model_saver.model_saver import BaseModelSaver
from mlagents.trainers.settings import TrainerSettings, SerializationSettings
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.torch.model_serialization import ModelSerializer


logger = get_logger(__name__)


class TorchModelSaver(BaseModelSaver):
    """
    ModelSaver class for PyTorch
    """

    def __init__(
        self, trainer_settings: TrainerSettings, model_path: str, load: bool = False
    ):
        super().__init__()
        self.model_path = model_path
        self.initialize_path = trainer_settings.init_path
        self._keep_checkpoints = trainer_settings.keep_checkpoints
        self.load = load

        self.policy: Optional[TorchPolicy] = None
        self.exporter: Optional[ModelSerializer] = None
        self.modules: Dict[str, torch.nn.Modules] = {}

    def register(self, module: Union[TorchPolicy, TorchOptimizer]) -> None:
        if isinstance(module, TorchPolicy) or isinstance(module, TorchOptimizer):
            self.modules.update(module.get_modules())  # type: ignore
        else:
            raise UnityPolicyException(
                "Registering Object of unsupported type {} to ModelSaver ".format(
                    type(module)
                )
            )
        if self.policy is None and isinstance(module, TorchPolicy):
            self.policy = module
            self.exporter = ModelSerializer(self.policy)

    def save_checkpoint(self, behavior_name: str, step: int) -> str:
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        checkpoint_path = os.path.join(self.model_path, f"{behavior_name}-{step}")
        state_dict = {
            name: module.state_dict() for name, module in self.modules.items()
        }
        torch.save(state_dict, f"{checkpoint_path}.pt")
        torch.save(state_dict, os.path.join(self.model_path, "checkpoint.pt"))
        self.export(checkpoint_path, behavior_name)
        return checkpoint_path

    def export(self, output_filepath: str, behavior_name: str) -> None:
        if self.exporter is not None:
            self.exporter.export_policy_model(output_filepath)

    def initialize_or_load(self, policy: Optional[TorchPolicy] = None) -> None:
        # Initialize/Load registered self.policy by default.
        # If given input argument policy, use the input policy instead.
        # This argument is mainly for initialization of the ghost trainer's fixed policy.
        reset_steps = not self.load
        if self.initialize_path is not None:
            self._load_model(
                self.initialize_path, policy, reset_global_steps=reset_steps
            )
        elif self.load:
            self._load_model(self.model_path, policy, reset_global_steps=reset_steps)

    def _load_model(
        self,
        load_path: str,
        policy: Optional[TorchPolicy] = None,
        reset_global_steps: bool = False,
    ) -> None:
        model_path = os.path.join(load_path, "checkpoint.pt")
        saved_state_dict = torch.load(model_path)
        if policy is None:
            modules = self.modules
            policy = self.policy
        else:
            modules = policy.get_modules()
        policy = cast(TorchPolicy, policy)

        for name, mod in modules.items():
            mod.load_state_dict(saved_state_dict[name])

        if reset_global_steps:
            policy.set_step(0)
            logger.info(
                "Starting training from step 0 and saving to {}.".format(
                    self.model_path
                )
            )
        else:
            logger.info(f"Resuming training from step {policy.get_current_step()}.")

    def copy_final_model(self, source_nn_path: str) -> None:
        """
        Copy the .nn file at the given source to the destination.
        Also copies the corresponding .onnx file if it exists.
        """
        final_model_name = os.path.splitext(source_nn_path)[0]

        if SerializationSettings.convert_to_onnx:
            try:
                source_path = f"{final_model_name}.onnx"
                destination_path = f"{self.model_path}.onnx"
                shutil.copyfile(source_path, destination_path)
                logger.info(f"Copied {source_path} to {destination_path}.")
            except OSError:
                pass

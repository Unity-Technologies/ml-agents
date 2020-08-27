# # Unity ML-Agents Toolkit
import abc
from typing import Any


class BaseModelSaver(abc.ABC):
    """This class is the base class for the ModelSaver"""

    def __init__(self):
        pass

    @abc.abstractmethod
    def register(self, module: Any) -> None:
        """
        Register the modules to the ModelSaver.
        The ModelSaver will store the module and include it in the saved files
        when saving checkpoint/exporting graph.
        :param module: the module to be registered
        """
        pass

    def _register_policy(self, policy):
        """
        Helper function for registering policy to the ModelSaver.
        :param policy: the policy to be registered
        """
        pass

    def _register_optimizer(self, optimizer):
        """
        Helper function for registering optimizer to the ModelSaver.
        :param optimizer: the optimizer to be registered
        """
        pass

    @abc.abstractmethod
    def save_checkpoint(self, behavior_name: str, step: int) -> str:
        """
        Checkpoints the policy on disk.
        :param checkpoint_path: filepath to write the checkpoint
        :param behavior_name: Behavior name of bevavior to be trained
        """
        pass

    @abc.abstractmethod
    def export(self, output_filepath: str, behavior_name: str) -> None:
        """
        Saves the serialized model, given a path and behavior name.
        This method will save the policy graph to the given filepath.  The path
        should be provided without an extension as multiple serialized model formats
        may be generated as a result.
        :param output_filepath: path (without suffix) for the model file(s)
        :param behavior_name: Behavior name of behavior to be trained.
        """
        pass

    @abc.abstractmethod
    def initialize_or_load(self, policy):
        """
        Initialize/Load registered modules by default.
        If given input argument policy, do with the input policy instead.
        This argument is mainly for the initialization of the ghost trainer's fixed policy.
        :param policy (optional): if given, perform the initializing/loading on this input policy.
                                  Otherwise, do with the registered policy
        """
        pass

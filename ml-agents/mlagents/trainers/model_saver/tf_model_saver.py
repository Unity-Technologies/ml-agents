import os
import shutil
from typing import Optional, Union, cast
from mlagents_envs.exception import UnityPolicyException
from mlagents_envs.logging_util import get_logger
from mlagents.tf_utils import tf
from mlagents.trainers.model_saver.model_saver import BaseModelSaver
from mlagents.trainers.tf.model_serialization import export_policy_model
from mlagents.trainers.settings import TrainerSettings, SerializationSettings
from mlagents.trainers.policy.tf_policy import TFPolicy
from mlagents.trainers.optimizer.tf_optimizer import TFOptimizer
from mlagents.trainers import __version__


logger = get_logger(__name__)


class TFModelSaver(BaseModelSaver):
    """
    ModelSaver class for TensorFlow
    """

    def __init__(
        self, trainer_settings: TrainerSettings, model_path: str, load: bool = False
    ):
        super().__init__()
        self.model_path = model_path
        self.initialize_path = trainer_settings.init_path
        self._keep_checkpoints = trainer_settings.keep_checkpoints
        self.load = load

        # Currently only support saving one policy. This is the one to be saved.
        self.policy: Optional[TFPolicy] = None
        self.graph = None
        self.sess = None
        self.tf_saver = None

    def register(self, module: Union[TFPolicy, TFOptimizer]) -> None:
        if isinstance(module, TFPolicy):
            self._register_policy(module)
        elif isinstance(module, TFOptimizer):
            self._register_optimizer(module)
        else:
            raise UnityPolicyException(
                "Registering Object of unsupported type {} to Saver ".format(
                    type(module)
                )
            )

    def _register_policy(self, policy: TFPolicy) -> None:
        if self.policy is None:
            self.policy = policy
            self.graph = self.policy.graph
            self.sess = self.policy.sess
            with self.policy.graph.as_default():
                self.tf_saver = tf.train.Saver(max_to_keep=self._keep_checkpoints)

    def save_checkpoint(self, behavior_name: str, step: int) -> str:
        checkpoint_path = os.path.join(self.model_path, f"{behavior_name}-{step}")
        # Save the TF checkpoint and graph definition
        if self.graph:
            with self.graph.as_default():
                if self.tf_saver:
                    self.tf_saver.save(self.sess, f"{checkpoint_path}.ckpt")
                tf.train.write_graph(
                    self.graph, self.model_path, "raw_graph_def.pb", as_text=False
                )
        # also save the policy so we have optimized model files for each checkpoint
        self.export(checkpoint_path, behavior_name)
        return checkpoint_path

    def export(self, output_filepath: str, behavior_name: str) -> None:
        # save model if there is only one worker or
        # only on worker-0 if there are multiple workers
        if self.policy and self.policy.rank is not None and self.policy.rank != 0:
            return
        export_policy_model(
            self.model_path, output_filepath, behavior_name, self.graph, self.sess
        )

    def initialize_or_load(self, policy: Optional[TFPolicy] = None) -> None:
        # If there is an initialize path, load from that. Else, load from the set model path.
        # If load is set to True, don't reset steps to 0. Else, do. This allows a user to,
        # e.g., resume from an initialize path.
        if policy is None:
            policy = self.policy
        policy = cast(TFPolicy, policy)
        reset_steps = not self.load
        if self.initialize_path is not None:
            self._load_graph(
                policy, self.initialize_path, reset_global_steps=reset_steps
            )
        elif self.load:
            self._load_graph(policy, self.model_path, reset_global_steps=reset_steps)
        else:
            policy.initialize()
        TFPolicy.broadcast_global_variables(0)

    def _load_graph(
        self, policy: TFPolicy, model_path: str, reset_global_steps: bool = False
    ) -> None:
        with policy.graph.as_default():
            logger.info(f"Loading model from {model_path}.")
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt is None:
                raise UnityPolicyException(
                    "The model {} could not be loaded. Make "
                    "sure you specified the right "
                    "--run-id and that the previous run you are loading from had the same "
                    "behavior names.".format(model_path)
                )
            if self.tf_saver:
                try:
                    self.tf_saver.restore(policy.sess, ckpt.model_checkpoint_path)
                except tf.errors.NotFoundError:
                    raise UnityPolicyException(
                        "The model {} was found but could not be loaded. Make "
                        "sure the model is from the same version of ML-Agents, has the same behavior parameters, "
                        "and is using the same trainer configuration as the current run.".format(
                            model_path
                        )
                    )
            self._check_model_version(__version__)
            if reset_global_steps:
                policy.set_step(0)
                logger.info(
                    "Starting training from step 0 and saving to {}.".format(
                        self.model_path
                    )
                )
            else:
                logger.info(f"Resuming training from step {policy.get_current_step()}.")

    def _check_model_version(self, version: str) -> None:
        """
        Checks whether the model being loaded was created with the same version of
        ML-Agents, and throw a warning if not so.
        """
        if self.policy is not None and self.policy.version_tensors is not None:
            loaded_ver = tuple(
                num.eval(session=self.sess) for num in self.policy.version_tensors
            )
            if loaded_ver != TFPolicy._convert_version_string(version):
                logger.warning(
                    f"The model checkpoint you are loading from was saved with ML-Agents version "
                    f"{loaded_ver[0]}.{loaded_ver[1]}.{loaded_ver[2]} but your current ML-Agents"
                    f"version is {version}. Model may not behave properly."
                )

    def copy_final_model(self, source_nn_path: str) -> None:
        """
        Copy the .nn file at the given source to the destination.
        Also copies the corresponding .onnx file if it exists.
        """
        final_model_name = os.path.splitext(source_nn_path)[0]

        if SerializationSettings.convert_to_barracuda:
            source_path = f"{final_model_name}.nn"
            destination_path = f"{self.model_path}.nn"
            shutil.copyfile(source_path, destination_path)
            logger.info(f"Copied {source_path} to {destination_path}.")

        if SerializationSettings.convert_to_onnx:
            try:
                source_path = f"{final_model_name}.onnx"
                destination_path = f"{self.model_path}.onnx"
                shutil.copyfile(source_path, destination_path)
                logger.info(f"Copied {source_path} to {destination_path}.")
            except OSError:
                pass

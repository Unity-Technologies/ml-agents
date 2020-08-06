import os
from mlagents_envs.exception import UnityPolicyException
from mlagents_envs.logging_util import get_logger
from mlagents.tf_utils import tf
from mlagents.trainers.saver.saver import BaseSaver
from mlagents.trainers.tf.model_serialization import export_policy_model
from mlagents.trainers.settings import TrainerSettings
from mlagents.trainers.policy.tf_policy import TFPolicy
from mlagents.trainers import __version__


logger = get_logger(__name__)


class TFSaver(BaseSaver):
    """
    Saver class for TensorFlow
    """

    def __init__(
        self,
        policy: TFPolicy,
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

        self.graph = self.policy.graph
        self.sess = self.policy.sess
        with self.graph.as_default():
            self.saver = tf.train.Saver(max_to_keep=self._keep_checkpoints)

    def register(self, module):
        pass

    def save_checkpoint(self, brain_name: str, step: int) -> str:
        """
        Checkpoints the policy on disk.

        :param checkpoint_path: filepath to write the checkpoint
        :param brain_name: Brain name of brain to be trained
        """
        checkpoint_path = os.path.join(self.model_path, f"{brain_name}-{step}")
        # Save the TF checkpoint and graph definition
        with self.graph.as_default():
            if self.saver:
                self.saver.save(self.sess, f"{checkpoint_path}.ckpt")
            tf.train.write_graph(
                self.graph, self.model_path, "raw_graph_def.pb", as_text=False
            )
        # also save the policy so we have optimized model files for each checkpoint
        self.export(checkpoint_path, brain_name)
        return checkpoint_path

    def export(self, output_filepath: str, brain_name: str) -> None:
        """
        Saves the serialized model, given a path and brain name.

        This method will save the policy graph to the given filepath.  The path
        should be provided without an extension as multiple serialized model formats
        may be generated as a result.

        :param output_filepath: path (without suffix) for the model file(s)
        :param brain_name: Brain name of brain to be trained.
        """
        export_policy_model(output_filepath, brain_name, self.graph, self.sess)

    def maybe_load(self):
        # If there is an initialize path, load from that. Else, load from the set model path.
        # If load is set to True, don't reset steps to 0. Else, do. This allows a user to,
        # e.g., resume from an initialize path.
        reset_steps = not self.load
        if self.initialize_path is not None:
            self._load_graph(self.initialize_path, reset_global_steps=reset_steps)
        elif self.load:
            self._load_graph(self.model_path, reset_global_steps=reset_steps)
        else:
            self.policy._initialize_graph()

    def _load_graph(self, model_path: str, reset_global_steps: bool = False) -> None:
        with self.graph.as_default():
            logger.info(f"Loading model from {model_path}.")
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt is None:
                raise UnityPolicyException(
                    "The model {} could not be loaded. Make "
                    "sure you specified the right "
                    "--run-id and that the previous run you are loading from had the same "
                    "behavior names.".format(model_path)
                )
            try:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
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
                self.policy.set_step(0)
                logger.info(
                    "Starting training from step 0 and saving to {}.".format(
                        self.model_path
                    )
                )
            else:
                logger.info(
                    f"Resuming training from step {self.policy.get_current_step()}."
                )

    def _check_model_version(self, version: str) -> None:
        """
        Checks whether the model being loaded was created with the same version of
        ML-Agents, and throw a warning if not so.
        """
        if self.policy.version_tensors is not None:
            loaded_ver = tuple(
                num.eval(session=self.sess) for num in self.policy.version_tensors
            )
            if loaded_ver != TFPolicy._convert_version_string(version):
                logger.warning(
                    f"The model checkpoint you are loading from was saved with ML-Agents version "
                    f"{loaded_ver[0]}.{loaded_ver[1]}.{loaded_ver[2]} but your current ML-Agents"
                    f"version is {version}. Model may not behave properly."
                )

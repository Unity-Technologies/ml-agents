import threading
from mlagents.torch_utils import torch

from mlagents_envs.logging_util import get_logger
from mlagents.trainers.settings import SerializationSettings


logger = get_logger(__name__)


class exporting_to_onnx:
    """
    Set this context by calling
    ```
    with exporting_to_onnx():
    ```
    Within this context, the variable exporting_to_onnx.is_exporting() will be true.
    This implementation is thread safe.
    """

    # local is_exporting flag for each thread
    _local_data = threading.local()
    _local_data._is_exporting = False

    # global lock shared among all threads, to make sure only one thread is exporting at a time
    _lock = threading.Lock()

    def __enter__(self):
        self._lock.acquire()
        self._local_data._is_exporting = True

    def __exit__(self, *args):
        self._local_data._is_exporting = False
        self._lock.release()

    @staticmethod
    def is_exporting():
        if not hasattr(exporting_to_onnx._local_data, "_is_exporting"):
            return False
        return exporting_to_onnx._local_data._is_exporting


class ModelSerializer:
    def __init__(self, policy):
        # ONNX only support input in NCHW (channel first) format.
        # Barracuda also expect to get data in NCHW.
        # Any multi-dimentional input should follow that otherwise will
        # cause problem to barracuda import.
        self.policy = policy
        observation_specs = self.policy.behavior_spec.observation_specs
        batch_dim = [1]
        seq_len_dim = [1]
        vec_obs_size = 0
        for obs_spec in observation_specs:
            if len(obs_spec.shape) == 1:
                vec_obs_size += obs_spec.shape[0]
        num_vis_obs = sum(
            1 for obs_spec in observation_specs if len(obs_spec.shape) == 3
        )
        dummy_vec_obs = [torch.zeros(batch_dim + [vec_obs_size])]
        # create input shape of NCHW
        # (It's NHWC in observation_specs.shape)
        dummy_vis_obs = [
            torch.zeros(
                batch_dim + [obs_spec.shape[2], obs_spec.shape[0], obs_spec.shape[1]]
            )
            for obs_spec in observation_specs
            if len(obs_spec.shape) == 3
        ]

        dummy_var_len_obs = [
            torch.zeros(batch_dim + [obs_spec.shape[0], obs_spec.shape[1]])
            for obs_spec in observation_specs
            if len(obs_spec.shape) == 2
        ]

        dummy_masks = torch.ones(
            batch_dim + [sum(self.policy.behavior_spec.action_spec.discrete_branches)]
        )
        dummy_memories = torch.zeros(
            batch_dim + seq_len_dim + [self.policy.export_memory_size]
        )

        self.dummy_input = (
            dummy_vec_obs,
            dummy_vis_obs,
            dummy_var_len_obs,
            dummy_masks,
            dummy_memories,
        )

        self.input_names = ["vector_observation"]
        for i in range(num_vis_obs):
            self.input_names.append(f"visual_observation_{i}")
        for i, obs_spec in enumerate(observation_specs):
            if len(obs_spec.shape) == 2:
                self.input_names.append(f"obs_{i}")
        self.input_names += ["action_masks", "memories"]

        self.dynamic_axes = {name: {0: "batch"} for name in self.input_names}

        self.output_names = ["version_number", "memory_size"]
        if self.policy.behavior_spec.action_spec.continuous_size > 0:
            self.output_names += [
                "continuous_actions",
                "continuous_action_output_shape",
            ]
            self.dynamic_axes.update({"continuous_actions": {0: "batch"}})
        if self.policy.behavior_spec.action_spec.discrete_size > 0:
            self.output_names += ["discrete_actions", "discrete_action_output_shape"]
            self.dynamic_axes.update({"discrete_actions": {0: "batch"}})
        if (
            self.policy.behavior_spec.action_spec.continuous_size == 0
            or self.policy.behavior_spec.action_spec.discrete_size == 0
        ):
            self.output_names += [
                "action",
                "is_continuous_control",
                "action_output_shape",
            ]
            self.dynamic_axes.update({"action": {0: "batch"}})

    def export_policy_model(self, output_filepath: str) -> None:
        """
        Exports a Torch model for a Policy to .onnx format for Unity embedding.

        :param output_filepath: file path to output the model (without file suffix)
        """
        onnx_output_path = f"{output_filepath}.onnx"
        logger.info(f"Converting to {onnx_output_path}")

        with exporting_to_onnx():
            torch.onnx.export(
                self.policy.actor_critic,
                self.dummy_input,
                onnx_output_path,
                opset_version=SerializationSettings.onnx_opset,
                input_names=self.input_names,
                output_names=self.output_names,
                dynamic_axes=self.dynamic_axes,
            )
        logger.info(f"Exported {onnx_output_path}")

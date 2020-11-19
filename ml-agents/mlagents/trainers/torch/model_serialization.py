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
        batch_dim = [1]
        seq_len_dim = [1]
        dummy_vec_obs = [torch.zeros(batch_dim + [self.policy.vec_obs_size])]
        # create input shape of NCHW
        # (It's NHWC in self.policy.behavior_spec.observation_shapes)
        dummy_vis_obs = [
            torch.zeros(batch_dim + [shape[2], shape[0], shape[1]])
            for shape in self.policy.behavior_spec.observation_shapes
            if len(shape) == 3
        ]
        dummy_masks = torch.ones(
            batch_dim + [sum(self.policy.behavior_spec.action_spec.discrete_branches)]
        )
        dummy_memories = torch.zeros(
            batch_dim + seq_len_dim + [self.policy.export_memory_size]
        )

        self.dummy_input = (dummy_vec_obs, dummy_vis_obs, dummy_masks, dummy_memories)

        self.input_names = (
            ["vector_observation"]
            + [f"visual_observation_{i}" for i in range(self.policy.vis_obs_size)]
            + ["action_masks", "memories"]
        )

        self.output_names = [
            "action",
            "version_number",
            "memory_size",
            "is_continuous_control",
            "action_output_shape",
        ]

        self.dynamic_axes = {name: {0: "batch"} for name in self.input_names}
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

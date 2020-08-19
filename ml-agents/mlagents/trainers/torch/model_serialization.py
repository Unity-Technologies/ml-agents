import os
import torch

from mlagents_envs.logging_util import get_logger
from mlagents.trainers.settings import SerializationSettings


logger = get_logger(__name__)


class ModelSerializer:
    def __init__(self, policy):
        self.policy = policy
        batch_dim = [1]
        seq_len_dim = [1]
        dummy_vec_obs = [torch.zeros(batch_dim + [self.policy.vec_obs_size])]
        dummy_vis_obs = [
            torch.zeros(batch_dim + list(shape))
            for shape in self.policy.behavior_spec.observation_shapes
            if len(shape) == 3
        ]
        dummy_masks = torch.ones(batch_dim + [sum(self.policy.actor_critic.act_size)])
        dummy_memories = torch.zeros(batch_dim + seq_len_dim + [self.policy.m_size])

        self.dummy_input = (dummy_vec_obs, dummy_vis_obs, dummy_masks, dummy_memories)

        self.input_names = (
            ["vector_observation"]
            + [f"visual_observation_{i}" for i in range(self.policy.vis_obs_size)]
            + ["action_masks", "memories"]
        )

        self.output_names = [
            "action",
            "action_probs",
            "version_number",
            "memory_size",
            "is_continuous_control",
            "action_output_shape",
        ]

        self.dynamic_axes = {name: {0: "batch"} for name in self.input_names}
        self.dynamic_axes.update({"action": {0: "batch"}, "action_probs": {0: "batch"}})

    def export_policy_model(self, output_filepath: str) -> None:
        """
        Exports a Torch model for a Policy to .onnx format for Unity embedding.

        :param output_filepath: file path to output the model (without file suffix)
        :param brain_name: Brain name of brain to be trained
        """
        if not os.path.exists(output_filepath):
            os.makedirs(output_filepath)

        onnx_output_path = f"{output_filepath}.onnx"
        logger.info(f"Converting to {onnx_output_path}")

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

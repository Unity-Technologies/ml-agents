import os
import torch

from mlagents_envs.logging_util import get_logger
from mlagents.trainers.settings import SerializationSettings


logger = get_logger(__name__)


class ModelSerializer:
    def __init__(self, policy):
        self.policy = policy
        # dimension for batch (and sequence_length if use recurrent)
        dummy_dim = [1, 1] if self.policy.use_recurrent else [1]

        dummy_vec_obs = [torch.zeros(dummy_dim + [self.policy.vec_obs_size])]
        dummy_vis_obs = (
            [torch.zeros(dummy_dim + list(self.policy.vis_obs_shape))]
            if self.policy.vis_obs_size > 0
            else []
        )
        dummy_masks = torch.ones([1] + self.policy.actor_critic.act_size)
        dummy_memories = torch.zeros(dummy_dim + [self.policy.m_size])

        self.input_names = [
            "vector_observation",
            "visual_observation",
            "action_mask",
            "memories",
        ]
        self.output_names = [
            "action",
            "action_probs",
            "version_number",
            "memory_size",
            "is_continuous_control",
            "action_output_shape",
        ]
        self.dynamic_axes = {
            "vector_observation": [0],
            "visual_observation": [0],
            "action_mask": [0],
            "memories": [0],
            "action": [0],
            "action_probs": [0],
        }
        self.dummy_input = (dummy_vec_obs, dummy_vis_obs, dummy_masks, dummy_memories)

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
            verbose=True,
            opset_version=SerializationSettings.onnx_opset,
            input_names=self.input_names,
            output_names=self.output_names,
            dynamic_axes=self.dynamic_axes,
        )
        logger.info(f"Exported {onnx_output_path}")

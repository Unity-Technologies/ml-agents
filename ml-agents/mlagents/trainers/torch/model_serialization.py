import os
import torch

from mlagents_envs.logging_util import get_logger
from mlagents.trainers.settings import SerializationSettings


logger = get_logger(__name__)


class ModelSerializer:
    def __init__(self, policy):
        self.policy = policy
        batch_dim = [1]
        dummy_vec_obs = [torch.zeros(batch_dim + [self.policy.vec_obs_size])]
        dummy_vis_obs = (
            [torch.zeros(batch_dim + list(self.policy.vis_obs_shape))]
            if self.policy.vis_obs_size > 0
            else []
        )
        dummy_masks = torch.ones(batch_dim + [sum(self.policy.actor_critic.act_size)])
        dummy_memories = torch.zeros(batch_dim + [1] + [self.policy.m_size])

        # Need to pass all posslible inputs since currently keyword arguments is not
        # supported by torch.nn.export()
        self.dummy_input = (dummy_vec_obs, dummy_vis_obs, dummy_masks, dummy_memories)

        # Input names can only contain actual input used since in torch.nn.export
        # it maps input_names only to input nodes that exist in the graph
        self.input_names = []
        self.dynamic_axes = {"action": {0: "batch"}, "action_probs": {0: "batch"}}
        if self.policy.use_vec_obs:
            self.input_names.append("vector_observation")
            self.dynamic_axes.update({"vector_observation": {0: "batch"}})
        if self.policy.use_vis_obs:
            self.input_names.append("visual_observation")
            self.dynamic_axes.update({"visual_observation": {0: "batch"}})
        if not self.policy.use_continuous_act:
            self.input_names.append("action_masks")
            self.dynamic_axes.update({"action_masks": {0: "batch"}})
        if self.policy.use_recurrent:
            self.input_names.append("memories")
            self.dynamic_axes.update({"memories": {0: "batch"}})

        self.output_names = [
            "action",
            "action_probs",
            "version_number",
            "memory_size",
            "is_continuous_control",
            "action_output_shape",
        ]

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
            verbose=False,
            opset_version=SerializationSettings.onnx_opset,
            input_names=self.input_names,
            output_names=self.output_names,
            dynamic_axes=self.dynamic_axes,
        )
        logger.info(f"Exported {onnx_output_path}")

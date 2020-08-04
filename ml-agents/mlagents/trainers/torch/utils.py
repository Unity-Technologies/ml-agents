from typing import List, Optional, Tuple
import torch
import numpy as np
from torch import nn

from mlagents.trainers.torch.encoders import (
    SimpleVisualEncoder,
    ResNetVisualEncoder,
    NatureVisualEncoder,
    VectorEncoder,
    VectorAndUnnormalizedInputEncoder,
)
from mlagents.trainers.settings import EncoderType
from mlagents.trainers.exception import UnityTrainerException
from mlagents.trainers.torch.distributions import DistInstance, DiscreteDistInstance


class ModelUtils:
    # Minimum supported side for each encoder type. If refactoring an encoder, please
    # adjust these also.
    MIN_RESOLUTION_FOR_ENCODER = {
        EncoderType.SIMPLE: 20,
        EncoderType.NATURE_CNN: 36,
        EncoderType.RESNET: 15,
    }

    @staticmethod
    def swish(input_activation: torch.Tensor) -> torch.Tensor:
        """Swish activation function. For more info: https://arxiv.org/abs/1710.05941"""
        return torch.mul(input_activation, torch.sigmoid(input_activation))

    @staticmethod
    def get_encoder_for_type(encoder_type: EncoderType) -> nn.Module:
        ENCODER_FUNCTION_BY_TYPE = {
            EncoderType.SIMPLE: SimpleVisualEncoder,
            EncoderType.NATURE_CNN: NatureVisualEncoder,
            EncoderType.RESNET: ResNetVisualEncoder,
        }
        return ENCODER_FUNCTION_BY_TYPE.get(encoder_type)

    @staticmethod
    def _check_resolution_for_encoder(
        vis_in: torch.Tensor, vis_encoder_type: EncoderType
    ) -> None:
        min_res = ModelUtils.MIN_RESOLUTION_FOR_ENCODER[vis_encoder_type]
        # Note: PyTorch uses NCHW not NHWC (TF).
        height = vis_in.shape[2]
        width = vis_in.shape[3]
        if height < min_res or width < min_res:
            raise UnityTrainerException(
                f"Visual observation resolution ({width}x{height}) is too small for"
                f"the provided EncoderType ({vis_encoder_type.value}). The min dimension is {min_res}"
            )

    @staticmethod
    def create_encoders(
        observation_shapes: List[Tuple[int, ...]],
        h_size: int,
        num_layers: int,
        vis_encode_type: EncoderType,
        unnormalized_inputs: int = 0,
        normalize: bool = False,
    ) -> Tuple[nn.ModuleList, nn.ModuleList]:
        """
        Creates visual and vector encoders, along with their normalizers.
        :param observation_shapes: List of Tuples that represent the action dimensions.
        :param action_size: Number of additional un-normalized inputs to each vector encoder. Used for
            conditioining network on other values (e.g. actions for a Q function)
        :param h_size: Number of hidden units per layer.
        :param num_layers: Depth of MLP per encoder.
        :param vis_encode_type: Type of visual encoder to use.
        :param unnormalized_inputs: Vector inputs that should not be normalized, and added to the vector
            obs.
        :param normalize: Normalize all vector inputs.
        :return: Tuple of visual encoders and vector encoders each as a list.
        """
        visual_encoders: List[nn.Module] = []
        vector_encoders: List[nn.Module] = []

        visual_encoder_class = ModelUtils.get_encoder_for_type(vis_encode_type)
        vector_size = 0
        for i, dimension in enumerate(observation_shapes):
            if len(dimension) == 3:
                visual_encoders.append(
                    visual_encoder_class(
                        dimension[0], dimension[1], dimension[2], h_size
                    )
                )
            elif len(dimension) == 1:
                vector_size += dimension[0]
            else:
                raise UnityTrainerException(
                    f"Unsupported shape of {dimension} for observation {i}"
                )
        if vector_size + unnormalized_inputs > 0:
            if unnormalized_inputs > 0:
                vector_encoders.append(
                    VectorAndUnnormalizedInputEncoder(
                        vector_size, h_size, unnormalized_inputs, num_layers, normalize
                    )
                )
            else:
                vector_encoders.append(
                    VectorEncoder(vector_size, h_size, num_layers, normalize)
                )
        return nn.ModuleList(visual_encoders), nn.ModuleList(vector_encoders)

    @staticmethod
    def list_to_tensor(
        ndarray_list: List[np.ndarray], dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """
        Converts a list of numpy arrays into a tensor. MUCH faster than
        calling as_tensor on the list directly.
        """
        return torch.as_tensor(np.asanyarray(ndarray_list), dtype=dtype)

    @staticmethod
    def break_into_branches(
        concatenated_logits: torch.Tensor, action_size: List[int]
    ) -> List[torch.Tensor]:
        """
        Takes a concatenated set of logits that represent multiple discrete action branches
        and breaks it up into one Tensor per branch.
        :param concatenated_logits: Tensor that represents the concatenated action branches
        :param action_size: List of ints containing the number of possible actions for each branch.
        :return: A List of Tensors containing one tensor per branch.
        """
        action_idx = [0] + list(np.cumsum(action_size))
        branched_logits = [
            concatenated_logits[:, action_idx[i] : action_idx[i + 1]]
            for i in range(len(action_size))
        ]
        return branched_logits

    @staticmethod
    def actions_to_onehot(
        discrete_actions: torch.Tensor, action_size: List[int]
    ) -> List[torch.Tensor]:
        """
        Takes a tensor of discrete actions and turns it into a List of onehot encoding for each
        action.
        :param discrete_actions: Actions in integer form.
        :param action_size: List of branch sizes. Should be of same size as discrete_actions'
        last dimension.
        :return: List of one-hot tensors, one representing each branch.
        """
        onehot_branches = [
            torch.nn.functional.one_hot(_act.T, action_size[i])
            for i, _act in enumerate(discrete_actions.T)
        ]
        return onehot_branches

    @staticmethod
    def get_probs_and_entropy(
        action_list: List[torch.Tensor], dists: List[DistInstance]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        log_probs_list = []
        all_probs_list = []
        entropies_list = []
        for action, action_dist in zip(action_list, dists):
            log_prob = action_dist.log_prob(action)
            log_probs_list.append(log_prob)
            entropies_list.append(action_dist.entropy())
            if isinstance(action_dist, DiscreteDistInstance):
                all_probs_list.append(action_dist.all_log_prob())
        log_probs = torch.stack(log_probs_list, dim=-1)
        entropies = torch.stack(entropies_list, dim=-1)
        if not all_probs_list:
            log_probs = log_probs.squeeze(-1)
            entropies = entropies.squeeze(-1)
            all_probs = None
        else:
            all_probs = torch.cat(all_probs, dim=-1)
        return log_probs, entropies, all_probs

from typing import List, Tuple
from mlagents_envs.communicator_objects.agent_info_pb2 import AgentInfoProto
from mlagents_envs.communicator_objects.observation_pb2 import (
    ObservationProto,
    NONE,
    PNG,
)
from mlagents_envs.communicator_objects.brain_parameters_pb2 import BrainParametersProto
import numpy as np
from mlagents_envs.base_env import AgentGroupSpec, ActionType
import io
from mlagents_envs.rpc_utils import (
    agent_group_spec_from_proto,
    process_pixels,
    _process_visual_observation,
    _process_vector_observation,
    batched_step_result_from_proto,
)
from PIL import Image


def generate_list_agent_proto(
    n_agent: int, shape: List[Tuple[int]]
) -> List[AgentInfoProto]:
    result = []
    for agent_index in range(n_agent):
        ap = AgentInfoProto()
        ap.reward = agent_index
        ap.done = agent_index % 2 == 0
        ap.max_step_reached = agent_index % 2 == 1
        ap.id = agent_index
        ap.action_mask.extend([True, False] * 5)
        obs_proto_list = []
        for obs_index in range(len(shape)):
            obs_proto = ObservationProto()
            obs_proto.shape.extend(list(shape[obs_index]))
            obs_proto.compression_type = NONE
            obs_proto.float_data.data.extend([0.1] * np.prod(shape[obs_index]))
            obs_proto_list.append(obs_proto)
        ap.observations.extend(obs_proto_list)
        result.append(ap)
    return result


def generate_compressed_data(in_array: np.ndarray) -> bytes:
    image_arr = (in_array * 255).astype(np.uint8)
    im = Image.fromarray(image_arr, "RGB")
    byteIO = io.BytesIO()
    im.save(byteIO, format="PNG")
    return byteIO.getvalue()


def generate_compressed_proto_obs(in_array: np.ndarray) -> ObservationProto:
    obs_proto = ObservationProto()
    obs_proto.compressed_data = generate_compressed_data(in_array)
    obs_proto.compression_type = PNG
    obs_proto.shape.extend(in_array.shape)
    return obs_proto


def test_process_pixels():
    in_array = np.random.rand(128, 128, 3)
    byte_arr = generate_compressed_data(in_array)
    out_array = process_pixels(byte_arr, False)
    assert out_array.shape == (128, 128, 3)
    assert np.sum(in_array - out_array) / np.prod(in_array.shape) < 0.01
    assert (in_array - out_array < 0.01).all()


def test_process_pixels_gray():
    in_array = np.random.rand(128, 128, 3)
    byte_arr = generate_compressed_data(in_array)
    out_array = process_pixels(byte_arr, True)
    assert out_array.shape == (128, 128, 1)
    assert np.mean(in_array.mean(axis=2, keepdims=True) - out_array) < 0.01
    assert (in_array.mean(axis=2, keepdims=True) - out_array < 0.01).all()


def test_vector_observation():
    n_agents = 10
    shapes = [(3,), (4,)]
    list_proto = generate_list_agent_proto(n_agents, shapes)
    for obs_index, shape in enumerate(shapes):
        arr = _process_vector_observation(obs_index, shape, list_proto)
        assert list(arr.shape) == ([n_agents] + list(shape))
        assert (np.abs(arr - 0.1) < 0.01).all()


def test_process_visual_observation():
    in_array_1 = np.random.rand(128, 128, 3)
    proto_obs_1 = generate_compressed_proto_obs(in_array_1)
    in_array_2 = np.random.rand(128, 128, 3)
    proto_obs_2 = generate_compressed_proto_obs(in_array_2)
    ap1 = AgentInfoProto()
    ap1.observations.extend([proto_obs_1])
    ap2 = AgentInfoProto()
    ap2.observations.extend([proto_obs_2])
    ap_list = [ap1, ap2]
    arr = _process_visual_observation(0, (128, 128, 3), ap_list)
    assert list(arr.shape) == [2, 128, 128, 3]
    assert (arr[0, :, :, :] - in_array_1 < 0.01).all()
    assert (arr[1, :, :, :] - in_array_2 < 0.01).all()


def test_batched_step_result_from_proto():
    n_agents = 10
    shapes = [(3,), (4,)]
    group_spec = AgentGroupSpec(shapes, ActionType.CONTINUOUS, 3)
    ap_list = generate_list_agent_proto(n_agents, shapes)
    result = batched_step_result_from_proto(ap_list, group_spec)
    assert list(result.reward) == list(range(n_agents))
    assert list(result.agent_id) == list(range(n_agents))
    for index in range(n_agents):
        assert result.done[index] == (index % 2 == 0)
        assert result.max_step[index] == (index % 2 == 1)
    assert list(result.obs[0].shape) == [n_agents] + list(shapes[0])
    assert list(result.obs[1].shape) == [n_agents] + list(shapes[1])


def test_action_masking_discrete():
    n_agents = 10
    shapes = [(3,), (4,)]
    group_spec = AgentGroupSpec(shapes, ActionType.DISCRETE, (7, 3))
    ap_list = generate_list_agent_proto(n_agents, shapes)
    result = batched_step_result_from_proto(ap_list, group_spec)
    masks = result.action_mask
    assert isinstance(masks, list)
    assert len(masks) == 2
    assert masks[0].shape == (n_agents, 7)
    assert masks[1].shape == (n_agents, 3)
    assert masks[0][0, 0]
    assert not masks[1][0, 0]
    assert masks[1][0, 1]


def test_action_masking_discrete_1():
    n_agents = 10
    shapes = [(3,), (4,)]
    group_spec = AgentGroupSpec(shapes, ActionType.DISCRETE, (10,))
    ap_list = generate_list_agent_proto(n_agents, shapes)
    result = batched_step_result_from_proto(ap_list, group_spec)
    masks = result.action_mask
    assert isinstance(masks, list)
    assert len(masks) == 1
    assert masks[0].shape == (n_agents, 10)
    assert masks[0][0, 0]


def test_action_masking_discrete_2():
    n_agents = 10
    shapes = [(3,), (4,)]
    group_spec = AgentGroupSpec(shapes, ActionType.DISCRETE, (2, 2, 6))
    ap_list = generate_list_agent_proto(n_agents, shapes)
    result = batched_step_result_from_proto(ap_list, group_spec)
    masks = result.action_mask
    assert isinstance(masks, list)
    assert len(masks) == 3
    assert masks[0].shape == (n_agents, 2)
    assert masks[1].shape == (n_agents, 2)
    assert masks[2].shape == (n_agents, 6)
    assert masks[0][0, 0]


def test_action_masking_continuous():
    n_agents = 10
    shapes = [(3,), (4,)]
    group_spec = AgentGroupSpec(shapes, ActionType.CONTINUOUS, 10)
    ap_list = generate_list_agent_proto(n_agents, shapes)
    result = batched_step_result_from_proto(ap_list, group_spec)
    masks = result.action_mask
    assert masks is None


def test_agent_group_spec_from_proto():
    agent_proto = generate_list_agent_proto(1, [(3,), (4,)])[0]
    bp = BrainParametersProto()
    bp.vector_action_size.extend([5, 4])
    bp.vector_action_space_type = 0
    group_spec = agent_group_spec_from_proto(bp, agent_proto)
    assert group_spec.is_action_discrete()
    assert not group_spec.is_action_continuous()
    assert group_spec.observation_shapes == [(3,), (4,)]
    assert group_spec.discrete_action_branches == (5, 4)
    assert group_spec.action_size == 2
    bp = BrainParametersProto()
    bp.vector_action_size.extend([6])
    bp.vector_action_space_type = 1
    group_spec = agent_group_spec_from_proto(bp, agent_proto)
    assert not group_spec.is_action_discrete()
    assert group_spec.is_action_continuous()
    assert group_spec.action_size == 6

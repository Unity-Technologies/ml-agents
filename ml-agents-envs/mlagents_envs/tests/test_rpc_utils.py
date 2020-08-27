import io
import numpy as np
import pytest
from typing import List, Tuple

from mlagents_envs.communicator_objects.agent_info_pb2 import AgentInfoProto
from mlagents_envs.communicator_objects.observation_pb2 import (
    ObservationProto,
    NONE,
    PNG,
)
from mlagents_envs.communicator_objects.brain_parameters_pb2 import BrainParametersProto
from mlagents_envs.communicator_objects.agent_info_action_pair_pb2 import (
    AgentInfoActionPairProto,
)
from mlagents_envs.communicator_objects.agent_action_pb2 import AgentActionProto
from mlagents_envs.base_env import (
    BehaviorSpec,
    ActionType,
    DecisionSteps,
    TerminalSteps,
)
from mlagents_envs.exception import UnityObservationException
from mlagents_envs.rpc_utils import (
    behavior_spec_from_proto,
    process_pixels,
    _process_visual_observation,
    _process_vector_observation,
    steps_from_proto,
)
from PIL import Image


def generate_list_agent_proto(
    n_agent: int,
    shape: List[Tuple[int]],
    infinite_rewards: bool = False,
    nan_observations: bool = False,
) -> List[AgentInfoProto]:
    result = []
    for agent_index in range(n_agent):
        ap = AgentInfoProto()
        ap.reward = float("inf") if infinite_rewards else agent_index
        ap.done = agent_index % 2 == 0
        ap.max_step_reached = agent_index % 4 == 0
        ap.id = agent_index
        ap.action_mask.extend([True, False] * 5)
        obs_proto_list = []
        for obs_index in range(len(shape)):
            obs_proto = ObservationProto()
            obs_proto.shape.extend(list(shape[obs_index]))
            obs_proto.compression_type = NONE
            obs_proto.float_data.data.extend(
                ([float("nan")] if nan_observations else [0.1])
                * np.prod(shape[obs_index])
            )
            obs_proto_list.append(obs_proto)
        ap.observations.extend(obs_proto_list)
        result.append(ap)
    return result


def generate_compressed_data(in_array: np.ndarray) -> bytes:
    image_arr = (in_array * 255).astype(np.uint8)
    bytes_out = bytes()

    num_channels = in_array.shape[2]
    num_images = (num_channels + 2) // 3
    # Split the input image into batches of 3 channels.
    for i in range(num_images):
        sub_image = image_arr[..., 3 * i : 3 * i + 3]
        if (i == num_images - 1) and (num_channels % 3) != 0:
            # Pad zeros
            zero_shape = list(in_array.shape)
            zero_shape[2] = 3 - (num_channels % 3)
            z = np.zeros(zero_shape, dtype=np.uint8)
            sub_image = np.concatenate([sub_image, z], axis=2)
        im = Image.fromarray(sub_image, "RGB")
        byteIO = io.BytesIO()
        im.save(byteIO, format="PNG")
        bytes_out += byteIO.getvalue()
    return bytes_out


def generate_compressed_proto_obs(in_array: np.ndarray) -> ObservationProto:
    obs_proto = ObservationProto()
    obs_proto.compressed_data = generate_compressed_data(in_array)
    obs_proto.compression_type = PNG
    obs_proto.shape.extend(in_array.shape)
    return obs_proto


def generate_uncompressed_proto_obs(in_array: np.ndarray) -> ObservationProto:
    obs_proto = ObservationProto()
    obs_proto.float_data.data.extend(in_array.flatten().tolist())
    obs_proto.compression_type = NONE
    obs_proto.shape.extend(in_array.shape)
    return obs_proto


def proto_from_steps(
    decision_steps: DecisionSteps, terminal_steps: TerminalSteps
) -> List[AgentInfoProto]:
    agent_info_protos: List[AgentInfoProto] = []
    # Take care of the DecisionSteps first
    for agent_id in decision_steps.agent_id:
        agent_id_index = decision_steps.agent_id_to_index[agent_id]
        reward = decision_steps.reward[agent_id_index]
        done = False
        max_step_reached = False
        agent_mask = None
        if decision_steps.action_mask is not None:
            agent_mask = []  # type: ignore
            for _branch in decision_steps.action_mask:
                agent_mask = np.concatenate(
                    (agent_mask, _branch[agent_id_index, :]), axis=0
                )
        observations: List[ObservationProto] = []
        for all_observations_of_type in decision_steps.obs:
            observation = all_observations_of_type[agent_id_index]
            if len(observation.shape) == 3:
                observations.append(generate_uncompressed_proto_obs(observation))
            else:
                observations.append(
                    ObservationProto(
                        float_data=ObservationProto.FloatData(data=observation),
                        shape=[len(observation)],
                        compression_type=NONE,
                    )
                )
        agent_info_proto = AgentInfoProto(
            reward=reward,
            done=done,
            id=agent_id,
            max_step_reached=max_step_reached,
            action_mask=agent_mask,
            observations=observations,
        )
        agent_info_protos.append(agent_info_proto)
    # Take care of the TerminalSteps second
    for agent_id in terminal_steps.agent_id:
        agent_id_index = terminal_steps.agent_id_to_index[agent_id]
        reward = terminal_steps.reward[agent_id_index]
        done = True
        max_step_reached = terminal_steps.interrupted[agent_id_index]

        final_observations: List[ObservationProto] = []
        for all_observations_of_type in terminal_steps.obs:
            observation = all_observations_of_type[agent_id_index]
            if len(observation.shape) == 3:
                final_observations.append(generate_uncompressed_proto_obs(observation))
            else:
                final_observations.append(
                    ObservationProto(
                        float_data=ObservationProto.FloatData(data=observation),
                        shape=[len(observation)],
                        compression_type=NONE,
                    )
                )
        agent_info_proto = AgentInfoProto(
            reward=reward,
            done=done,
            id=agent_id,
            max_step_reached=max_step_reached,
            action_mask=None,
            observations=final_observations,
        )
        agent_info_protos.append(agent_info_proto)

    return agent_info_protos


# The arguments here are the DecisionSteps, TerminalSteps and actions for a single agent name
def proto_from_steps_and_action(
    decision_steps: DecisionSteps, terminal_steps: TerminalSteps, actions: np.ndarray
) -> List[AgentInfoActionPairProto]:
    agent_info_protos = proto_from_steps(decision_steps, terminal_steps)
    agent_action_protos = [
        AgentActionProto(vector_actions=action) for action in actions
    ]
    agent_info_action_pair_protos = [
        AgentInfoActionPairProto(agent_info=agent_info_proto, action_info=action_proto)
        for agent_info_proto, action_proto in zip(
            agent_info_protos, agent_action_protos
        )
    ]
    return agent_info_action_pair_protos


def test_process_pixels():
    in_array = np.random.rand(128, 64, 3)
    byte_arr = generate_compressed_data(in_array)
    out_array = process_pixels(byte_arr, 3)
    assert out_array.shape == (128, 64, 3)
    assert np.sum(in_array - out_array) / np.prod(in_array.shape) < 0.01
    assert np.allclose(in_array, out_array, atol=0.01)


def test_process_pixels_multi_png():
    height = 128
    width = 64
    num_channels = 7
    in_array = np.random.rand(height, width, num_channels)
    byte_arr = generate_compressed_data(in_array)
    out_array = process_pixels(byte_arr, num_channels)
    assert out_array.shape == (height, width, num_channels)
    assert np.sum(in_array - out_array) / np.prod(in_array.shape) < 0.01
    assert np.allclose(in_array, out_array, atol=0.01)


def test_process_pixels_gray():
    in_array = np.random.rand(128, 64, 3)
    byte_arr = generate_compressed_data(in_array)
    out_array = process_pixels(byte_arr, 1)
    assert out_array.shape == (128, 64, 1)
    assert np.mean(in_array.mean(axis=2, keepdims=True) - out_array) < 0.01
    assert np.allclose(in_array.mean(axis=2, keepdims=True), out_array, atol=0.01)


def test_vector_observation():
    n_agents = 10
    shapes = [(3,), (4,)]
    list_proto = generate_list_agent_proto(n_agents, shapes)
    for obs_index, shape in enumerate(shapes):
        arr = _process_vector_observation(obs_index, shape, list_proto)
        assert list(arr.shape) == ([n_agents] + list(shape))
        assert np.allclose(arr, 0.1, atol=0.01)


def test_process_visual_observation():
    in_array_1 = np.random.rand(128, 64, 3)
    proto_obs_1 = generate_compressed_proto_obs(in_array_1)
    in_array_2 = np.random.rand(128, 64, 3)
    proto_obs_2 = generate_uncompressed_proto_obs(in_array_2)
    ap1 = AgentInfoProto()
    ap1.observations.extend([proto_obs_1])
    ap2 = AgentInfoProto()
    ap2.observations.extend([proto_obs_2])
    ap_list = [ap1, ap2]
    arr = _process_visual_observation(0, (128, 64, 3), ap_list)
    assert list(arr.shape) == [2, 128, 64, 3]
    assert np.allclose(arr[0, :, :, :], in_array_1, atol=0.01)
    assert np.allclose(arr[1, :, :, :], in_array_2, atol=0.01)


def test_process_visual_observation_bad_shape():
    in_array_1 = np.random.rand(128, 64, 3)
    proto_obs_1 = generate_compressed_proto_obs(in_array_1)
    ap1 = AgentInfoProto()
    ap1.observations.extend([proto_obs_1])
    ap_list = [ap1]
    with pytest.raises(UnityObservationException):
        _process_visual_observation(0, (128, 42, 3), ap_list)


def test_batched_step_result_from_proto():
    n_agents = 10
    shapes = [(3,), (4,)]
    spec = BehaviorSpec(shapes, ActionType.CONTINUOUS, 3)
    ap_list = generate_list_agent_proto(n_agents, shapes)
    decision_steps, terminal_steps = steps_from_proto(ap_list, spec)
    for agent_id in range(n_agents):
        if agent_id in decision_steps:
            # we set the reward equal to the agent id in generate_list_agent_proto
            assert decision_steps[agent_id].reward == agent_id
        elif agent_id in terminal_steps:
            assert terminal_steps[agent_id].reward == agent_id
        else:
            raise Exception("Missing agent from the steps")
    # We sort the AgentId since they are split between DecisionSteps and TerminalSteps
    combined_agent_id = list(decision_steps.agent_id) + list(terminal_steps.agent_id)
    combined_agent_id.sort()
    assert combined_agent_id == list(range(n_agents))
    for agent_id in range(n_agents):
        assert (agent_id in terminal_steps) == (agent_id % 2 == 0)
        if agent_id in terminal_steps:
            assert terminal_steps[agent_id].interrupted == (agent_id % 4 == 0)
    assert decision_steps.obs[0].shape[1] == shapes[0][0]
    assert decision_steps.obs[1].shape[1] == shapes[1][0]
    assert terminal_steps.obs[0].shape[1] == shapes[0][0]
    assert terminal_steps.obs[1].shape[1] == shapes[1][0]


def test_action_masking_discrete():
    n_agents = 10
    shapes = [(3,), (4,)]
    behavior_spec = BehaviorSpec(shapes, ActionType.DISCRETE, (7, 3))
    ap_list = generate_list_agent_proto(n_agents, shapes)
    decision_steps, terminal_steps = steps_from_proto(ap_list, behavior_spec)
    masks = decision_steps.action_mask
    assert isinstance(masks, list)
    assert len(masks) == 2
    assert masks[0].shape == (n_agents / 2, 7)  # half agents are done
    assert masks[1].shape == (n_agents / 2, 3)  # half agents are done
    assert masks[0][0, 0]
    assert not masks[1][0, 0]
    assert masks[1][0, 1]


def test_action_masking_discrete_1():
    n_agents = 10
    shapes = [(3,), (4,)]
    behavior_spec = BehaviorSpec(shapes, ActionType.DISCRETE, (10,))
    ap_list = generate_list_agent_proto(n_agents, shapes)
    decision_steps, terminal_steps = steps_from_proto(ap_list, behavior_spec)
    masks = decision_steps.action_mask
    assert isinstance(masks, list)
    assert len(masks) == 1
    assert masks[0].shape == (n_agents / 2, 10)
    assert masks[0][0, 0]


def test_action_masking_discrete_2():
    n_agents = 10
    shapes = [(3,), (4,)]
    behavior_spec = BehaviorSpec(shapes, ActionType.DISCRETE, (2, 2, 6))
    ap_list = generate_list_agent_proto(n_agents, shapes)
    decision_steps, terminal_steps = steps_from_proto(ap_list, behavior_spec)
    masks = decision_steps.action_mask
    assert isinstance(masks, list)
    assert len(masks) == 3
    assert masks[0].shape == (n_agents / 2, 2)
    assert masks[1].shape == (n_agents / 2, 2)
    assert masks[2].shape == (n_agents / 2, 6)
    assert masks[0][0, 0]


def test_action_masking_continuous():
    n_agents = 10
    shapes = [(3,), (4,)]
    behavior_spec = BehaviorSpec(shapes, ActionType.CONTINUOUS, 10)
    ap_list = generate_list_agent_proto(n_agents, shapes)
    decision_steps, terminal_steps = steps_from_proto(ap_list, behavior_spec)
    masks = decision_steps.action_mask
    assert masks is None


def test_agent_behavior_spec_from_proto():
    agent_proto = generate_list_agent_proto(1, [(3,), (4,)])[0]
    bp = BrainParametersProto()
    bp.vector_action_size.extend([5, 4])
    bp.vector_action_space_type = 0
    behavior_spec = behavior_spec_from_proto(bp, agent_proto)
    assert behavior_spec.is_action_discrete()
    assert not behavior_spec.is_action_continuous()
    assert behavior_spec.observation_shapes == [(3,), (4,)]
    assert behavior_spec.discrete_action_branches == (5, 4)
    assert behavior_spec.action_size == 2
    bp = BrainParametersProto()
    bp.vector_action_size.extend([6])
    bp.vector_action_space_type = 1
    behavior_spec = behavior_spec_from_proto(bp, agent_proto)
    assert not behavior_spec.is_action_discrete()
    assert behavior_spec.is_action_continuous()
    assert behavior_spec.action_size == 6


def test_batched_step_result_from_proto_raises_on_infinite():
    n_agents = 10
    shapes = [(3,), (4,)]
    behavior_spec = BehaviorSpec(shapes, ActionType.CONTINUOUS, 3)
    ap_list = generate_list_agent_proto(n_agents, shapes, infinite_rewards=True)
    with pytest.raises(RuntimeError):
        steps_from_proto(ap_list, behavior_spec)


def test_batched_step_result_from_proto_raises_on_nan():
    n_agents = 10
    shapes = [(3,), (4,)]
    behavior_spec = BehaviorSpec(shapes, ActionType.CONTINUOUS, 3)
    ap_list = generate_list_agent_proto(n_agents, shapes, nan_observations=True)
    with pytest.raises(RuntimeError):
        steps_from_proto(ap_list, behavior_spec)

import io
import numpy as np
import pytest
from typing import List, Tuple, Any

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
    ActionSpec,
    DecisionSteps,
    TerminalSteps,
)
from mlagents_envs.exception import UnityObservationException
from mlagents_envs.rpc_utils import (
    behavior_spec_from_proto,
    process_pixels,
    _process_maybe_compressed_observation,
    _process_rank_one_or_two_observation,
    steps_from_proto,
)
from PIL import Image
from dummy_config import create_observation_specs_with_shapes


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


# test helper function for old C# API (no compressed channel mapping)
def generate_compressed_proto_obs(
    in_array: np.ndarray, grayscale: bool = False
) -> ObservationProto:
    obs_proto = ObservationProto()
    obs_proto.compressed_data = generate_compressed_data(in_array)
    obs_proto.compression_type = PNG
    if grayscale:
        # grayscale flag is only used for old API without mapping
        expected_shape = [in_array.shape[0], in_array.shape[1], 1]
        obs_proto.shape.extend(expected_shape)
    else:
        obs_proto.shape.extend(in_array.shape)
    return obs_proto


# test helper function for new C# API (with compressed channel mapping)
def generate_compressed_proto_obs_with_mapping(
    in_array: np.ndarray, mapping: List[int]
) -> ObservationProto:
    obs_proto = ObservationProto()
    obs_proto.compressed_data = generate_compressed_data(in_array)
    obs_proto.compression_type = PNG
    if mapping is not None:
        obs_proto.compressed_channel_mapping.extend(mapping)
        expected_shape = [
            in_array.shape[0],
            in_array.shape[1],
            len({m for m in mapping if m >= 0}),
        ]
        obs_proto.shape.extend(expected_shape)
    else:
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
        agent_mask: Any = None
        if decision_steps.action_mask is not None:
            agent_mask = []
            for _branch in decision_steps.action_mask:
                agent_mask = np.concatenate(
                    (agent_mask, _branch[agent_id_index, :]), axis=0
                )
            agent_mask = agent_mask.astype(bool).tolist()
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
            max_step_reached=bool(max_step_reached),
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
            max_step_reached=bool(max_step_reached),
            action_mask=None,
            observations=final_observations,
        )
        agent_info_protos.append(agent_info_proto)

    return agent_info_protos


# The arguments here are the DecisionSteps, TerminalSteps and continuous/discrete actions for a single agent name
def proto_from_steps_and_action(
    decision_steps: DecisionSteps,
    terminal_steps: TerminalSteps,
    continuous_actions: np.ndarray,
    discrete_actions: np.ndarray,
) -> List[AgentInfoActionPairProto]:
    agent_info_protos = proto_from_steps(decision_steps, terminal_steps)
    agent_action_protos = []
    num_agents = (
        len(continuous_actions)
        if continuous_actions is not None
        else len(discrete_actions)
    )
    for i in range(num_agents):
        proto = AgentActionProto()
        if continuous_actions is not None:
            proto.continuous_actions.extend(continuous_actions[i])
            proto.vector_actions_deprecated.extend(continuous_actions[i])
        if discrete_actions is not None:
            proto.discrete_actions.extend(discrete_actions[i])
            proto.vector_actions_deprecated.extend(discrete_actions[i])
        agent_action_protos.append(proto)
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
    obs_specs = create_observation_specs_with_shapes(shapes)
    list_proto = generate_list_agent_proto(n_agents, shapes)
    for obs_index, shape in enumerate(shapes):
        arr = _process_rank_one_or_two_observation(
            obs_index, obs_specs[obs_index], list_proto
        )
        assert list(arr.shape) == ([n_agents] + list(shape))
        assert np.allclose(arr, 0.1, atol=0.01)


def test_process_visual_observation():
    shape = (128, 64, 3)
    in_array_1 = np.random.rand(*shape)
    proto_obs_1 = generate_compressed_proto_obs(in_array_1)
    in_array_2 = np.random.rand(*shape)
    in_array_2_mapping = [0, 1, 2]
    proto_obs_2 = generate_compressed_proto_obs_with_mapping(
        in_array_2, in_array_2_mapping
    )

    ap1 = AgentInfoProto()
    ap1.observations.extend([proto_obs_1])
    ap2 = AgentInfoProto()
    ap2.observations.extend([proto_obs_2])
    ap_list = [ap1, ap2]
    obs_spec = create_observation_specs_with_shapes([shape])[0]
    arr = _process_maybe_compressed_observation(0, obs_spec, ap_list)
    assert list(arr.shape) == [2, 128, 64, 3]
    assert np.allclose(arr[0, :, :, :], in_array_1, atol=0.01)
    assert np.allclose(arr[1, :, :, :], in_array_2, atol=0.01)


def test_process_visual_observation_grayscale():
    in_array_1 = np.random.rand(128, 64, 3)
    proto_obs_1 = generate_compressed_proto_obs(in_array_1, grayscale=True)
    expected_out_array_1 = np.mean(in_array_1, axis=2, keepdims=True)
    in_array_2 = np.random.rand(128, 64, 3)
    in_array_2_mapping = [0, 0, 0]
    proto_obs_2 = generate_compressed_proto_obs_with_mapping(
        in_array_2, in_array_2_mapping
    )
    expected_out_array_2 = np.mean(in_array_2, axis=2, keepdims=True)

    ap1 = AgentInfoProto()
    ap1.observations.extend([proto_obs_1])
    ap2 = AgentInfoProto()
    ap2.observations.extend([proto_obs_2])
    ap_list = [ap1, ap2]
    shape = (128, 64, 1)
    obs_spec = create_observation_specs_with_shapes([shape])[0]
    arr = _process_maybe_compressed_observation(0, obs_spec, ap_list)
    assert list(arr.shape) == [2, 128, 64, 1]
    assert np.allclose(arr[0, :, :, :], expected_out_array_1, atol=0.01)
    assert np.allclose(arr[1, :, :, :], expected_out_array_2, atol=0.01)


def test_process_visual_observation_padded_channels():
    in_array_1 = np.random.rand(128, 64, 12)
    in_array_1_mapping = [0, 1, 2, 3, -1, -1, 4, 5, 6, 7, -1, -1]
    proto_obs_1 = generate_compressed_proto_obs_with_mapping(
        in_array_1, in_array_1_mapping
    )
    expected_out_array_1 = np.take(in_array_1, [0, 1, 2, 3, 6, 7, 8, 9], axis=2)

    ap1 = AgentInfoProto()
    ap1.observations.extend([proto_obs_1])
    ap_list = [ap1]
    shape = (128, 64, 8)
    obs_spec = create_observation_specs_with_shapes([shape])[0]

    arr = _process_maybe_compressed_observation(0, obs_spec, ap_list)
    assert list(arr.shape) == [1, 128, 64, 8]
    assert np.allclose(arr[0, :, :, :], expected_out_array_1, atol=0.01)


def test_process_visual_observation_bad_shape():
    in_array_1 = np.random.rand(128, 64, 3)
    proto_obs_1 = generate_compressed_proto_obs(in_array_1)
    ap1 = AgentInfoProto()
    ap1.observations.extend([proto_obs_1])
    ap_list = [ap1]

    shape = (128, 42, 3)
    obs_spec = create_observation_specs_with_shapes([shape])[0]

    with pytest.raises(UnityObservationException):
        _process_maybe_compressed_observation(0, obs_spec, ap_list)


def test_batched_step_result_from_proto():
    n_agents = 10
    shapes = [(3,), (4,)]
    spec = BehaviorSpec(
        create_observation_specs_with_shapes(shapes), ActionSpec.create_continuous(3)
    )
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


def test_mismatch_observations_raise_in_step_result_from_proto():
    n_agents = 10
    shapes = [(3,), (4,)]
    spec = BehaviorSpec(
        create_observation_specs_with_shapes(shapes), ActionSpec.create_continuous(3)
    )
    ap_list = generate_list_agent_proto(n_agents, shapes)
    # Hack an observation to be larger, we should get an exception
    ap_list[0].observations[0].shape[0] += 1
    ap_list[0].observations[0].float_data.data.append(0.42)
    with pytest.raises(UnityObservationException):
        steps_from_proto(ap_list, spec)


def test_action_masking_discrete():
    n_agents = 10
    shapes = [(3,), (4,)]
    behavior_spec = BehaviorSpec(
        create_observation_specs_with_shapes(shapes), ActionSpec.create_discrete((7, 3))
    )
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
    behavior_spec = BehaviorSpec(
        create_observation_specs_with_shapes(shapes), ActionSpec.create_discrete((10,))
    )
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
    behavior_spec = BehaviorSpec(
        create_observation_specs_with_shapes(shapes),
        ActionSpec.create_discrete((2, 2, 6)),
    )
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
    behavior_spec = BehaviorSpec(
        create_observation_specs_with_shapes(shapes), ActionSpec.create_continuous(10)
    )
    ap_list = generate_list_agent_proto(n_agents, shapes)
    decision_steps, terminal_steps = steps_from_proto(ap_list, behavior_spec)
    masks = decision_steps.action_mask
    assert masks is None


def test_agent_behavior_spec_from_proto():
    agent_proto = generate_list_agent_proto(1, [(3,), (4,)])[0]
    bp = BrainParametersProto()
    bp.vector_action_size_deprecated.extend([5, 4])
    bp.vector_action_space_type_deprecated = 0
    behavior_spec = behavior_spec_from_proto(bp, agent_proto)
    assert behavior_spec.action_spec.is_discrete()
    assert not behavior_spec.action_spec.is_continuous()
    assert [spec.shape for spec in behavior_spec.observation_specs] == [(3,), (4,)]
    assert behavior_spec.action_spec.discrete_branches == (5, 4)
    assert behavior_spec.action_spec.discrete_size == 2
    bp = BrainParametersProto()
    bp.vector_action_size_deprecated.extend([6])
    bp.vector_action_space_type_deprecated = 1
    behavior_spec = behavior_spec_from_proto(bp, agent_proto)
    assert not behavior_spec.action_spec.is_discrete()
    assert behavior_spec.action_spec.is_continuous()
    assert behavior_spec.action_spec.continuous_size == 6


def test_batched_step_result_from_proto_raises_on_infinite():
    n_agents = 10
    shapes = [(3,), (4,)]
    behavior_spec = BehaviorSpec(
        create_observation_specs_with_shapes(shapes), ActionSpec.create_continuous(3)
    )
    ap_list = generate_list_agent_proto(n_agents, shapes, infinite_rewards=True)
    with pytest.raises(RuntimeError):
        steps_from_proto(ap_list, behavior_spec)


def test_batched_step_result_from_proto_raises_on_nan():
    n_agents = 10
    shapes = [(3,), (4,)]
    behavior_spec = BehaviorSpec(
        create_observation_specs_with_shapes(shapes), ActionSpec.create_continuous(3)
    )
    ap_list = generate_list_agent_proto(n_agents, shapes, nan_observations=True)
    with pytest.raises(RuntimeError):
        steps_from_proto(ap_list, behavior_spec)

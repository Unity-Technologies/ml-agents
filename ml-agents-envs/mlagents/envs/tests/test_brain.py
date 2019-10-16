import logging
import numpy as np
import sys
from unittest import mock

from mlagents.envs.communicator_objects.agent_info_pb2 import AgentInfoProto
from mlagents.envs.brain import BrainInfo, BrainParameters

test_brain = BrainParameters(
    brain_name="test_brain",
    vector_observation_space_size=3,
    num_stacked_vector_observations=1,
    camera_resolutions=[],
    vector_action_space_size=[],
    vector_action_descriptions=[],
    vector_action_space_type=1,
)


@mock.patch.object(np, "nan_to_num", wraps=np.nan_to_num)
@mock.patch.object(logging.Logger, "warning")
def test_from_agent_proto_nan(mock_warning, mock_nan_to_num):
    agent_info_proto = AgentInfoProto()
    agent_info_proto.stacked_vector_observation.extend([1.0, 2.0, float("nan")])

    brain_info = BrainInfo.from_agent_proto(1, [agent_info_proto], test_brain)
    # nan gets set to 0.0
    expected = [1.0, 2.0, 0.0]
    assert (brain_info.vector_observations == expected).all()
    mock_nan_to_num.assert_called()
    mock_warning.assert_called()


@mock.patch.object(np, "nan_to_num", wraps=np.nan_to_num)
@mock.patch.object(logging.Logger, "warning")
def test_from_agent_proto_inf(mock_warning, mock_nan_to_num):
    agent_info_proto = AgentInfoProto()
    agent_info_proto.stacked_vector_observation.extend([1.0, float("inf"), 0.0])

    brain_info = BrainInfo.from_agent_proto(1, [agent_info_proto], test_brain)
    # inf should get set to float_max
    expected = [1.0, sys.float_info.max, 0.0]
    assert (brain_info.vector_observations == expected).all()
    mock_nan_to_num.assert_called()
    # We don't warn on inf, just NaN
    mock_warning.assert_not_called()


@mock.patch.object(np, "nan_to_num", wraps=np.nan_to_num)
@mock.patch.object(logging.Logger, "warning")
def test_from_agent_proto_fast_path(mock_warning, mock_nan_to_num):
    """
    Check that all finite values skips the nan_to_num call
    """
    agent_info_proto = AgentInfoProto()
    agent_info_proto.stacked_vector_observation.extend([1.0, 2.0, 3.0])

    brain_info = BrainInfo.from_agent_proto(1, [agent_info_proto], test_brain)
    expected = [1.0, 2.0, 3.0]
    assert (brain_info.vector_observations == expected).all()
    mock_nan_to_num.assert_not_called()
    mock_warning.assert_not_called()

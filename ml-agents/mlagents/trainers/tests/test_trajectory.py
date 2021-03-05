import numpy as np

from mlagents.trainers.tests.mock_brain import make_fake_trajectory
from mlagents.trainers.tests.dummy_config import create_observation_specs_with_shapes
from mlagents.trainers.trajectory import GroupObsUtil
from mlagents_envs.base_env import ActionSpec
from mlagents.trainers.buffer import AgentBuffer, BufferKey, ObservationKeyPrefix

VEC_OBS_SIZE = 6
ACTION_SIZE = 4


def test_trajectory_to_agentbuffer():
    length = 15
    # These keys should be of type np.ndarray
    wanted_keys = [
        (ObservationKeyPrefix.OBSERVATION, 0),
        (ObservationKeyPrefix.OBSERVATION, 1),
        (ObservationKeyPrefix.NEXT_OBSERVATION, 0),
        (ObservationKeyPrefix.NEXT_OBSERVATION, 1),
        BufferKey.MEMORY,
        BufferKey.MASKS,
        BufferKey.DONE,
        BufferKey.CONTINUOUS_ACTION,
        BufferKey.DISCRETE_ACTION,
        BufferKey.CONTINUOUS_LOG_PROBS,
        BufferKey.DISCRETE_LOG_PROBS,
        BufferKey.ACTION_MASK,
        BufferKey.PREV_ACTION,
        BufferKey.ENVIRONMENT_REWARDS,
        BufferKey.GROUP_REWARD,
    ]
    # These keys should be of type List
    wanted_group_keys = [
        BufferKey.GROUPMATE_REWARDS,
        BufferKey.GROUP_CONTINUOUS_ACTION,
        BufferKey.GROUP_DISCRETE_ACTION,
        BufferKey.GROUP_DONES,
        BufferKey.GROUP_NEXT_CONT_ACTION,
        BufferKey.GROUP_NEXT_DISC_ACTION,
    ]
    wanted_keys = set(wanted_keys + wanted_group_keys)
    trajectory = make_fake_trajectory(
        length=length,
        observation_specs=create_observation_specs_with_shapes(
            [(VEC_OBS_SIZE,), (84, 84, 3)]
        ),
        action_spec=ActionSpec.create_continuous(ACTION_SIZE),
        num_other_agents_in_group=4,
    )
    agentbuffer = trajectory.to_agentbuffer()
    seen_keys = set()
    for key, field in agentbuffer.items():
        assert len(field) == length
        seen_keys.add(key)

    assert seen_keys.issuperset(wanted_keys)

    for _key in wanted_group_keys:
        for step in agentbuffer[_key]:
            assert len(step) == 4


def test_obsutil_group_from_buffer():
    buff = AgentBuffer()
    # Create some obs
    for _ in range(3):
        buff[GroupObsUtil.get_name_at(0)].append(3 * [np.ones((5,), dtype=np.float32)])
    # Some agents have died
    for _ in range(2):
        buff[GroupObsUtil.get_name_at(0)].append(1 * [np.ones((5,), dtype=np.float32)])

    # Get the group obs, which will be a List of Lists of np.ndarray, where each element is the same
    # length as the AgentBuffer but contains only one agent's obs. Dead agents are padded by
    # NaNs.
    gobs = GroupObsUtil.from_buffer(buff, 1)
    # Agent 0 is full
    agent_0_obs = gobs[0]
    for obs in agent_0_obs:
        assert obs.shape == (buff.num_experiences, 5)
        assert not np.isnan(obs).any()

    agent_1_obs = gobs[1]
    for obs in agent_1_obs:
        assert obs.shape == (buff.num_experiences, 5)
        for i, _exp_obs in enumerate(obs):
            if i >= 3:
                assert np.isnan(_exp_obs).all()
            else:
                assert not np.isnan(_exp_obs).any()

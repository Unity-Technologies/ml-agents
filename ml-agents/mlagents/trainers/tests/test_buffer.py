import numpy as np
from mlagents.trainers.buffer import (
    AgentBuffer,
    AgentBufferField,
    BufferKey,
    ObservationKeyPrefix,
    RewardSignalKeyPrefix,
)
from mlagents.trainers.trajectory import ObsUtil


def assert_array(a, b):
    assert a.shape == b.shape
    la = list(a.flatten())
    lb = list(b.flatten())
    for i in range(len(la)):
        assert la[i] == lb[i]


def construct_fake_buffer(fake_agent_id):
    b = AgentBuffer()
    for step in range(9):
        b[ObsUtil.get_name_at(0)].append(
            np.array(
                [
                    100 * fake_agent_id + 10 * step + 1,
                    100 * fake_agent_id + 10 * step + 2,
                    100 * fake_agent_id + 10 * step + 3,
                ],
                dtype=np.float32,
            )
        )
        b[BufferKey.CONTINUOUS_ACTION].append(
            np.array(
                [
                    100 * fake_agent_id + 10 * step + 4,
                    100 * fake_agent_id + 10 * step + 5,
                ],
                dtype=np.float32,
            )
        )
        b[BufferKey.GROUP_CONTINUOUS_ACTION].append(
            [
                np.array(
                    [
                        100 * fake_agent_id + 10 * step + 4,
                        100 * fake_agent_id + 10 * step + 5,
                    ],
                    dtype=np.float32,
                )
            ]
            * 3
        )
    return b


def test_buffer():
    agent_1_buffer = construct_fake_buffer(1)
    agent_2_buffer = construct_fake_buffer(2)
    agent_3_buffer = construct_fake_buffer(3)

    # Test get_batch
    a = agent_1_buffer[ObsUtil.get_name_at(0)].get_batch(
        batch_size=2, training_length=1, sequential=True
    )
    assert_array(
        np.array(a), np.array([[171, 172, 173], [181, 182, 183]], dtype=np.float32)
    )

    # Test get_batch
    a = agent_2_buffer[ObsUtil.get_name_at(0)].get_batch(
        batch_size=2, training_length=3, sequential=True
    )
    assert_array(
        np.array(a),
        np.array(
            [
                [231, 232, 233],
                [241, 242, 243],
                [251, 252, 253],
                [261, 262, 263],
                [271, 272, 273],
                [281, 282, 283],
            ],
            dtype=np.float32,
        ),
    )
    a = agent_2_buffer[ObsUtil.get_name_at(0)].get_batch(
        batch_size=2, training_length=3, sequential=False
    )
    assert_array(
        np.array(a),
        np.array(
            [
                [251, 252, 253],
                [261, 262, 263],
                [271, 272, 273],
                [261, 262, 263],
                [271, 272, 273],
                [281, 282, 283],
            ]
        ),
    )

    # Test padding
    a = agent_2_buffer[ObsUtil.get_name_at(0)].get_batch(
        batch_size=None, training_length=4, sequential=True
    )
    assert_array(
        np.array(a),
        np.array(
            [
                [201, 202, 203],
                [211, 212, 213],
                [221, 222, 223],
                [231, 232, 233],
                [241, 242, 243],
                [251, 252, 253],
                [261, 262, 263],
                [271, 272, 273],
                [281, 282, 283],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        ),
    )
    # Test group entries return Lists of Lists. Make sure to pad properly!
    a = agent_2_buffer[BufferKey.GROUP_CONTINUOUS_ACTION].get_batch(
        batch_size=None, training_length=4, sequential=True
    )
    for _group_entry in a[:-3]:
        assert len(_group_entry) == 3
    for _group_entry in a[-3:]:
        assert len(_group_entry) == 0

    agent_1_buffer.reset_agent()
    assert agent_1_buffer.num_experiences == 0
    update_buffer = AgentBuffer()
    agent_2_buffer.resequence_and_append(
        update_buffer, batch_size=None, training_length=2
    )
    agent_3_buffer.resequence_and_append(
        update_buffer, batch_size=None, training_length=2
    )
    assert len(update_buffer[BufferKey.CONTINUOUS_ACTION]) == 20

    assert np.array(update_buffer[BufferKey.CONTINUOUS_ACTION]).shape == (20, 2)

    c = update_buffer.make_mini_batch(start=0, end=1)
    assert c.keys() == update_buffer.keys()
    # Make sure the values of c are AgentBufferField
    for val in c.values():
        assert isinstance(val, AgentBufferField)
    assert np.array(c[BufferKey.CONTINUOUS_ACTION]).shape == (1, 2)


def test_agentbufferfield():
    # Test constructor
    a = AgentBufferField([0, 1, 2])
    for i, num in enumerate(a):
        assert num == i
        # Test indexing
        assert a[i] == num

    # Test slicing
    b = a[1:3]
    assert b == [1, 2]
    assert isinstance(b, AgentBufferField)

    # Test padding
    c = AgentBufferField()
    for _ in range(2):
        c.append([np.array(1), np.array(2)])

    for _ in range(2):
        c.append([np.array(1)])

    padded = c.padded_to_batch(pad_value=3)
    assert np.array_equal(padded[0], np.array([1, 1, 1, 1]))
    assert np.array_equal(padded[1], np.array([2, 2, 3, 3]))

    # Make sure it doesn't fail when the field isn't a list
    padded_a = a.padded_to_batch()
    assert np.array_equal(padded_a, a)


def fakerandint(values):
    return 19


def test_buffer_sample():
    agent_1_buffer = construct_fake_buffer(1)
    agent_2_buffer = construct_fake_buffer(2)
    update_buffer = AgentBuffer()
    agent_1_buffer.resequence_and_append(
        update_buffer, batch_size=None, training_length=2
    )
    agent_2_buffer.resequence_and_append(
        update_buffer, batch_size=None, training_length=2
    )
    # Test non-LSTM
    mb = update_buffer.sample_mini_batch(batch_size=4, sequence_length=1)
    assert mb.keys() == update_buffer.keys()
    assert np.array(mb[BufferKey.CONTINUOUS_ACTION]).shape == (4, 2)

    # Test LSTM
    # We need to check if we ever get a breaking start - this will maximize the probability
    mb = update_buffer.sample_mini_batch(batch_size=20, sequence_length=19)
    assert mb.keys() == update_buffer.keys()
    # Should only return one sequence
    assert np.array(mb[BufferKey.CONTINUOUS_ACTION]).shape == (19, 2)


def test_num_experiences():
    agent_1_buffer = construct_fake_buffer(1)
    agent_2_buffer = construct_fake_buffer(2)
    update_buffer = AgentBuffer()

    assert len(update_buffer[BufferKey.CONTINUOUS_ACTION]) == 0
    assert update_buffer.num_experiences == 0
    agent_1_buffer.resequence_and_append(
        update_buffer, batch_size=None, training_length=2
    )
    agent_2_buffer.resequence_and_append(
        update_buffer, batch_size=None, training_length=2
    )

    assert len(update_buffer[BufferKey.CONTINUOUS_ACTION]) == 20
    assert update_buffer.num_experiences == 20


def test_buffer_truncate():
    agent_1_buffer = construct_fake_buffer(1)
    agent_2_buffer = construct_fake_buffer(2)
    update_buffer = AgentBuffer()
    agent_1_buffer.resequence_and_append(
        update_buffer, batch_size=None, training_length=2
    )
    agent_2_buffer.resequence_and_append(
        update_buffer, batch_size=None, training_length=2
    )
    # Test non-LSTM
    update_buffer.truncate(2)
    assert update_buffer.num_experiences == 2

    agent_1_buffer.resequence_and_append(
        update_buffer, batch_size=None, training_length=2
    )
    agent_2_buffer.resequence_and_append(
        update_buffer, batch_size=None, training_length=2
    )
    # Test LSTM, truncate should be some multiple of sequence_length
    update_buffer.truncate(4, sequence_length=3)
    assert update_buffer.num_experiences == 3
    for buffer_field in update_buffer.values():
        assert isinstance(buffer_field, AgentBufferField)


def test_key_encode_decode():
    keys = (
        list(BufferKey)
        + [(k, 42) for k in ObservationKeyPrefix]
        + [(k, "gail") for k in RewardSignalKeyPrefix]
    )
    for k in keys:
        assert k == AgentBuffer._decode_key(AgentBuffer._encode_key(k))


def test_buffer_save_load():
    original = construct_fake_buffer(3)
    import io

    write_buffer = io.BytesIO()
    original.save_to_file(write_buffer)

    loaded = AgentBuffer()
    loaded.load_from_file(write_buffer)

    assert len(original) == len(loaded)
    for k in original.keys():
        assert np.allclose(original[k], loaded[k])

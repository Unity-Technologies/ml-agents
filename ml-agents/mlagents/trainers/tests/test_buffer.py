import numpy as np
from mlagents.trainers.buffer import AgentBuffer


def assert_array(a, b):
    assert a.shape == b.shape
    la = list(a.flatten())
    lb = list(b.flatten())
    for i in range(len(la)):
        assert la[i] == lb[i]


def construct_fake_buffer(fake_agent_id):
    b = AgentBuffer()
    for step in range(9):
        b["vector_observation"].append(
            [
                100 * fake_agent_id + 10 * step + 1,
                100 * fake_agent_id + 10 * step + 2,
                100 * fake_agent_id + 10 * step + 3,
            ]
        )
        b["action"].append(
            [100 * fake_agent_id + 10 * step + 4, 100 * fake_agent_id + 10 * step + 5]
        )
    return b


def test_buffer():
    agent_1_buffer = construct_fake_buffer(1)
    agent_2_buffer = construct_fake_buffer(2)
    agent_3_buffer = construct_fake_buffer(3)
    a = agent_1_buffer["vector_observation"].get_batch(
        batch_size=2, training_length=1, sequential=True
    )
    assert_array(np.array(a), np.array([[171, 172, 173], [181, 182, 183]]))
    a = agent_2_buffer["vector_observation"].get_batch(
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
            ]
        ),
    )
    a = agent_2_buffer["vector_observation"].get_batch(
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
    agent_1_buffer.reset_agent()
    assert agent_1_buffer.num_experiences == 0
    update_buffer = AgentBuffer()
    agent_2_buffer.resequence_and_append(
        update_buffer, batch_size=None, training_length=2
    )
    agent_3_buffer.resequence_and_append(
        update_buffer, batch_size=None, training_length=2
    )
    assert len(update_buffer["action"]) == 20

    assert np.array(update_buffer["action"]).shape == (20, 2)

    c = update_buffer.make_mini_batch(start=0, end=1)
    assert c.keys() == update_buffer.keys()
    assert np.array(c["action"]).shape == (1, 2)


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
    assert np.array(mb["action"]).shape == (4, 2)

    # Test LSTM
    # We need to check if we ever get a breaking start - this will maximize the probability
    mb = update_buffer.sample_mini_batch(batch_size=20, sequence_length=19)
    assert mb.keys() == update_buffer.keys()
    # Should only return one sequence
    assert np.array(mb["action"]).shape == (19, 2)


def test_num_experiences():
    agent_1_buffer = construct_fake_buffer(1)
    agent_2_buffer = construct_fake_buffer(2)
    update_buffer = AgentBuffer()

    assert len(update_buffer["action"]) == 0
    assert update_buffer.num_experiences == 0
    agent_1_buffer.resequence_and_append(
        update_buffer, batch_size=None, training_length=2
    )
    agent_2_buffer.resequence_and_append(
        update_buffer, batch_size=None, training_length=2
    )

    assert len(update_buffer["action"]) == 20
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
        assert isinstance(buffer_field, AgentBuffer.AgentBufferField)

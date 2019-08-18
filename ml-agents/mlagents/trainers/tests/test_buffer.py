import numpy as np
from mlagents.trainers.buffer import Buffer


def assert_array(a, b):
    assert a.shape == b.shape
    la = list(a.flatten())
    lb = list(b.flatten())
    for i in range(len(la)):
        assert la[i] == lb[i]


def test_buffer():
    b = Buffer()
    for fake_agent_id in range(4):
        for step in range(9):
            b[fake_agent_id]["vector_observation"].append(
                [
                    100 * fake_agent_id + 10 * step + 1,
                    100 * fake_agent_id + 10 * step + 2,
                    100 * fake_agent_id + 10 * step + 3,
                ]
            )
            b[fake_agent_id]["action"].append(
                [
                    100 * fake_agent_id + 10 * step + 4,
                    100 * fake_agent_id + 10 * step + 5,
                ]
            )
    a = b[1]["vector_observation"].get_batch(
        batch_size=2, training_length=1, sequential=True
    )
    assert_array(a, np.array([[171, 172, 173], [181, 182, 183]]))
    a = b[2]["vector_observation"].get_batch(
        batch_size=2, training_length=3, sequential=True
    )
    assert_array(
        a,
        np.array(
            [
                [[231, 232, 233], [241, 242, 243], [251, 252, 253]],
                [[261, 262, 263], [271, 272, 273], [281, 282, 283]],
            ]
        ),
    )
    a = b[2]["vector_observation"].get_batch(
        batch_size=2, training_length=3, sequential=False
    )
    assert_array(
        a,
        np.array(
            [
                [[251, 252, 253], [261, 262, 263], [271, 272, 273]],
                [[261, 262, 263], [271, 272, 273], [281, 282, 283]],
            ]
        ),
    )
    b[4].reset_agent()
    assert len(b[4]) == 0
    b.append_update_buffer(3, batch_size=None, training_length=2)
    b.append_update_buffer(2, batch_size=None, training_length=2)
    assert len(b.update_buffer["action"]) == 10
    assert np.array(b.update_buffer["action"]).shape == (10, 2, 2)

    c = b.update_buffer.make_mini_batch(start=0, end=1)
    assert c.keys() == b.update_buffer.keys()
    assert c["action"].shape == (1, 2, 2)

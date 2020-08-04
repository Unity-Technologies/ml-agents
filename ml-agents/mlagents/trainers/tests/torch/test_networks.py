import pytest

import torch
from mlagents.trainers.torch.networks import NetworkBody, ValueNetwork, Actor
from mlagents.trainers.settings import NetworkSettings
from mlagents_envs.base_env import ActionType
from mlagents.trainers.torch.distributions import (
    GaussianDistInstance,
    CategoricalDistInstance,
)


def test_networkbody_vector():
    obs_size = 4
    network_settings = NetworkSettings()
    obs_shapes = [(obs_size,)]

    networkbody = NetworkBody(obs_shapes, network_settings, encoded_act_size=2)
    optimizer = torch.optim.Adam(networkbody.parameters(), lr=3e-3)
    sample_obs = torch.ones((1, obs_size))
    sample_act = torch.ones((1, 2))

    for _ in range(100):
        encoded, _ = networkbody([sample_obs], [], sample_act)
        assert encoded.shape == (1, network_settings.hidden_units)
        # Try to force output to 1
        loss = torch.nn.functional.mse_loss(encoded, torch.ones(encoded.shape))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # In the last step, values should be close to 1
    for _enc in encoded.flatten():
        assert _enc == pytest.approx(1.0, abs=0.1)


def test_networkbody_lstm():
    obs_size = 4
    seq_len = 16
    network_settings = NetworkSettings(
        memory=NetworkSettings.MemorySettings(sequence_length=seq_len, memory_size=4)
    )
    obs_shapes = [(obs_size,)]

    networkbody = NetworkBody(obs_shapes, network_settings)
    optimizer = torch.optim.Adam(networkbody.parameters(), lr=3e-3)
    sample_obs = torch.ones((1, seq_len, obs_size))

    for _ in range(100):
        encoded, _ = networkbody([sample_obs], [], memories=torch.ones(1, seq_len, 4))
        # Try to force output to 1
        loss = torch.nn.functional.mse_loss(encoded, torch.ones(encoded.shape))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # In the last step, values should be close to 1
    for _enc in encoded.flatten():
        assert _enc == pytest.approx(1.0, abs=0.1)


def test_networkbody_visual():
    vec_obs_size = 4
    obs_size = (84, 84, 3)
    network_settings = NetworkSettings()
    obs_shapes = [(vec_obs_size,), obs_size]
    torch.random.manual_seed(0)

    networkbody = NetworkBody(obs_shapes, network_settings)
    optimizer = torch.optim.Adam(networkbody.parameters(), lr=3e-3)
    sample_obs = torch.ones((1, 84, 84, 3))
    sample_vec_obs = torch.ones((1, vec_obs_size))

    for _ in range(100):
        encoded, _ = networkbody([sample_vec_obs], [sample_obs])
        assert encoded.shape == (1, network_settings.hidden_units)
        # Try to force output to 1
        loss = torch.nn.functional.mse_loss(encoded, torch.ones(encoded.shape))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # In the last step, values should be close to 1
    for _enc in encoded.flatten():
        assert _enc == pytest.approx(1.0, abs=0.1)


def test_valuenetwork():
    obs_size = 4
    num_outputs = 2
    network_settings = NetworkSettings()
    obs_shapes = [(obs_size,)]

    stream_names = [f"stream_name{n}" for n in range(4)]
    value_net = ValueNetwork(
        stream_names, obs_shapes, network_settings, outputs_per_stream=num_outputs
    )
    optimizer = torch.optim.Adam(value_net.parameters(), lr=3e-3)

    for _ in range(50):
        sample_obs = torch.ones((1, obs_size))
        values, _ = value_net([sample_obs], [])
        loss = 0
        for s_name in stream_names:
            assert values[s_name].shape == (1, num_outputs)
            # Try to force output to 1
            loss += torch.nn.functional.mse_loss(
                values[s_name], torch.ones((1, num_outputs))
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # In the last step, values should be close to 1
    for value in values.values():
        for _out in value:
            assert _out[0] == pytest.approx(1.0, abs=0.1)


@pytest.mark.parametrize("action_type", [ActionType.DISCRETE, ActionType.CONTINUOUS])
def test_actor(action_type):
    obs_size = 4
    network_settings = NetworkSettings()
    obs_shapes = [(obs_size,)]
    act_size = [2]
    masks = None if action_type == ActionType.CONTINUOUS else torch.ones((1, 1))
    actor = Actor(obs_shapes, network_settings, action_type, act_size)
    # Test get_dist
    sample_obs = torch.ones((1, obs_size))
    dists, _ = actor.get_dists([sample_obs], [], masks=masks)
    for dist in dists:
        if action_type == ActionType.CONTINUOUS:
            assert isinstance(dist, GaussianDistInstance)
        else:
            assert isinstance(dist, CategoricalDistInstance)

    # Test sample_actions
    actions = actor.sample_action(dists)
    for act in actions:
        if action_type == ActionType.CONTINUOUS:
            assert act.shape == (1, act_size[0])
        else:
            assert act.shape == (1, 1)

    # Test forward
    actions, probs, ver_num, mem_size, is_cont, act_size_vec = actor.forward(
        [sample_obs], [], masks=masks
    )
    for act in actions:
        if action_type == ActionType.CONTINUOUS:
            assert act.shape == (
                act_size[0],
                1,
            )  # This is different from above for ONNX export
        else:
            assert act.shape == (1, 1)

    # TODO: Once export works properly. fix the shapes here.
    assert mem_size == 0
    assert is_cont == int(action_type == ActionType.CONTINUOUS)
    assert act_size_vec == torch.tensor(act_size)

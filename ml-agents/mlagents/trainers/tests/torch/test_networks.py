import pytest

from mlagents.torch_utils import torch
from mlagents.trainers.torch.networks import (
    NetworkBody,
    ValueNetwork,
    SharedActorCritic,
    SeparateActorCritic,
)
from mlagents.trainers.settings import NetworkSettings
from mlagents_envs.base_env import ActionSpec
from mlagents.trainers.tests.torch.test_encoders import compare_models


def test_networkbody_vector():
    torch.manual_seed(0)
    obs_size = 4
    network_settings = NetworkSettings()
    obs_shapes = [(obs_size,)]

    networkbody = NetworkBody(obs_shapes, network_settings, encoded_act_size=2)
    optimizer = torch.optim.Adam(networkbody.parameters(), lr=3e-3)
    sample_obs = 0.1 * torch.ones((1, obs_size))
    sample_act = 0.1 * torch.ones((1, 2))

    for _ in range(300):
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
    torch.manual_seed(0)
    obs_size = 4
    seq_len = 16
    network_settings = NetworkSettings(
        memory=NetworkSettings.MemorySettings(sequence_length=seq_len, memory_size=12)
    )
    obs_shapes = [(obs_size,)]

    networkbody = NetworkBody(obs_shapes, network_settings)
    optimizer = torch.optim.Adam(networkbody.parameters(), lr=3e-4)
    sample_obs = torch.ones((1, seq_len, obs_size))

    for _ in range(200):
        encoded, _ = networkbody([sample_obs], [], memories=torch.ones(1, seq_len, 12))
        # Try to force output to 1
        loss = torch.nn.functional.mse_loss(encoded, torch.ones(encoded.shape))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # In the last step, values should be close to 1
    for _enc in encoded.flatten():
        assert _enc == pytest.approx(1.0, abs=0.1)


def test_networkbody_visual():
    torch.manual_seed(0)
    vec_obs_size = 4
    obs_size = (84, 84, 3)
    network_settings = NetworkSettings()
    obs_shapes = [(vec_obs_size,), obs_size]

    networkbody = NetworkBody(obs_shapes, network_settings)
    optimizer = torch.optim.Adam(networkbody.parameters(), lr=3e-3)
    sample_obs = 0.1 * torch.ones((1, 84, 84, 3))
    sample_vec_obs = torch.ones((1, vec_obs_size))

    for _ in range(150):
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
    torch.manual_seed(0)
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


@pytest.mark.parametrize("ac_type", [SharedActorCritic, SeparateActorCritic])
@pytest.mark.parametrize("lstm", [True, False])
def test_actor_critic(ac_type, lstm):
    obs_size = 4
    network_settings = NetworkSettings(
        memory=NetworkSettings.MemorySettings() if lstm else None, normalize=True
    )
    obs_shapes = [(obs_size,)]
    act_size = 2
    mask = torch.ones([1, act_size * 2])
    stream_names = [f"stream_name{n}" for n in range(4)]
    # action_spec = ActionSpec.create_continuous(act_size[0])
    action_spec = ActionSpec(act_size, tuple(act_size for _ in range(act_size)))
    actor = ac_type(obs_shapes, network_settings, action_spec, stream_names)
    if lstm:
        sample_obs = torch.ones((1, network_settings.memory.sequence_length, obs_size))
        memories = torch.ones(
            (1, network_settings.memory.sequence_length, actor.memory_size)
        )
    else:
        sample_obs = torch.ones((1, obs_size))
        memories = torch.tensor([])
        # memories isn't always set to None, the network should be able to
        # deal with that.
    # Test critic pass
    value_out, memories_out = actor.critic_pass([sample_obs], [], memories=memories)
    for stream in stream_names:
        if lstm:
            assert value_out[stream].shape == (network_settings.memory.sequence_length,)
            assert memories_out.shape == memories.shape
        else:
            assert value_out[stream].shape == (1,)

    # Test get action stats and_value
    action, log_probs, entropies, value_out, mem_out = actor.get_action_stats_and_value(
        [sample_obs], [], memories=memories, masks=mask
    )
    if lstm:
        assert action.continuous_tensor.shape == (64, 2)
    else:
        assert action.continuous_tensor.shape == (1, 2)

    assert len(action.discrete_list) == 2
    for _disc in action.discrete_list:
        if lstm:
            assert _disc.shape == (64, 1)
        else:
            assert _disc.shape == (1, 1)

    if mem_out is not None:
        assert mem_out.shape == memories.shape
    for stream in stream_names:
        if lstm:
            assert value_out[stream].shape == (network_settings.memory.sequence_length,)
        else:
            assert value_out[stream].shape == (1,)

    # Test normalization
    actor.update_normalization(sample_obs)
    if isinstance(actor, SeparateActorCritic):
        for act_proc, crit_proc in zip(
            actor.network_body.vector_processors,
            actor.critic.network_body.vector_processors,
        ):
            assert compare_models(act_proc, crit_proc)

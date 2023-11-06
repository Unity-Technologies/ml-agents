import pytest

from mlagents.torch_utils import torch
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents.trainers.torch_entities.networks import (
    NetworkBody,
    MultiAgentNetworkBody,
    ValueNetwork,
    SimpleActor,
    SharedActorCritic,
)
from mlagents.trainers.settings import NetworkSettings
from mlagents_envs.base_env import ActionSpec
from mlagents.trainers.tests.dummy_config import create_observation_specs_with_shapes


def test_networkbody_vector():
    torch.manual_seed(0)
    obs_size = 4
    network_settings = NetworkSettings()
    obs_shapes = [(obs_size,)]

    networkbody = NetworkBody(
        create_observation_specs_with_shapes(obs_shapes),
        network_settings,
        encoded_act_size=2,
    )
    optimizer = torch.optim.Adam(networkbody.parameters(), lr=3e-3)
    sample_obs = 0.1 * torch.ones((1, obs_size))
    sample_act = 0.1 * torch.ones((1, 2))

    for _ in range(300):
        encoded, _ = networkbody([sample_obs], sample_act)
        assert encoded.shape == (1, network_settings.hidden_units)
        # Try to force output to 1
        loss = torch.nn.functional.mse_loss(encoded, torch.ones(encoded.shape))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # In the last step, values should be close to 1
    for _enc in encoded.flatten().tolist():
        assert _enc == pytest.approx(1.0, abs=0.1)


def test_networkbody_lstm():
    torch.manual_seed(0)
    obs_size = 4
    seq_len = 6
    network_settings = NetworkSettings(
        memory=NetworkSettings.MemorySettings(sequence_length=seq_len, memory_size=12)
    )
    obs_shapes = [(obs_size,)]

    networkbody = NetworkBody(
        create_observation_specs_with_shapes(obs_shapes), network_settings
    )
    optimizer = torch.optim.Adam(networkbody.parameters(), lr=3e-4)
    sample_obs = torch.ones((seq_len, obs_size))

    for _ in range(300):
        encoded, _ = networkbody(
            [sample_obs], memories=torch.ones(1, 1, 12), sequence_length=seq_len
        )
        # Try to force output to 1
        loss = torch.nn.functional.mse_loss(encoded, torch.ones(encoded.shape))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # In the last step, values should be close to 1
    for _enc in encoded.flatten().tolist():
        assert _enc == pytest.approx(1.0, abs=0.1)


def test_networkbody_visual():
    torch.manual_seed(1)
    vec_obs_size = 4
    obs_size = (3, 84, 84)
    network_settings = NetworkSettings()
    obs_shapes = [(vec_obs_size,), obs_size]

    networkbody = NetworkBody(
        create_observation_specs_with_shapes(obs_shapes), network_settings
    )
    optimizer = torch.optim.Adam(networkbody.parameters(), lr=3e-3)
    sample_obs = 0.1 * torch.ones((1, 3, 84, 84), dtype=torch.float32)
    sample_vec_obs = torch.ones((1, vec_obs_size), dtype=torch.float32)
    obs = [sample_vec_obs] + [sample_obs]
    loss = 1
    step = 0
    while loss > 1e-6 and step < 1e3:
        encoded, _ = networkbody(obs)
        assert encoded.shape == (1, network_settings.hidden_units)
        # Try to force output to 1
        loss = torch.nn.functional.mse_loss(encoded, torch.ones(encoded.shape))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1
    # In the last step, values should be close to 1
    for _enc in encoded.flatten().tolist():
        assert _enc == pytest.approx(1.0, abs=0.1)


@pytest.mark.parametrize("with_actions", [True, False], ids=["actions", "no_actions"])
def test_multinetworkbody_vector(with_actions):
    torch.manual_seed(0)
    obs_size = 4
    act_size = 2
    n_agents = 3
    network_settings = NetworkSettings()
    obs_shapes = [(obs_size,)]
    action_spec = ActionSpec(act_size, tuple(act_size for _ in range(act_size)))
    networkbody = MultiAgentNetworkBody(
        create_observation_specs_with_shapes(obs_shapes), network_settings, action_spec
    )
    optimizer = torch.optim.Adam(networkbody.parameters(), lr=3e-3)
    sample_obs = [[0.1 * torch.ones((1, obs_size))] for _ in range(n_agents)]
    # simulate baseline in POCA
    sample_act = [
        AgentAction(
            0.1 * torch.ones((1, 2)), [0.1 * torch.ones(1) for _ in range(act_size)]
        )
        for _ in range(n_agents - 1)
    ]

    for _ in range(300):
        if with_actions:
            encoded, _ = networkbody(
                obs_only=sample_obs[:1], obs=sample_obs[1:], actions=sample_act
            )
        else:
            encoded, _ = networkbody(obs_only=sample_obs, obs=[], actions=[])
        # Try to force output to 1
        loss = torch.nn.functional.mse_loss(encoded, torch.ones(encoded.shape))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # In the last step, values should be close to 1
    for _enc in encoded.flatten().tolist():
        assert _enc == pytest.approx(1.0, abs=0.1)


@pytest.mark.parametrize("with_actions", [True, False], ids=["actions", "no_actions"])
def test_multinetworkbody_lstm(with_actions):
    torch.manual_seed(0)
    obs_size = 4
    act_size = 2
    seq_len = 16
    n_agents = 3
    network_settings = NetworkSettings(
        memory=NetworkSettings.MemorySettings(sequence_length=seq_len, memory_size=12)
    )

    obs_shapes = [(obs_size,)]
    action_spec = ActionSpec(act_size, tuple(act_size for _ in range(act_size)))
    networkbody = MultiAgentNetworkBody(
        create_observation_specs_with_shapes(obs_shapes), network_settings, action_spec
    )
    optimizer = torch.optim.Adam(networkbody.parameters(), lr=3e-4)
    sample_obs = [[0.1 * torch.ones((seq_len, obs_size))] for _ in range(n_agents)]
    # simulate baseline in POCA
    sample_act = [
        AgentAction(
            0.1 * torch.ones((seq_len, 2)),
            [0.1 * torch.ones(seq_len) for _ in range(act_size)],
        )
        for _ in range(n_agents - 1)
    ]

    for _ in range(300):
        if with_actions:
            encoded, _ = networkbody(
                obs_only=sample_obs[:1],
                obs=sample_obs[1:],
                actions=sample_act,
                memories=torch.ones(1, 1, 12),
                sequence_length=seq_len,
            )
        else:
            encoded, _ = networkbody(
                obs_only=sample_obs,
                obs=[],
                actions=[],
                memories=torch.ones(1, 1, 12),
                sequence_length=seq_len,
            )
        # Try to force output to 1
        loss = torch.nn.functional.mse_loss(encoded, torch.ones(encoded.shape))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # In the last step, values should be close to 1
    for _enc in encoded.flatten().tolist():
        assert _enc == pytest.approx(1.0, abs=0.1)


@pytest.mark.parametrize("with_actions", [True, False], ids=["actions", "no_actions"])
def test_multinetworkbody_visual(with_actions):
    torch.manual_seed(0)
    act_size = 2
    n_agents = 3
    obs_size = 4
    vis_obs_size = (3, 84, 84)
    network_settings = NetworkSettings()
    obs_shapes = [(obs_size,), vis_obs_size]
    action_spec = ActionSpec(act_size, tuple(act_size for _ in range(act_size)))
    networkbody = MultiAgentNetworkBody(
        create_observation_specs_with_shapes(obs_shapes), network_settings, action_spec
    )
    optimizer = torch.optim.Adam(networkbody.parameters(), lr=3e-3)
    sample_obs = [
        [0.1 * torch.ones((1, obs_size))] + [0.1 * torch.ones((1, 3, 84, 84))]
        for _ in range(n_agents)
    ]
    # simulate baseline in POCA
    sample_act = [
        AgentAction(
            0.1 * torch.ones((1, 2)), [0.1 * torch.ones(1) for _ in range(act_size)]
        )
        for _ in range(n_agents - 1)
    ]
    for _ in range(300):
        if with_actions:
            encoded, _ = networkbody(
                obs_only=sample_obs[:1], obs=sample_obs[1:], actions=sample_act
            )
        else:
            encoded, _ = networkbody(obs_only=sample_obs, obs=[], actions=[])
        # Try to force output to 1
        loss = torch.nn.functional.mse_loss(encoded, torch.ones(encoded.shape))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # In the last step, values should be close to 1
    for _enc in encoded.flatten().tolist():
        assert _enc == pytest.approx(1.0, abs=0.1)


def test_valuenetwork():
    torch.manual_seed(0)
    obs_size = 4
    num_outputs = 2
    network_settings = NetworkSettings()
    obs_spec = create_observation_specs_with_shapes([(obs_size,)])

    stream_names = [f"stream_name{n}" for n in range(4)]
    value_net = ValueNetwork(
        stream_names, obs_spec, network_settings, outputs_per_stream=num_outputs
    )
    optimizer = torch.optim.Adam(value_net.parameters(), lr=3e-3)

    for _ in range(50):
        sample_obs = torch.ones((1, obs_size))
        values, _ = value_net([sample_obs])
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
        for _out in value.tolist():
            assert _out[0] == pytest.approx(1.0, abs=0.1)


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("lstm", [True, False])
def test_actor_critic(lstm, shared):
    obs_size = 4
    vis_obs_size = (3, 84, 84)
    network_settings = NetworkSettings(
        memory=NetworkSettings.MemorySettings() if lstm else None, normalize=True
    )
    obs_spec = create_observation_specs_with_shapes([(obs_size,), vis_obs_size])
    act_size = 2
    mask = torch.ones([1, act_size * 2])
    stream_names = [f"stream_name{n}" for n in range(4)]
    action_spec = ActionSpec(act_size, tuple(act_size for _ in range(act_size)))
    if shared:
        actor = critic = SharedActorCritic(
            obs_spec, network_settings, action_spec, stream_names, network_settings
        )
    else:
        actor = SimpleActor(obs_spec, network_settings, action_spec)
        critic = ValueNetwork(stream_names, obs_spec, network_settings)
    if lstm:
        sample_vis_obs = torch.ones(
            (network_settings.memory.sequence_length, 3, 84, 84), dtype=torch.float32
        )
        sample_obs = torch.ones((network_settings.memory.sequence_length, obs_size))
        memories = torch.ones(
            (1, network_settings.memory.sequence_length, actor.memory_size)
        )
    else:
        sample_vis_obs = 0.1 * torch.ones((1, 3, 84, 84), dtype=torch.float32)
        sample_obs = torch.ones((1, obs_size))
        memories = torch.tensor([])
        # memories isn't always set to None, the network should be able to
        # deal with that.
    # Test critic pass
    value_out, memories_out = critic.critic_pass(
        [sample_obs] + [sample_vis_obs], memories=memories
    )
    for stream in stream_names:
        if lstm:
            assert value_out[stream].shape == (network_settings.memory.sequence_length,)
            assert memories_out.shape == memories.shape
        else:
            assert value_out[stream].shape == (1,)

    # Test get action stats and_value
    action, run_out, mem_out = actor.get_action_and_stats(
        [sample_obs] + [sample_vis_obs], memories=memories, masks=mask
    )
    log_probs = run_out["log_probs"]
    entropy = run_out["entropy"]

    eval_run_out = actor.get_stats(
        [sample_obs] + [sample_vis_obs], action, memories=memories, masks=mask
    )
    eval_log_probs = eval_run_out["log_probs"]
    eval_entropy = eval_run_out["entropy"]

    if lstm:
        assert action.continuous_tensor.shape == (64, 2)
        assert log_probs.continuous_tensor.shape == (64, 2)
        assert entropy.shape == (64,)
        assert eval_log_probs.continuous_tensor.shape == (64, 2)
        assert eval_entropy.shape == (64,)

    else:
        assert action.continuous_tensor.shape == (1, 2)
        assert log_probs.continuous_tensor.shape == (1, 2)
        assert entropy.shape == (1,)
        assert eval_log_probs.continuous_tensor.shape == (1, 2)
        assert eval_entropy.shape == (1,)

    assert len(action.discrete_list) == 2
    for _disc, _disc_prob, _eval_disc_prob in zip(
        action.discrete_list, log_probs.discrete_list, eval_log_probs.discrete_list
    ):
        if lstm:
            assert _disc.shape == (64, 1)
            assert _eval_disc_prob.shape == (64,)
        else:
            assert _disc.shape == (1, 1)
            assert _disc_prob.shape == (1,)
            assert _eval_disc_prob.shape == (1,)

    if mem_out is not None:
        assert mem_out.shape == memories.shape


@pytest.mark.parametrize("with_actions", [True, False], ids=["actions", "no_actions"])
def test_multinetworkbody_num_agents(with_actions):
    torch.manual_seed(0)
    act_size = 2
    obs_size = 4
    network_settings = NetworkSettings()
    obs_shapes = [(obs_size,)]
    action_spec = ActionSpec(act_size, tuple(act_size for _ in range(act_size)))
    networkbody = MultiAgentNetworkBody(
        create_observation_specs_with_shapes(obs_shapes), network_settings, action_spec
    )
    sample_obs = [[0.1 * torch.ones((1, obs_size))]]
    # simulate baseline in POCA
    sample_act = [
        AgentAction(
            0.1 * torch.ones((1, 2)), [0.1 * torch.ones(1) for _ in range(act_size)]
        )
    ]
    for n_agent, max_so_far in [(1, 1), (5, 5), (4, 5), (10, 10), (5, 10), (1, 10)]:
        if with_actions:
            encoded, _ = networkbody(
                obs_only=sample_obs * (n_agent - 1), obs=sample_obs, actions=sample_act
            )
        else:
            encoded, _ = networkbody(obs_only=sample_obs * n_agent, obs=[], actions=[])
        # look at the last value of the hidden units (the number of agents)
        target = (n_agent * 1.0 / max_so_far) * 2 - 1
        assert abs(encoded[0, -1].item() - target) < 1e-6
        assert encoded[0, -1].item() <= 1
        assert encoded[0, -1].item() >= -1

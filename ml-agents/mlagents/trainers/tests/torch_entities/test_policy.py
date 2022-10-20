import pytest

from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.tests import mock_brain as mb
from mlagents.trainers.settings import NetworkSettings
from mlagents.trainers.torch_entities.networks import SimpleActor

VECTOR_ACTION_SPACE = 2
VECTOR_OBS_SPACE = 8
DISCRETE_ACTION_SPACE = [3, 3, 3, 2]
BUFFER_INIT_SAMPLES = 32
NUM_AGENTS = 12
EPSILON = 1e-7


def create_policy_mock(
    dummy_config: NetworkSettings,
    use_rnn: bool = False,
    use_discrete: bool = True,
    use_visual: bool = False,
    seed: int = 0,
) -> TorchPolicy:
    mock_spec = mb.setup_test_behavior_specs(
        use_discrete,
        use_visual,
        vector_action_space=DISCRETE_ACTION_SPACE
        if use_discrete
        else VECTOR_ACTION_SPACE,
        vector_obs_space=VECTOR_OBS_SPACE,
    )

    network_settings = dummy_config
    network_settings.memory = NetworkSettings.MemorySettings() if use_rnn else None
    actor_kwargs = {
        "conditional_sigma": False,
        "tanh_squash": False,
    }
    policy = TorchPolicy(seed, mock_spec, network_settings, SimpleActor, actor_kwargs)
    return policy


@pytest.mark.parametrize("discrete", [True, False], ids=["discrete", "continuous"])
@pytest.mark.parametrize("visual", [True, False], ids=["visual", "vector"])
@pytest.mark.parametrize("rnn", [True, False], ids=["rnn", "no_rnn"])
def test_policy_evaluate(rnn, visual, discrete):
    # Test evaluate
    policy = create_policy_mock(
        NetworkSettings(), use_rnn=rnn, use_discrete=discrete, use_visual=visual
    )
    decision_step, terminal_step = mb.create_steps_from_behavior_spec(
        policy.behavior_spec, num_agents=NUM_AGENTS
    )

    run_out = policy.evaluate(decision_step, list(decision_step.agent_id))
    if discrete:
        run_out["action"].discrete.shape == (NUM_AGENTS, len(DISCRETE_ACTION_SPACE))
    else:
        assert run_out["action"].continuous.shape == (NUM_AGENTS, VECTOR_ACTION_SPACE)


def test_step_overflow():
    policy = create_policy_mock(NetworkSettings())
    policy.set_step(2**31 - 1)
    assert policy.get_current_step() == 2**31 - 1  # step = 2147483647
    policy.increment_step(3)
    assert policy.get_current_step() == 2**31 + 2  # step = 2147483650

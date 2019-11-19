import unittest.mock as mock
import pytest
import yaml
import os
import mlagents.trainers.tests.mock_brain as mb
from mlagents.trainers.ppo.policy import PPOPolicy
from mlagents.trainers.sac.policy import SACPolicy


def ppo_dummy_config():
    return yaml.safe_load(
        """
        trainer: ppo
        batch_size: 32
        beta: 5.0e-3
        buffer_size: 512
        epsilon: 0.2
        hidden_units: 128
        lambd: 0.95
        learning_rate: 3.0e-4
        max_steps: 5.0e4
        normalize: true
        num_epoch: 5
        num_layers: 2
        time_horizon: 64
        sequence_length: 64
        summary_freq: 1000
        use_recurrent: false
        memory_size: 8
        reward_signals:
          extrinsic:
            strength: 1.0
            gamma: 0.99
        """
    )


def sac_dummy_config():
    return yaml.safe_load(
        """
        trainer: sac
        batch_size: 128
        buffer_size: 50000
        buffer_init_steps: 0
        hidden_units: 128
        init_entcoef: 1.0
        learning_rate: 3.0e-4
        max_steps: 5.0e4
        memory_size: 256
        normalize: false
        num_update: 1
        train_interval: 1
        num_layers: 2
        time_horizon: 64
        sequence_length: 64
        summary_freq: 1000
        tau: 0.005
        use_recurrent: false
        vis_encode_type: simple
        pretraining:
            demo_path: ./demos/ExpertPyramid.demo
            strength: 1.0
            steps: 10000000
        reward_signals:
            extrinsic:
                strength: 1.0
                gamma: 0.99
        """
    )


@pytest.fixture
def gail_dummy_config():
    return {
        "gail": {
            "strength": 0.1,
            "gamma": 0.9,
            "encoding_size": 128,
            "use_vail": True,
            "demo_path": os.path.dirname(os.path.abspath(__file__)) + "/test.demo",
        }
    }


@pytest.fixture
def curiosity_dummy_config():
    return {"curiosity": {"strength": 0.1, "gamma": 0.9, "encoding_size": 128}}


VECTOR_ACTION_SPACE = [2]
VECTOR_OBS_SPACE = 8
DISCRETE_ACTION_SPACE = [3, 3, 3, 2]
BUFFER_INIT_SAMPLES = 20
NUM_AGENTS = 12


def create_policy_mock(
    mock_env, trainer_config, reward_signal_config, use_rnn, use_discrete, use_visual
):
    env, mock_brain, _ = mb.setup_mock_env_and_brains(
        mock_env,
        use_discrete,
        use_visual,
        num_agents=NUM_AGENTS,
        vector_action_space=VECTOR_ACTION_SPACE,
        vector_obs_space=VECTOR_OBS_SPACE,
        discrete_action_space=DISCRETE_ACTION_SPACE,
    )

    trainer_parameters = trainer_config
    model_path = env.external_brain_names[0]
    trainer_parameters["model_path"] = model_path
    trainer_parameters["keep_checkpoints"] = 3
    trainer_parameters["reward_signals"].update(reward_signal_config)
    trainer_parameters["use_recurrent"] = use_rnn
    if trainer_config["trainer"] == "ppo":
        policy = PPOPolicy(0, mock_brain, trainer_parameters, False, False)
    else:
        policy = SACPolicy(0, mock_brain, trainer_parameters, False, False)
    return env, policy


def reward_signal_eval(env, policy, reward_signal_name):
    brain_infos = env.reset()
    brain_info = brain_infos[env.external_brain_names[0]]
    next_brain_info = env.step()[env.external_brain_names[0]]
    # Test evaluate
    rsig_result = policy.reward_signals[reward_signal_name].evaluate(
        brain_info, next_brain_info
    )
    assert rsig_result.scaled_reward.shape == (NUM_AGENTS,)
    assert rsig_result.unscaled_reward.shape == (NUM_AGENTS,)


def reward_signal_update(env, policy, reward_signal_name):
    buffer = mb.simulate_rollout(env, policy, BUFFER_INIT_SAMPLES)
    feed_dict = policy.reward_signals[reward_signal_name].prepare_update(
        policy.model, buffer.update_buffer.make_mini_batch(0, 10), 2
    )
    out = policy._execute_model(
        feed_dict, policy.reward_signals[reward_signal_name].update_dict
    )
    assert type(out) is dict


@pytest.mark.parametrize(
    "trainer_config", [ppo_dummy_config(), sac_dummy_config()], ids=["ppo", "sac"]
)
@mock.patch("mlagents.envs.environment.UnityEnvironment")
def test_gail_cc(mock_env, trainer_config, gail_dummy_config):
    env, policy = create_policy_mock(
        mock_env, trainer_config, gail_dummy_config, False, False, False
    )
    reward_signal_eval(env, policy, "gail")
    reward_signal_update(env, policy, "gail")


@pytest.mark.parametrize(
    "trainer_config", [ppo_dummy_config(), sac_dummy_config()], ids=["ppo", "sac"]
)
@mock.patch("mlagents.envs.environment.UnityEnvironment")
def test_gail_dc_visual(mock_env, trainer_config, gail_dummy_config):
    gail_dummy_config["gail"]["demo_path"] = (
        os.path.dirname(os.path.abspath(__file__)) + "/testdcvis.demo"
    )
    env, policy = create_policy_mock(
        mock_env, trainer_config, gail_dummy_config, False, True, True
    )
    reward_signal_eval(env, policy, "gail")
    reward_signal_update(env, policy, "gail")


@pytest.mark.parametrize(
    "trainer_config", [ppo_dummy_config(), sac_dummy_config()], ids=["ppo", "sac"]
)
@mock.patch("mlagents.envs.environment.UnityEnvironment")
def test_gail_rnn(mock_env, trainer_config, gail_dummy_config):
    env, policy = create_policy_mock(
        mock_env, trainer_config, gail_dummy_config, True, False, False
    )
    reward_signal_eval(env, policy, "gail")
    reward_signal_update(env, policy, "gail")


@pytest.mark.parametrize(
    "trainer_config", [ppo_dummy_config(), sac_dummy_config()], ids=["ppo", "sac"]
)
@mock.patch("mlagents.envs.environment.UnityEnvironment")
def test_curiosity_cc(mock_env, trainer_config, curiosity_dummy_config):
    env, policy = create_policy_mock(
        mock_env, trainer_config, curiosity_dummy_config, False, False, False
    )
    reward_signal_eval(env, policy, "curiosity")
    reward_signal_update(env, policy, "curiosity")


@pytest.mark.parametrize(
    "trainer_config", [ppo_dummy_config(), sac_dummy_config()], ids=["ppo", "sac"]
)
@mock.patch("mlagents.envs.environment.UnityEnvironment")
def test_curiosity_dc(mock_env, trainer_config, curiosity_dummy_config):
    env, policy = create_policy_mock(
        mock_env, trainer_config, curiosity_dummy_config, False, True, False
    )
    reward_signal_eval(env, policy, "curiosity")
    reward_signal_update(env, policy, "curiosity")


@pytest.mark.parametrize(
    "trainer_config", [ppo_dummy_config(), sac_dummy_config()], ids=["ppo", "sac"]
)
@mock.patch("mlagents.envs.environment.UnityEnvironment")
def test_curiosity_visual(mock_env, trainer_config, curiosity_dummy_config):
    env, policy = create_policy_mock(
        mock_env, trainer_config, curiosity_dummy_config, False, False, True
    )
    reward_signal_eval(env, policy, "curiosity")
    reward_signal_update(env, policy, "curiosity")


@pytest.mark.parametrize(
    "trainer_config", [ppo_dummy_config(), sac_dummy_config()], ids=["ppo", "sac"]
)
@mock.patch("mlagents.envs.environment.UnityEnvironment")
def test_curiosity_rnn(mock_env, trainer_config, curiosity_dummy_config):
    env, policy = create_policy_mock(
        mock_env, trainer_config, curiosity_dummy_config, True, False, False
    )
    reward_signal_eval(env, policy, "curiosity")
    reward_signal_update(env, policy, "curiosity")


@pytest.mark.parametrize(
    "trainer_config", [ppo_dummy_config(), sac_dummy_config()], ids=["ppo", "sac"]
)
@mock.patch("mlagents.envs.environment.UnityEnvironment")
def test_extrinsic(mock_env, trainer_config, curiosity_dummy_config):
    env, policy = create_policy_mock(
        mock_env, trainer_config, curiosity_dummy_config, False, False, False
    )
    reward_signal_eval(env, policy, "extrinsic")
    reward_signal_update(env, policy, "extrinsic")


if __name__ == "__main__":
    pytest.main()

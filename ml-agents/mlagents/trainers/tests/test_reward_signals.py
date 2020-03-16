import pytest
import yaml
import os
import mlagents.trainers.tests.mock_brain as mb
from mlagents.trainers.policy.nn_policy import NNPolicy
from mlagents.trainers.sac.optimizer import SACOptimizer
from mlagents.trainers.ppo.optimizer import PPOOptimizer


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
        behavioral_cloning:
            demo_path: ./Project/Assets/ML-Agents/Examples/Pyramids/Demos/ExpertPyramid.demo
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
BATCH_SIZE = 12
NUM_AGENTS = 12


def create_optimizer_mock(
    trainer_config, reward_signal_config, use_rnn, use_discrete, use_visual
):
    mock_brain = mb.setup_mock_brain(
        use_discrete,
        use_visual,
        vector_action_space=VECTOR_ACTION_SPACE,
        vector_obs_space=VECTOR_OBS_SPACE,
        discrete_action_space=DISCRETE_ACTION_SPACE,
    )

    trainer_parameters = trainer_config
    model_path = "testpath"
    trainer_parameters["model_path"] = model_path
    trainer_parameters["keep_checkpoints"] = 3
    trainer_parameters["reward_signals"].update(reward_signal_config)
    trainer_parameters["use_recurrent"] = use_rnn
    policy = NNPolicy(
        0, mock_brain, trainer_parameters, False, False, create_tf_graph=False
    )
    if trainer_parameters["trainer"] == "sac":
        optimizer = SACOptimizer(policy, trainer_parameters)
    else:
        optimizer = PPOOptimizer(policy, trainer_parameters)
    return optimizer


def reward_signal_eval(optimizer, reward_signal_name):
    buffer = mb.simulate_rollout(BATCH_SIZE, optimizer.policy.brain)
    # Test evaluate
    rsig_result = optimizer.reward_signals[reward_signal_name].evaluate_batch(buffer)
    assert rsig_result.scaled_reward.shape == (BATCH_SIZE,)
    assert rsig_result.unscaled_reward.shape == (BATCH_SIZE,)


def reward_signal_update(optimizer, reward_signal_name):
    buffer = mb.simulate_rollout(BUFFER_INIT_SAMPLES, optimizer.policy.brain)
    feed_dict = optimizer.reward_signals[reward_signal_name].prepare_update(
        optimizer.policy, buffer.make_mini_batch(0, 10), 2
    )
    out = optimizer.policy._execute_model(
        feed_dict, optimizer.reward_signals[reward_signal_name].update_dict
    )
    assert type(out) is dict


@pytest.mark.parametrize(
    "trainer_config", [ppo_dummy_config(), sac_dummy_config()], ids=["ppo", "sac"]
)
def test_gail_cc(trainer_config, gail_dummy_config):
    optimizer = create_optimizer_mock(
        trainer_config, gail_dummy_config, False, False, False
    )
    reward_signal_eval(optimizer, "gail")
    reward_signal_update(optimizer, "gail")


@pytest.mark.parametrize(
    "trainer_config", [ppo_dummy_config(), sac_dummy_config()], ids=["ppo", "sac"]
)
def test_gail_dc_visual(trainer_config, gail_dummy_config):
    gail_dummy_config["gail"]["demo_path"] = (
        os.path.dirname(os.path.abspath(__file__)) + "/testdcvis.demo"
    )
    optimizer = create_optimizer_mock(
        trainer_config, gail_dummy_config, False, True, True
    )
    reward_signal_eval(optimizer, "gail")
    reward_signal_update(optimizer, "gail")


@pytest.mark.parametrize(
    "trainer_config", [ppo_dummy_config(), sac_dummy_config()], ids=["ppo", "sac"]
)
def test_gail_rnn(trainer_config, gail_dummy_config):
    policy = create_optimizer_mock(
        trainer_config, gail_dummy_config, True, False, False
    )
    reward_signal_eval(policy, "gail")
    reward_signal_update(policy, "gail")


@pytest.mark.parametrize(
    "trainer_config", [ppo_dummy_config(), sac_dummy_config()], ids=["ppo", "sac"]
)
def test_curiosity_cc(trainer_config, curiosity_dummy_config):
    policy = create_optimizer_mock(
        trainer_config, curiosity_dummy_config, False, False, False
    )
    reward_signal_eval(policy, "curiosity")
    reward_signal_update(policy, "curiosity")


@pytest.mark.parametrize(
    "trainer_config", [ppo_dummy_config(), sac_dummy_config()], ids=["ppo", "sac"]
)
def test_curiosity_dc(trainer_config, curiosity_dummy_config):
    policy = create_optimizer_mock(
        trainer_config, curiosity_dummy_config, False, True, False
    )
    reward_signal_eval(policy, "curiosity")
    reward_signal_update(policy, "curiosity")


@pytest.mark.parametrize(
    "trainer_config", [ppo_dummy_config(), sac_dummy_config()], ids=["ppo", "sac"]
)
def test_curiosity_visual(trainer_config, curiosity_dummy_config):
    policy = create_optimizer_mock(
        trainer_config, curiosity_dummy_config, False, False, True
    )
    reward_signal_eval(policy, "curiosity")
    reward_signal_update(policy, "curiosity")


@pytest.mark.parametrize(
    "trainer_config", [ppo_dummy_config(), sac_dummy_config()], ids=["ppo", "sac"]
)
def test_curiosity_rnn(trainer_config, curiosity_dummy_config):
    policy = create_optimizer_mock(
        trainer_config, curiosity_dummy_config, True, False, False
    )
    reward_signal_eval(policy, "curiosity")
    reward_signal_update(policy, "curiosity")


@pytest.mark.parametrize(
    "trainer_config", [ppo_dummy_config(), sac_dummy_config()], ids=["ppo", "sac"]
)
def test_extrinsic(trainer_config, curiosity_dummy_config):
    policy = create_optimizer_mock(
        trainer_config, curiosity_dummy_config, False, False, False
    )
    reward_signal_eval(policy, "extrinsic")
    reward_signal_update(policy, "extrinsic")


if __name__ == "__main__":
    pytest.main()

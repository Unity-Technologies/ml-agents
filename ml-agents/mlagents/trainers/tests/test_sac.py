import pytest
from unittest import mock
import yaml

from mlagents.tf_utils import tf


from mlagents.trainers.sac.trainer import SACTrainer
from mlagents.trainers.sac.optimizer import SACOptimizer
from mlagents.trainers.policy.nn_policy import NNPolicy
from mlagents.trainers.agent_processor import AgentManagerQueue
from mlagents.trainers.tests import mock_brain as mb
from mlagents.trainers.tests.mock_brain import make_brain_parameters
from mlagents.trainers.tests.test_trajectory import make_fake_trajectory
from mlagents.trainers.exception import UnityTrainerException


@pytest.fixture
def dummy_config():
    return yaml.safe_load(
        """
        trainer: sac
        batch_size: 32
        buffer_size: 10240
        buffer_init_steps: 0
        hidden_units: 32
        init_entcoef: 0.1
        learning_rate: 3.0e-4
        max_steps: 1024
        memory_size: 10
        normalize: true
        num_update: 1
        train_interval: 1
        num_layers: 1
        time_horizon: 64
        sequence_length: 16
        summary_freq: 1000
        tau: 0.005
        use_recurrent: false
        curiosity_enc_size: 128
        demo_path: None
        vis_encode_type: simple
        reward_signals:
            extrinsic:
                strength: 1.0
                gamma: 0.99
        """
    )


VECTOR_ACTION_SPACE = [2]
VECTOR_OBS_SPACE = 8
DISCRETE_ACTION_SPACE = [3, 3, 3, 2]
BUFFER_INIT_SAMPLES = 64
NUM_AGENTS = 12


def create_sac_optimizer_mock(dummy_config, use_rnn, use_discrete, use_visual):
    mock_brain = mb.setup_mock_brain(
        use_discrete,
        use_visual,
        vector_action_space=VECTOR_ACTION_SPACE,
        vector_obs_space=VECTOR_OBS_SPACE,
        discrete_action_space=DISCRETE_ACTION_SPACE,
    )

    trainer_parameters = dummy_config
    model_path = "testmodel"
    trainer_parameters["model_path"] = model_path
    trainer_parameters["keep_checkpoints"] = 3
    trainer_parameters["use_recurrent"] = use_rnn
    policy = NNPolicy(
        0, mock_brain, trainer_parameters, False, False, create_tf_graph=False
    )
    optimizer = SACOptimizer(policy, trainer_parameters)
    return optimizer


@pytest.mark.parametrize("discrete", [True, False], ids=["discrete", "continuous"])
@pytest.mark.parametrize("visual", [True, False], ids=["visual", "vector"])
@pytest.mark.parametrize("rnn", [True, False], ids=["rnn", "no_rnn"])
def test_sac_optimizer_update(dummy_config, rnn, visual, discrete):
    # Test evaluate
    tf.reset_default_graph()
    optimizer = create_sac_optimizer_mock(
        dummy_config, use_rnn=rnn, use_discrete=discrete, use_visual=visual
    )
    # Test update
    update_buffer = mb.simulate_rollout(BUFFER_INIT_SAMPLES, optimizer.policy.brain)
    # Mock out reward signal eval
    update_buffer["extrinsic_rewards"] = update_buffer["environment_rewards"]
    optimizer.update(
        update_buffer,
        num_sequences=update_buffer.num_experiences // optimizer.policy.sequence_length,
    )


@pytest.mark.parametrize("discrete", [True, False], ids=["discrete", "continuous"])
def test_sac_update_reward_signals(dummy_config, discrete):
    # Test evaluate
    tf.reset_default_graph()
    # Add a Curiosity module
    dummy_config["reward_signals"]["curiosity"] = {}
    dummy_config["reward_signals"]["curiosity"]["strength"] = 1.0
    dummy_config["reward_signals"]["curiosity"]["gamma"] = 0.99
    dummy_config["reward_signals"]["curiosity"]["encoding_size"] = 128
    optimizer = create_sac_optimizer_mock(
        dummy_config, use_rnn=False, use_discrete=discrete, use_visual=False
    )

    # Test update, while removing PPO-specific buffer elements.
    update_buffer = mb.simulate_rollout(BUFFER_INIT_SAMPLES, optimizer.policy.brain)

    # Mock out reward signal eval
    update_buffer["extrinsic_rewards"] = update_buffer["environment_rewards"]
    update_buffer["curiosity_rewards"] = update_buffer["environment_rewards"]
    optimizer.update_reward_signals(
        {"curiosity": update_buffer}, num_sequences=update_buffer.num_experiences
    )


def test_sac_save_load_buffer(tmpdir, dummy_config):
    mock_brain = mb.setup_mock_brain(
        False,
        False,
        vector_action_space=VECTOR_ACTION_SPACE,
        vector_obs_space=VECTOR_OBS_SPACE,
        discrete_action_space=DISCRETE_ACTION_SPACE,
    )
    trainer_params = dummy_config
    trainer_params["summary_path"] = str(tmpdir)
    trainer_params["model_path"] = str(tmpdir)
    trainer_params["save_replay_buffer"] = True
    trainer = SACTrainer(mock_brain.brain_name, 1, trainer_params, True, False, 0, 0)
    policy = trainer.create_policy(mock_brain.brain_name, mock_brain)
    trainer.add_policy(mock_brain.brain_name, policy)

    trainer.update_buffer = mb.simulate_rollout(BUFFER_INIT_SAMPLES, policy.brain)
    buffer_len = trainer.update_buffer.num_experiences
    trainer.save_model(mock_brain.brain_name)

    # Wipe Trainer and try to load
    trainer2 = SACTrainer(mock_brain.brain_name, 1, trainer_params, True, True, 0, 0)

    policy = trainer2.create_policy(mock_brain.brain_name, mock_brain)
    trainer2.add_policy(mock_brain.brain_name, policy)
    assert trainer2.update_buffer.num_experiences == buffer_len


@mock.patch("mlagents.trainers.sac.trainer.SACOptimizer")
def test_add_get_policy(sac_optimizer, dummy_config):
    brain_params = make_brain_parameters(
        discrete_action=False, visual_inputs=0, vec_obs_size=6
    )
    mock_optimizer = mock.Mock()
    mock_optimizer.reward_signals = {}
    sac_optimizer.return_value = mock_optimizer

    dummy_config["summary_path"] = "./summaries/test_trainer_summary"
    dummy_config["model_path"] = "./models/test_trainer_models/TestModel"
    trainer = SACTrainer(brain_params, 0, dummy_config, True, False, 0, "0")
    policy = mock.Mock(spec=NNPolicy)
    policy.get_current_step.return_value = 2000

    trainer.add_policy(brain_params.brain_name, policy)
    assert trainer.get_policy(brain_params.brain_name) == policy

    # Make sure the summary steps were loaded properly
    assert trainer.get_step == 2000
    assert trainer.next_summary_step > 2000

    # Test incorrect class of policy
    policy = mock.Mock()
    with pytest.raises(RuntimeError):
        trainer.add_policy(brain_params, policy)


def test_process_trajectory(dummy_config):
    brain_params = make_brain_parameters(
        discrete_action=False, visual_inputs=0, vec_obs_size=6
    )
    dummy_config["summary_path"] = "./summaries/test_trainer_summary"
    dummy_config["model_path"] = "./models/test_trainer_models/TestModel"
    trainer = SACTrainer(brain_params, 0, dummy_config, True, False, 0, "0")
    policy = trainer.create_policy(brain_params.brain_name, brain_params)
    trainer.add_policy(brain_params.brain_name, policy)

    trajectory_queue = AgentManagerQueue("testbrain")
    trainer.subscribe_trajectory_queue(trajectory_queue)

    trajectory = make_fake_trajectory(
        length=15,
        max_step_complete=True,
        vec_obs_size=6,
        num_vis_obs=0,
        action_space=[2],
    )
    trajectory_queue.put(trajectory)
    trainer.advance()

    # Check that trainer put trajectory in update buffer
    assert trainer.update_buffer.num_experiences == 15

    # Check that the stats are being collected as episode isn't complete
    for reward in trainer.collected_rewards.values():
        for agent in reward.values():
            assert agent > 0

    # Add a terminal trajectory
    trajectory = make_fake_trajectory(
        length=15,
        max_step_complete=False,
        vec_obs_size=6,
        num_vis_obs=0,
        action_space=[2],
    )
    trajectory_queue.put(trajectory)
    trainer.advance()

    # Check that the stats are reset as episode is finished
    for reward in trainer.collected_rewards.values():
        for agent in reward.values():
            assert agent == 0
    assert trainer.stats_reporter.get_stats_summaries("Policy/Extrinsic Reward").num > 0
    # Assert we're not just using the default values
    assert (
        trainer.stats_reporter.get_stats_summaries("Policy/Extrinsic Reward").mean > 0
    )


def test_bad_config(dummy_config):
    brain_params = make_brain_parameters(
        discrete_action=False, visual_inputs=0, vec_obs_size=6
    )
    # Test that we throw an error if we have sequence length greater than batch size
    dummy_config["sequence_length"] = 64
    dummy_config["batch_size"] = 32
    dummy_config["use_recurrent"] = True
    dummy_config["summary_path"] = "./summaries/test_trainer_summary"
    dummy_config["model_path"] = "./models/test_trainer_models/TestModel"
    with pytest.raises(UnityTrainerException):
        _ = SACTrainer(brain_params, 0, dummy_config, True, False, 0, "0")


if __name__ == "__main__":
    pytest.main()

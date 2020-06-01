import pytest
from unittest import mock
import copy

from mlagents.tf_utils import tf


from mlagents.trainers.sac.trainer import SACTrainer
from mlagents.trainers.sac.optimizer import SACOptimizer
from mlagents.trainers.policy.nn_policy import NNPolicy
from mlagents.trainers.agent_processor import AgentManagerQueue
from mlagents.trainers.tests import mock_brain as mb
from mlagents.trainers.tests.mock_brain import make_brain_parameters
from mlagents.trainers.tests.test_trajectory import make_fake_trajectory
from mlagents.trainers.tests.test_simple_rl import SAC_CONFIG
from mlagents.trainers.settings import NetworkSettings
from mlagents.trainers.tests.test_reward_signals import (  # noqa: F401; pylint: disable=unused-variable
    curiosity_dummy_config,
)


@pytest.fixture
def dummy_config():
    return copy.deepcopy(SAC_CONFIG)


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
    trainer_settings = dummy_config
    trainer_settings.network_settings.memory = (
        NetworkSettings.MemorySettings(sequence_length=16, memory_size=10)
        if use_rnn
        else None
    )
    policy = NNPolicy(
        0, mock_brain, trainer_settings, False, False, create_tf_graph=False
    )
    optimizer = SACOptimizer(policy, trainer_settings)
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
def test_sac_update_reward_signals(
    dummy_config, curiosity_dummy_config, discrete  # noqa: F811
):
    # Test evaluate
    tf.reset_default_graph()
    # Add a Curiosity module
    dummy_config.reward_signals = curiosity_dummy_config
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
    trainer_params.hyperparameters.save_replay_buffer = True
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


def test_advance(dummy_config):
    brain_params = make_brain_parameters(
        discrete_action=False, visual_inputs=0, vec_obs_size=6
    )
    dummy_config.hyperparameters.steps_per_update = 20
    dummy_config.hyperparameters.reward_signal_steps_per_update = 20
    dummy_config.hyperparameters.buffer_init_steps = 0
    trainer = SACTrainer(brain_params, 0, dummy_config, True, False, 0, "0")
    policy = trainer.create_policy(brain_params.brain_name, brain_params)
    trainer.add_policy(brain_params.brain_name, policy)

    trajectory_queue = AgentManagerQueue("testbrain")
    policy_queue = AgentManagerQueue("testbrain")
    trainer.subscribe_trajectory_queue(trajectory_queue)
    trainer.publish_policy_queue(policy_queue)

    trajectory = make_fake_trajectory(
        length=15,
        max_step_complete=True,
        vec_obs_size=6,
        num_vis_obs=0,
        action_space=[2],
        is_discrete=False,
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
        length=6,
        max_step_complete=False,
        vec_obs_size=6,
        num_vis_obs=0,
        action_space=[2],
        is_discrete=False,
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

    # Make sure there is a policy on the queue
    policy_queue.get_nowait()

    # Add another trajectory. Since this is less than 20 steps total (enough for)
    # two updates, there should NOT be a policy on the queue.
    trajectory = make_fake_trajectory(
        length=5,
        max_step_complete=False,
        vec_obs_size=6,
        num_vis_obs=0,
        action_space=[2],
        is_discrete=False,
    )
    trajectory_queue.put(trajectory)
    trainer.advance()
    with pytest.raises(AgentManagerQueue.Empty):
        policy_queue.get_nowait()

    # Call add_policy and check that we update the correct number of times.
    # This is to emulate a load from checkpoint.
    policy = trainer.create_policy(brain_params.brain_name, brain_params)
    policy.get_current_step = lambda: 200
    trainer.add_policy(brain_params.brain_name, policy)
    trainer.optimizer.update = mock.Mock()
    trainer.optimizer.update_reward_signals = mock.Mock()
    trainer.optimizer.update_reward_signals.return_value = {}
    trainer.optimizer.update.return_value = {}
    trajectory_queue.put(trajectory)
    trainer.advance()
    # Make sure we did exactly 1 update
    assert trainer.optimizer.update.call_count == 1
    assert trainer.optimizer.update_reward_signals.call_count == 1


if __name__ == "__main__":
    pytest.main()

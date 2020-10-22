import pytest
from unittest import mock
import attr

from mlagents.tf_utils import tf
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers

from mlagents.trainers.trainer.rl_trainer import RLTrainer
from mlagents.trainers.sac.trainer import SACTrainer
from mlagents.trainers.sac.optimizer_tf import SACOptimizer
from mlagents.trainers.policy.tf_policy import TFPolicy
from mlagents.trainers.agent_processor import AgentManagerQueue
from mlagents.trainers.tests import mock_brain as mb
from mlagents.trainers.tests.mock_brain import setup_test_behavior_specs
from mlagents.trainers.tests.test_trajectory import make_fake_trajectory
from mlagents.trainers.settings import NetworkSettings, FrameworkType
from mlagents.trainers.tests.dummy_config import (  # noqa: F401; pylint: disable=unused-variable
    curiosity_dummy_config,
    sac_dummy_config,
)


@pytest.fixture
def dummy_config():
    return attr.evolve(sac_dummy_config(), framework=FrameworkType.TENSORFLOW)


VECTOR_ACTION_SPACE = 2
VECTOR_OBS_SPACE = 8
DISCRETE_ACTION_SPACE = [3, 3, 3, 2]
BUFFER_INIT_SAMPLES = 64
NUM_AGENTS = 12


def create_sac_optimizer_mock(dummy_config, use_rnn, use_discrete, use_visual):
    mock_brain = mb.setup_test_behavior_specs(
        use_discrete,
        use_visual,
        vector_action_space=DISCRETE_ACTION_SPACE
        if use_discrete
        else VECTOR_ACTION_SPACE,
        vector_obs_space=VECTOR_OBS_SPACE if not use_visual else 0,
    )
    trainer_settings = dummy_config
    trainer_settings.network_settings.memory = (
        NetworkSettings.MemorySettings(sequence_length=16, memory_size=10)
        if use_rnn
        else None
    )
    policy = TFPolicy(
        0, mock_brain, trainer_settings, "test", False, create_tf_graph=False
    )
    optimizer = SACOptimizer(policy, trainer_settings)
    optimizer.policy.initialize()
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
    update_buffer = mb.simulate_rollout(
        BUFFER_INIT_SAMPLES, optimizer.policy.behavior_spec
    )
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
    update_buffer = mb.simulate_rollout(
        BUFFER_INIT_SAMPLES, optimizer.policy.behavior_spec
    )

    # Mock out reward signal eval
    update_buffer["extrinsic_rewards"] = update_buffer["environment_rewards"]
    update_buffer["curiosity_rewards"] = update_buffer["environment_rewards"]
    optimizer.update_reward_signals(
        {"curiosity": update_buffer}, num_sequences=update_buffer.num_experiences
    )


def test_sac_save_load_buffer(tmpdir, dummy_config):
    mock_specs = mb.setup_test_behavior_specs(
        False,
        False,
        vector_action_space=VECTOR_ACTION_SPACE,
        vector_obs_space=VECTOR_OBS_SPACE,
    )
    trainer_params = dummy_config
    trainer_params.hyperparameters.save_replay_buffer = True
    trainer = SACTrainer("test", 1, trainer_params, True, False, 0, "testdir")
    behavior_id = BehaviorIdentifiers.from_name_behavior_id(trainer.brain_name)
    policy = trainer.create_policy(behavior_id, mock_specs)
    trainer.add_policy(behavior_id, policy)

    trainer.update_buffer = mb.simulate_rollout(
        BUFFER_INIT_SAMPLES, policy.behavior_spec
    )
    buffer_len = trainer.update_buffer.num_experiences
    trainer.save_model()

    # Wipe Trainer and try to load
    trainer2 = SACTrainer("test", 1, trainer_params, True, True, 0, "testdir")

    policy = trainer2.create_policy(behavior_id, mock_specs)
    trainer2.add_policy(behavior_id, policy)
    assert trainer2.update_buffer.num_experiences == buffer_len


@mock.patch.object(RLTrainer, "create_model_saver")
@mock.patch("mlagents.trainers.sac.trainer.SACOptimizer")
def test_add_get_policy(sac_optimizer, mock_create_model_saver, dummy_config):
    mock_optimizer = mock.Mock()
    mock_optimizer.reward_signals = {}
    sac_optimizer.return_value = mock_optimizer

    trainer = SACTrainer("test", 0, dummy_config, True, False, 0, "0")
    policy = mock.Mock(spec=TFPolicy)
    policy.get_current_step.return_value = 2000
    behavior_id = BehaviorIdentifiers.from_name_behavior_id(trainer.brain_name)
    trainer.add_policy(behavior_id, policy)
    assert trainer.get_policy(behavior_id.behavior_id) == policy

    # Make sure the summary steps were loaded properly
    assert trainer.get_step == 2000


def test_advance(dummy_config):
    specs = setup_test_behavior_specs(
        use_discrete=False, use_visual=False, vector_action_space=2
    )
    dummy_config.hyperparameters.steps_per_update = 20
    dummy_config.hyperparameters.reward_signal_steps_per_update = 20
    dummy_config.hyperparameters.buffer_init_steps = 0
    trainer = SACTrainer("test", 0, dummy_config, True, False, 0, "0")
    behavior_id = BehaviorIdentifiers.from_name_behavior_id(trainer.brain_name)
    policy = trainer.create_policy(behavior_id, specs)
    trainer.add_policy(behavior_id, policy)

    trajectory_queue = AgentManagerQueue("testbrain")
    policy_queue = AgentManagerQueue("testbrain")
    trainer.subscribe_trajectory_queue(trajectory_queue)
    trainer.publish_policy_queue(policy_queue)

    trajectory = make_fake_trajectory(
        length=15,
        observation_shapes=specs.observation_shapes,
        max_step_complete=True,
        action_spec=specs.action_spec,
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
        observation_shapes=specs.observation_shapes,
        max_step_complete=False,
        action_spec=specs.action_spec,
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
        observation_shapes=specs.observation_shapes,
        action_spec=specs.action_spec,
        max_step_complete=False,
    )
    trajectory_queue.put(trajectory)
    trainer.advance()
    with pytest.raises(AgentManagerQueue.Empty):
        policy_queue.get_nowait()

    # Call add_policy and check that we update the correct number of times.
    # This is to emulate a load from checkpoint.
    behavior_id = BehaviorIdentifiers.from_name_behavior_id(trainer.brain_name)
    policy = trainer.create_policy(behavior_id, specs)
    policy.get_current_step = lambda: 200
    trainer.add_policy(behavior_id, policy)
    trainer.optimizer.update = mock.Mock()
    trainer.model_saver.initialize_or_load(policy)
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

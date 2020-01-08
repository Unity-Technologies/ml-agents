from unittest import mock
import pytest
import yaml

import numpy as np
from mlagents.tf_utils import tf


from mlagents.trainers.sac.models import SACModel
from mlagents.trainers.sac.policy import SACPolicy
from mlagents.trainers.sac.trainer import SACTrainer
from mlagents.trainers.tests import mock_brain as mb
from mlagents.trainers.tests.mock_brain import make_brain_parameters
from mlagents.trainers.tests.test_trajectory import make_fake_trajectory


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
        memory_size: 8
        normalize: false
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
BUFFER_INIT_SAMPLES = 32
NUM_AGENTS = 12


def create_sac_policy_mock(mock_env, dummy_config, use_rnn, use_discrete, use_visual):
    env, mock_brain, _ = mb.setup_mock_env_and_brains(
        mock_env,
        use_discrete,
        use_visual,
        num_agents=NUM_AGENTS,
        vector_action_space=VECTOR_ACTION_SPACE,
        vector_obs_space=VECTOR_OBS_SPACE,
        discrete_action_space=DISCRETE_ACTION_SPACE,
    )

    trainer_parameters = dummy_config
    model_path = env.external_brain_names[0]
    trainer_parameters["model_path"] = model_path
    trainer_parameters["keep_checkpoints"] = 3
    trainer_parameters["use_recurrent"] = use_rnn
    policy = SACPolicy(0, mock_brain, trainer_parameters, False, False)
    return env, policy


@mock.patch("mlagents_envs.environment.UnityEnvironment")
def test_sac_cc_policy(mock_env, dummy_config):
    # Test evaluate
    tf.reset_default_graph()
    env, policy = create_sac_policy_mock(
        mock_env, dummy_config, use_rnn=False, use_discrete=False, use_visual=False
    )
    brain_infos = env.reset()
    brain_info = brain_infos[env.external_brain_names[0]]
    run_out = policy.evaluate(brain_info)
    assert run_out["action"].shape == (NUM_AGENTS, VECTOR_ACTION_SPACE[0])

    # Test update
    update_buffer = mb.simulate_rollout(env, policy, BUFFER_INIT_SAMPLES)
    # Mock out reward signal eval
    update_buffer["extrinsic_rewards"] = update_buffer["rewards"]
    policy.update(update_buffer, num_sequences=update_buffer.num_experiences)
    env.close()


@pytest.mark.parametrize("discrete", [True, False], ids=["discrete", "continuous"])
@mock.patch("mlagents_envs.environment.UnityEnvironment")
def test_sac_update_reward_signals(mock_env, dummy_config, discrete):
    # Test evaluate
    tf.reset_default_graph()
    # Add a Curiosity module
    dummy_config["reward_signals"]["curiosity"] = {}
    dummy_config["reward_signals"]["curiosity"]["strength"] = 1.0
    dummy_config["reward_signals"]["curiosity"]["gamma"] = 0.99
    dummy_config["reward_signals"]["curiosity"]["encoding_size"] = 128
    env, policy = create_sac_policy_mock(
        mock_env, dummy_config, use_rnn=False, use_discrete=discrete, use_visual=False
    )

    # Test update, while removing PPO-specific buffer elements.
    update_buffer = mb.simulate_rollout(
        env, policy, BUFFER_INIT_SAMPLES, exclude_key_list=["advantages", "actions_pre"]
    )

    # Mock out reward signal eval
    update_buffer["extrinsic_rewards"] = update_buffer["rewards"]
    update_buffer["curiosity_rewards"] = update_buffer["rewards"]
    policy.update_reward_signals(
        {"curiosity": update_buffer}, num_sequences=update_buffer.num_experiences
    )
    env.close()


@mock.patch("mlagents_envs.environment.UnityEnvironment")
def test_sac_dc_policy(mock_env, dummy_config):
    # Test evaluate
    tf.reset_default_graph()
    env, policy = create_sac_policy_mock(
        mock_env, dummy_config, use_rnn=False, use_discrete=True, use_visual=False
    )
    brain_infos = env.reset()
    brain_info = brain_infos[env.external_brain_names[0]]
    run_out = policy.evaluate(brain_info)
    assert run_out["action"].shape == (NUM_AGENTS, len(DISCRETE_ACTION_SPACE))

    # Test update
    update_buffer = mb.simulate_rollout(env, policy, BUFFER_INIT_SAMPLES)
    # Mock out reward signal eval
    update_buffer["extrinsic_rewards"] = update_buffer["rewards"]
    policy.update(update_buffer, num_sequences=update_buffer.num_experiences)
    env.close()


@mock.patch("mlagents_envs.environment.UnityEnvironment")
def test_sac_visual_policy(mock_env, dummy_config):
    # Test evaluate
    tf.reset_default_graph()
    env, policy = create_sac_policy_mock(
        mock_env, dummy_config, use_rnn=False, use_discrete=True, use_visual=True
    )
    brain_infos = env.reset()
    brain_info = brain_infos[env.external_brain_names[0]]
    run_out = policy.evaluate(brain_info)
    assert run_out["action"].shape == (NUM_AGENTS, len(DISCRETE_ACTION_SPACE))

    # Test update
    update_buffer = mb.simulate_rollout(env, policy, BUFFER_INIT_SAMPLES)
    # Mock out reward signal eval
    update_buffer["extrinsic_rewards"] = update_buffer["rewards"]
    run_out = policy.update(update_buffer, num_sequences=update_buffer.num_experiences)
    assert type(run_out) is dict


@mock.patch("mlagents_envs.environment.UnityEnvironment")
def test_sac_rnn_policy(mock_env, dummy_config):
    # Test evaluate
    tf.reset_default_graph()
    env, policy = create_sac_policy_mock(
        mock_env, dummy_config, use_rnn=True, use_discrete=True, use_visual=False
    )
    brain_infos = env.reset()
    brain_info = brain_infos[env.external_brain_names[0]]
    run_out = policy.evaluate(brain_info)
    assert run_out["action"].shape == (NUM_AGENTS, len(DISCRETE_ACTION_SPACE))

    # Test update
    update_buffer = mb.simulate_rollout(env, policy, BUFFER_INIT_SAMPLES)
    # Mock out reward signal eval
    update_buffer["extrinsic_rewards"] = update_buffer["rewards"]
    policy.update(update_buffer, num_sequences=2)
    env.close()


def test_sac_model_cc_vector():
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            model = SACModel(
                make_brain_parameters(discrete_action=False, visual_inputs=0)
            )
            init = tf.global_variables_initializer()
            sess.run(init)

            run_list = [model.output, model.value, model.entropy, model.learning_rate]
            feed_dict = {
                model.batch_size: 2,
                model.sequence_length: 1,
                model.vector_in: np.array([[1, 2, 3, 1, 2, 3], [3, 4, 5, 3, 4, 5]]),
            }
            sess.run(run_list, feed_dict=feed_dict)


def test_sac_model_cc_visual():
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            model = SACModel(
                make_brain_parameters(discrete_action=False, visual_inputs=2)
            )
            init = tf.global_variables_initializer()
            sess.run(init)

            run_list = [model.output, model.value, model.entropy, model.learning_rate]
            feed_dict = {
                model.batch_size: 2,
                model.sequence_length: 1,
                model.vector_in: np.array([[1, 2, 3, 1, 2, 3], [3, 4, 5, 3, 4, 5]]),
                model.visual_in[0]: np.ones([2, 40, 30, 3], dtype=np.float32),
                model.visual_in[1]: np.ones([2, 40, 30, 3], dtype=np.float32),
            }
            sess.run(run_list, feed_dict=feed_dict)


def test_sac_model_dc_visual():
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            model = SACModel(
                make_brain_parameters(discrete_action=True, visual_inputs=2)
            )
            init = tf.global_variables_initializer()
            sess.run(init)

            run_list = [model.output, model.value, model.entropy, model.learning_rate]
            feed_dict = {
                model.batch_size: 2,
                model.sequence_length: 1,
                model.vector_in: np.array([[1, 2, 3, 1, 2, 3], [3, 4, 5, 3, 4, 5]]),
                model.visual_in[0]: np.ones([2, 40, 30, 3], dtype=np.float32),
                model.visual_in[1]: np.ones([2, 40, 30, 3], dtype=np.float32),
                model.action_masks: np.ones([2, 2], dtype=np.float32),
            }
            sess.run(run_list, feed_dict=feed_dict)


def test_sac_model_dc_vector():
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            model = SACModel(
                make_brain_parameters(discrete_action=True, visual_inputs=0)
            )
            init = tf.global_variables_initializer()
            sess.run(init)

            run_list = [model.output, model.value, model.entropy, model.learning_rate]
            feed_dict = {
                model.batch_size: 2,
                model.sequence_length: 1,
                model.vector_in: np.array([[1, 2, 3, 1, 2, 3], [3, 4, 5, 3, 4, 5]]),
                model.action_masks: np.ones([2, 2], dtype=np.float32),
            }
            sess.run(run_list, feed_dict=feed_dict)


def test_sac_model_dc_vector_rnn():
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            memory_size = 128
            model = SACModel(
                make_brain_parameters(discrete_action=True, visual_inputs=0),
                use_recurrent=True,
                m_size=memory_size,
            )
            init = tf.global_variables_initializer()
            sess.run(init)

            run_list = [
                model.output,
                model.all_log_probs,
                model.value,
                model.entropy,
                model.learning_rate,
                model.memory_out,
            ]
            feed_dict = {
                model.batch_size: 1,
                model.sequence_length: 2,
                model.prev_action: [[0], [0]],
                model.memory_in: np.zeros((1, memory_size), dtype=np.float32),
                model.vector_in: np.array([[1, 2, 3, 1, 2, 3], [3, 4, 5, 3, 4, 5]]),
                model.action_masks: np.ones([1, 2], dtype=np.float32),
            }
            sess.run(run_list, feed_dict=feed_dict)


def test_sac_model_cc_vector_rnn():
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            memory_size = 128
            model = SACModel(
                make_brain_parameters(discrete_action=False, visual_inputs=0),
                use_recurrent=True,
                m_size=memory_size,
            )
            init = tf.global_variables_initializer()
            sess.run(init)

            run_list = [
                model.output,
                model.all_log_probs,
                model.value,
                model.entropy,
                model.learning_rate,
                model.memory_out,
            ]
            feed_dict = {
                model.batch_size: 1,
                model.sequence_length: 2,
                model.memory_in: np.zeros((1, memory_size), dtype=np.float32),
                model.vector_in: np.array([[1, 2, 3, 1, 2, 3], [3, 4, 5, 3, 4, 5]]),
            }
            sess.run(run_list, feed_dict=feed_dict)


def test_sac_save_load_buffer(tmpdir, dummy_config):
    env, mock_brain, _ = mb.setup_mock_env_and_brains(
        mock.Mock(),
        False,
        False,
        num_agents=NUM_AGENTS,
        vector_action_space=VECTOR_ACTION_SPACE,
        vector_obs_space=VECTOR_OBS_SPACE,
        discrete_action_space=DISCRETE_ACTION_SPACE,
    )
    trainer_params = dummy_config
    trainer_params["summary_path"] = str(tmpdir)
    trainer_params["model_path"] = str(tmpdir)
    trainer_params["save_replay_buffer"] = True
    trainer = SACTrainer(mock_brain.brain_name, 1, trainer_params, True, False, 0, 0)
    policy = trainer.create_policy(mock_brain)
    trainer.add_policy(mock_brain.brain_name, policy)

    trainer.update_buffer = mb.simulate_rollout(
        env, trainer.policy, BUFFER_INIT_SAMPLES
    )
    buffer_len = trainer.update_buffer.num_experiences
    trainer.save_model(mock_brain.brain_name)

    # Wipe Trainer and try to load
    trainer2 = SACTrainer(mock_brain.brain_name, 1, trainer_params, True, True, 0, 0)

    policy = trainer2.create_policy(mock_brain)
    trainer2.add_policy(mock_brain.brain_name, policy)
    assert trainer2.update_buffer.num_experiences == buffer_len


def test_process_trajectory(dummy_config):
    brain_params = make_brain_parameters(
        discrete_action=False, visual_inputs=0, vec_obs_size=6
    )
    dummy_config["summary_path"] = "./summaries/test_trainer_summary"
    dummy_config["model_path"] = "./models/test_trainer_models/TestModel"
    trainer = SACTrainer(brain_params, 0, dummy_config, True, False, 0, "0")
    policy = trainer.create_policy(brain_params)
    trainer.add_policy(brain_params.brain_name, policy)

    trajectory = make_fake_trajectory(
        length=15, max_step_complete=True, vec_obs_size=6, num_vis_obs=0, action_space=2
    )
    trainer.process_trajectory(trajectory)

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
        action_space=2,
    )
    trainer.process_trajectory(trajectory)

    # Check that the stats are reset as episode is finished
    for reward in trainer.collected_rewards.values():
        for agent in reward.values():
            assert agent == 0
    assert trainer.stats_reporter.get_stats_summaries("Policy/Extrinsic Reward").num > 0


if __name__ == "__main__":
    pytest.main()

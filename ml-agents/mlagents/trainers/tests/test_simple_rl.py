import math
import tempfile
import pytest
import yaml

from mlagents.trainers.tests.simple_test_envs import Simple1DEnvironment
from mlagents.trainers.trainer_controller import TrainerController
from mlagents.trainers.trainer_util import TrainerFactory
from mlagents.trainers.simple_env_manager import SimpleEnvManager
from mlagents.trainers.sampler_class import SamplerManager
from mlagents.trainers.stats import StatsReporter
from mlagents_envs.side_channel.float_properties_channel import FloatPropertiesChannel

BRAIN_NAME = "1D"  # __name__

PPO_CONFIG = f"""
    {BRAIN_NAME}:
        trainer: ppo
        batch_size: 16
        beta: 5.0e-3
        buffer_size: 64
        epsilon: 0.2
        hidden_units: 128
        lambd: 0.95
        learning_rate: 5.0e-3
        max_steps: 2500
        memory_size: 256
        normalize: false
        num_epoch: 3
        num_layers: 2
        time_horizon: 64
        sequence_length: 64
        summary_freq: 500
        use_recurrent: false
        reward_signals:
            extrinsic:
                strength: 1.0
                gamma: 0.99
    """

SAC_CONFIG = f"""
    {BRAIN_NAME}:
        trainer: sac
        batch_size: 8
        buffer_size: 500
        buffer_init_steps: 100
        hidden_units: 16
        init_entcoef: 0.01
        learning_rate: 5.0e-3
        max_steps: 1000
        memory_size: 256
        normalize: false
        num_update: 1
        train_interval: 1
        num_layers: 1
        time_horizon: 64
        sequence_length: 64
        summary_freq: 100
        tau: 0.01
        use_recurrent: false
        curiosity_enc_size: 128
        demo_path: None
        vis_encode_type: simple
        reward_signals:
            extrinsic:
                strength: 1.0
                gamma: 0.99
    """

GHOST_CONFIG = f"""
    {BRAIN_NAME}:
        trainer: ppo
        batch_size: 16
        beta: 5.0e-3
        buffer_size: 64
        epsilon: 0.2
        hidden_units: 128
        lambd: 0.95
        learning_rate: 5.0e-3
        max_steps: 2500
        memory_size: 256
        normalize: false
        num_epoch: 3
        num_layers: 2
        time_horizon: 64
        sequence_length: 64
        summary_freq: 500
        use_recurrent: false
        reward_signals:
            extrinsic:
                strength: 1.0
                gamma: 0.99
        self_play:
            play_against_current_self_ratio: 1.0
            save_steps: 2000
            swap_steps: 2000
    """


def _check_environment_trains(env, config, meta_curriculum=None, success_threshold=1.0):
    # Create controller and begin training.
    with tempfile.TemporaryDirectory() as dir:
        run_id = "id"
        save_freq = 99999
        seed = 1337
        StatsReporter.writers.clear()  # Clear StatsReporters so we don't write to file
        trainer_config = yaml.safe_load(config)
        env_manager = SimpleEnvManager(env, FloatPropertiesChannel())
        trainer_factory = TrainerFactory(
            trainer_config=trainer_config,
            summaries_dir=dir,
            run_id=run_id,
            model_path=dir,
            keep_checkpoints=1,
            train_model=True,
            load_model=False,
            seed=seed,
            meta_curriculum=meta_curriculum,
            multi_gpu=False,
        )

        tc = TrainerController(
            trainer_factory=trainer_factory,
            summaries_dir=dir,
            model_path=dir,
            run_id=run_id,
            meta_curriculum=meta_curriculum,
            train=True,
            training_seed=seed,
            sampler_manager=SamplerManager(None),
            resampling_interval=None,
            save_freq=save_freq,
        )

        # Begin training
        tc.start_learning(env_manager)
        if (
            success_threshold is not None
        ):  # For tests where we are just checking setup and not reward

            print(env.final_rewards.values())
            # for mean_reward in tc._get_measure_vals().values():
            for name, mean_reward in env.final_rewards.items():
                print(name)
                assert not math.isnan(mean_reward)
                assert mean_reward > success_threshold


@pytest.mark.parametrize("use_discrete", [True, False])
def test_simple_ppo(use_discrete):
    env = Simple1DEnvironment([BRAIN_NAME], use_discrete=use_discrete)
    _check_environment_trains(env, PPO_CONFIG)


@pytest.mark.parametrize("use_discrete", [True, False])
def test_simple_sac(use_discrete):
    env = Simple1DEnvironment([BRAIN_NAME], use_discrete=use_discrete)
    _check_environment_trains(env, SAC_CONFIG)


@pytest.mark.parametrize("use_discrete", [True, False])
def test_simple_ghost(use_discrete):
    env = Simple1DEnvironment(
        [BRAIN_NAME + "?team=0", BRAIN_NAME + "?team=1"], use_discrete=use_discrete
    )
    _check_environment_trains(env, GHOST_CONFIG)

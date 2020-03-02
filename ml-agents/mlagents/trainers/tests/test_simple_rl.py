import math
import tempfile
import pytest
import yaml
import numpy as np
from typing import Dict

from mlagents.trainers.tests.simple_test_envs import (
    Simple1DEnvironment,
    Memory1DEnvironment,
)
from mlagents.trainers.trainer_controller import TrainerController
from mlagents.trainers.trainer_util import TrainerFactory
from mlagents.trainers.simple_env_manager import SimpleEnvManager
from mlagents.trainers.sampler_class import SamplerManager
from mlagents.trainers.stats import StatsReporter, StatsWriter, StatsSummary
from mlagents_envs.side_channel.float_properties_channel import FloatPropertiesChannel

BRAIN_NAME = "1D"

PPO_CONFIG = f"""
    {BRAIN_NAME}:
        trainer: ppo
        batch_size: 16
        beta: 5.0e-3
        buffer_size: 64
        epsilon: 0.2
        hidden_units: 32
        lambd: 0.95
        learning_rate: 5.0e-3
        learning_rate_schedule: constant
        max_steps: 1500
        memory_size: 256
        normalize: false
        num_epoch: 3
        num_layers: 1
        time_horizon: 64
        sequence_length: 64
        summary_freq: 500
        use_recurrent: false
        reward_signals:
            extrinsic:
                strength: 1.0
                gamma: 0.99
    """

PPO_CONFIG_RECURRENT = f"""
    {BRAIN_NAME}:
        trainer: ppo
        batch_size: 16
        beta: 5.0e-3
        buffer_size: 64
        epsilon: 0.2
        hidden_units: 32
        lambd: 0.95
        learning_rate: 5.0e-3
        max_steps: 4000
        memory_size: 8
        normalize: false
        learning_rate_schedule: constant
        num_epoch: 3
        num_layers: 1
        time_horizon: 64
        sequence_length: 64
        summary_freq: 500
        use_recurrent: true
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

SAC_CONFIG_RECURRENT = f"""
    {BRAIN_NAME}:
        trainer: sac
        batch_size: 8
        buffer_size: 500
        buffer_init_steps: 100
        hidden_units: 16
        init_entcoef: 0.01
        learning_rate: 5.0e-3
        max_steps: 1000
        memory_size: 16
        normalize: false
        num_update: 1
        train_interval: 1
        num_layers: 1
        time_horizon: 64
        sequence_length: 16
        summary_freq: 100
        tau: 0.01
        use_recurrent: true
        curiosity_enc_size: 128
        demo_path: None
        vis_encode_type: simple
        reward_signals:
            extrinsic:
                strength: 1.0
                gamma: 0.99
    """

GHOST_CONFIG_PASS = f"""
    {BRAIN_NAME}:
        trainer: ppo
        batch_size: 16
        beta: 5.0e-3
        buffer_size: 64
        epsilon: 0.2
        hidden_units: 32
        lambd: 0.95
        learning_rate: 5.0e-3
        max_steps: 2500
        memory_size: 256
        normalize: false
        num_epoch: 3
        num_layers: 1
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

# This config should fail because the ghosted policy is never swapped with a competent policy.
# Swap occurs after max step is reached.
GHOST_CONFIG_FAIL = f"""
    {BRAIN_NAME}:
        trainer: ppo
        batch_size: 16
        beta: 5.0e-3
        buffer_size: 64
        epsilon: 0.2
        hidden_units: 32
        lambd: 0.95
        learning_rate: 5.0e-3
        max_steps: 2500
        memory_size: 256
        normalize: false
        num_epoch: 3
        num_layers: 1
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
            swap_steps: 4000
    """


# The reward processor is passed as an argument to _check_environment_trains.
# It is applied to the list pf all final rewards for each brain individually.
# This is so that we can process all final rewards in different ways for different algorithms.
# Custom reward processors shuld be built within the test function and passed to _check_environment_trains
# Default is average over the last 5 final rewards
def default_reward_processor(rewards, last_n_rewards=5):
    return np.array(rewards[-last_n_rewards:], dtype=np.float32).mean()


class DebugWriter(StatsWriter):
    """
    Print to stdout so stats can be viewed in pytest
    """

    def write_stats(
        self, category: str, values: Dict[str, StatsSummary], step: int
    ) -> None:
        for val, stats_summary in values.items():
            if val == "Environment/Cumulative Reward":
                print(step, val, stats_summary.mean)

    def write_text(self, category: str, text: str, step: int) -> None:
        pass


def _check_environment_trains(
    env,
    config,
    reward_processor=default_reward_processor,
    meta_curriculum=None,
    success_threshold=0.99,
):
    # Create controller and begin training.
    with tempfile.TemporaryDirectory() as dir:
        run_id = "id"
        save_freq = 99999
        seed = 1337
        StatsReporter.writers.clear()  # Clear StatsReporters so we don't write to file
        debug_writer = DebugWriter()
        StatsReporter.add_writer(debug_writer)
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

            processed_rewards = [
                reward_processor(rewards) for rewards in env.final_rewards.values()
            ]
            assert all(not math.isnan(reward) for reward in processed_rewards)
            assert all(reward > success_threshold for reward in processed_rewards)


@pytest.mark.parametrize("use_discrete", [True, False])
def test_simple_ppo(use_discrete):
    env = Simple1DEnvironment([BRAIN_NAME], use_discrete=use_discrete)
    _check_environment_trains(env, PPO_CONFIG)


@pytest.mark.parametrize("use_discrete", [True, False])
def test_recurrent_ppo(use_discrete):
    env = Memory1DEnvironment([BRAIN_NAME], use_discrete=use_discrete)
    _check_environment_trains(env, PPO_CONFIG_RECURRENT)


@pytest.mark.parametrize("use_discrete", [True, False])
def test_recurrent_sac(use_discrete):
    env = Memory1DEnvironment([BRAIN_NAME], use_discrete=use_discrete)
    _check_environment_trains(env, SAC_CONFIG_RECURRENT)


@pytest.mark.parametrize("use_discrete", [True, False])
def test_simple_sac(use_discrete):
    env = Simple1DEnvironment([BRAIN_NAME], use_discrete=use_discrete)
    _check_environment_trains(env, SAC_CONFIG)


@pytest.mark.parametrize("use_discrete", [True, False])
def test_simple_ghost(use_discrete):
    env = Simple1DEnvironment(
        [BRAIN_NAME + "?team=0", BRAIN_NAME + "?team=1"], use_discrete=use_discrete
    )
    _check_environment_trains(env, GHOST_CONFIG_PASS)


@pytest.mark.parametrize("use_discrete", [True, False])
def test_simple_ghost_fails(use_discrete):
    env = Simple1DEnvironment(
        [BRAIN_NAME + "?team=0", BRAIN_NAME + "?team=1"], use_discrete=use_discrete
    )
    _check_environment_trains(env, GHOST_CONFIG_FAIL, success_threshold=None)
    processed_rewards = [
        default_reward_processor(rewards) for rewards in env.final_rewards.values()
    ]
    success_threshold = 0.99
    assert any(reward > success_threshold for reward in processed_rewards) and any(
        reward < success_threshold for reward in processed_rewards
    )

import math
import tempfile
import pytest
import yaml
import numpy as np
from typing import Dict, Any

from mlagents.trainers.tests.simple_test_envs import (
    Simple1DEnvironment,
    Memory1DEnvironment,
    Record1DEnvironment,
)
from mlagents.trainers.trainer_controller import TrainerController
from mlagents.trainers.trainer_util import TrainerFactory
from mlagents.trainers.simple_env_manager import SimpleEnvManager
from mlagents.trainers.sampler_class import SamplerManager
from mlagents.trainers.demo_loader import write_demo
from mlagents.trainers.stats import StatsReporter, StatsWriter, StatsSummary
from mlagents_envs.side_channel.float_properties_channel import FloatPropertiesChannel
from mlagents_envs.communicator_objects.demonstration_meta_pb2 import (
    DemonstrationMetaProto,
)
from mlagents_envs.communicator_objects.brain_parameters_pb2 import BrainParametersProto
from mlagents_envs.communicator_objects.space_type_pb2 import discrete, continuous

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
        max_steps: 2000
        memory_size: 16
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
        memory_size: 16
        normalize: false
        num_update: 1
        train_interval: 1
        num_layers: 1
        time_horizon: 64
        sequence_length: 32
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


def generate_config(
    config: str, override_vals: Dict[str, Any] = None
) -> Dict[str, Any]:
    trainer_config = yaml.safe_load(config)
    if override_vals is not None:
        trainer_config[BRAIN_NAME].update(override_vals)
    return trainer_config


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

    def __init__(self):
        self._last_reward_summary: Dict[str, float] = {}

    def get_last_rewards(self):
        return self._last_reward_summary

    def write_stats(
        self, category: str, values: Dict[str, StatsSummary], step: int
    ) -> None:
        for val, stats_summary in values.items():
            if val == "Environment/Cumulative Reward":
                print(step, val, stats_summary.mean)
                self._last_reward_summary[category] = stats_summary.mean


def _check_environment_trains(
    env,
    trainer_config,
    reward_processor=default_reward_processor,
    meta_curriculum=None,
    success_threshold=0.99,
    env_manager=None,
):
    # Create controller and begin training.
    with tempfile.TemporaryDirectory() as dir:
        run_id = "id"
        save_freq = 99999
        seed = 1337
        StatsReporter.writers.clear()  # Clear StatsReporters so we don't write to file
        debug_writer = DebugWriter()
        StatsReporter.add_writer(debug_writer)
        if env_manager is None:
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
    config = generate_config(PPO_CONFIG)
    _check_environment_trains(env, config)


@pytest.mark.parametrize("use_discrete", [True, False])
@pytest.mark.parametrize("num_visual", [1, 2])
def test_visual_ppo(num_visual, use_discrete):
    env = Simple1DEnvironment(
        [BRAIN_NAME],
        use_discrete=use_discrete,
        num_visual=num_visual,
        num_vector=0,
        step_size=0.2,
    )
    override_vals = {"learning_rate": 3.0e-4}
    config = generate_config(PPO_CONFIG, override_vals)
    _check_environment_trains(env, config)


@pytest.mark.parametrize("num_visual", [1, 2])
@pytest.mark.parametrize("vis_encode_type", ["resnet", "nature_cnn"])
def test_visual_advanced_ppo(vis_encode_type, num_visual):
    env = Simple1DEnvironment(
        [BRAIN_NAME],
        use_discrete=True,
        num_visual=num_visual,
        num_vector=0,
        step_size=0.5,
        vis_obs_size=(36, 36, 3),
    )
    override_vals = {
        "learning_rate": 3.0e-4,
        "vis_encode_type": vis_encode_type,
        "max_steps": 500,
        "summary_freq": 100,
    }
    config = generate_config(PPO_CONFIG, override_vals)
    # The number of steps is pretty small for these encoders
    _check_environment_trains(env, config, success_threshold=0.5)


@pytest.mark.parametrize("use_discrete", [True, False])
def test_recurrent_ppo(use_discrete):
    env = Memory1DEnvironment([BRAIN_NAME], use_discrete=use_discrete)
    override_vals = {
        "max_steps": 3000,
        "batch_size": 64,
        "buffer_size": 128,
        "use_recurrent": True,
    }
    config = generate_config(PPO_CONFIG, override_vals)
    _check_environment_trains(env, config)


@pytest.mark.parametrize("use_discrete", [True, False])
def test_simple_sac(use_discrete):
    env = Simple1DEnvironment([BRAIN_NAME], use_discrete=use_discrete)
    config = generate_config(SAC_CONFIG)
    _check_environment_trains(env, config)


@pytest.mark.parametrize("use_discrete", [True, False])
@pytest.mark.parametrize("num_visual", [1, 2])
def test_visual_sac(num_visual, use_discrete):
    env = Simple1DEnvironment(
        [BRAIN_NAME],
        use_discrete=use_discrete,
        num_visual=num_visual,
        num_vector=0,
        step_size=0.2,
    )
    override_vals = {"batch_size": 16, "learning_rate": 3e-4}
    config = generate_config(SAC_CONFIG, override_vals)
    _check_environment_trains(env, config)


@pytest.mark.parametrize("num_visual", [1, 2])
@pytest.mark.parametrize("vis_encode_type", ["resnet", "nature_cnn"])
def test_visual_advanced_sac(vis_encode_type, num_visual):
    env = Simple1DEnvironment(
        [BRAIN_NAME],
        use_discrete=True,
        num_visual=num_visual,
        num_vector=0,
        step_size=0.5,
        vis_obs_size=(36, 36, 3),
    )
    override_vals = {
        "batch_size": 16,
        "learning_rate": 3.0e-4,
        "vis_encode_type": vis_encode_type,
        "buffer_init_steps": 0,
        "max_steps": 100,
    }
    config = generate_config(SAC_CONFIG, override_vals)
    # The number of steps is pretty small for these encoders
    _check_environment_trains(env, config, success_threshold=0.5)


@pytest.mark.parametrize("use_discrete", [True, False])
def test_recurrent_sac(use_discrete):
    env = Memory1DEnvironment([BRAIN_NAME], use_discrete=use_discrete)
    override_vals = {"batch_size": 32, "use_recurrent": True, "max_steps": 2000}
    config = generate_config(SAC_CONFIG, override_vals)
    _check_environment_trains(env, config)


@pytest.mark.parametrize("use_discrete", [True, False])
def test_simple_ghost(use_discrete):
    env = Simple1DEnvironment(
        [BRAIN_NAME + "?team=0", BRAIN_NAME + "?team=1"], use_discrete=use_discrete
    )
    override_vals = {
        "max_steps": 2500,
        "self_play": {
            "play_against_current_self_ratio": 1.0,
            "save_steps": 2000,
            "swap_steps": 2000,
        },
    }
    config = generate_config(PPO_CONFIG, override_vals)
    _check_environment_trains(env, config)


@pytest.mark.parametrize("use_discrete", [True, False])
def test_simple_ghost_fails(use_discrete):
    env = Simple1DEnvironment(
        [BRAIN_NAME + "?team=0", BRAIN_NAME + "?team=1"], use_discrete=use_discrete
    )
    # This config should fail because the ghosted policy is never swapped with a competent policy.
    # Swap occurs after max step is reached.
    override_vals = {
        "max_steps": 2500,
        "self_play": {
            "play_against_current_self_ratio": 1.0,
            "save_steps": 2000,
            "swap_steps": 4000,
        },
    }
    config = generate_config(PPO_CONFIG, override_vals)
    _check_environment_trains(env, config, success_threshold=None)
    processed_rewards = [
        default_reward_processor(rewards) for rewards in env.final_rewards.values()
    ]
    success_threshold = 0.99
    assert any(reward > success_threshold for reward in processed_rewards) and any(
        reward < success_threshold for reward in processed_rewards
    )


@pytest.fixture(scope="session")
def simple_record(tmpdir_factory):
    def record_demo(use_discrete, num_visual=0, num_vector=1):
        env = Record1DEnvironment(
            [BRAIN_NAME],
            use_discrete=use_discrete,
            num_visual=num_visual,
            num_vector=num_vector,
            n_demos=100,
        )
        # If we want to use true demos, we can solve the env in the usual way
        # Otherwise, we can just call solve to execute the optimal policy
        env.solve()
        agent_info_protos = env.demonstration_protos[BRAIN_NAME]
        meta_data_proto = DemonstrationMetaProto()
        brain_param_proto = BrainParametersProto(
            vector_action_size=[1],
            vector_action_descriptions=[""],
            vector_action_space_type=discrete if use_discrete else continuous,
            brain_name=BRAIN_NAME,
            is_training=True,
        )
        action_type = "Discrete" if use_discrete else "Continuous"
        demo_path_name = "1DTest" + action_type + ".demo"
        demo_path = str(tmpdir_factory.mktemp("tmp_demo").join(demo_path_name))
        write_demo(demo_path, meta_data_proto, brain_param_proto, agent_info_protos)
        return demo_path

    return record_demo


@pytest.mark.parametrize("use_discrete", [True, False])
@pytest.mark.parametrize("trainer_config", [PPO_CONFIG, SAC_CONFIG])
def test_gail(simple_record, use_discrete, trainer_config):
    demo_path = simple_record(use_discrete)
    env = Simple1DEnvironment([BRAIN_NAME], use_discrete=use_discrete, step_size=0.2)
    override_vals = {
        "max_steps": 500,
        "behavioral_cloning": {"demo_path": demo_path, "strength": 1.0, "steps": 1000},
        "reward_signals": {
            "gail": {
                "strength": 1.0,
                "gamma": 0.99,
                "encoding_size": 32,
                "demo_path": demo_path,
            }
        },
    }
    config = generate_config(trainer_config, override_vals)
    _check_environment_trains(env, config, success_threshold=0.9)


@pytest.mark.parametrize("use_discrete", [True, False])
def test_gail_visual_ppo(simple_record, use_discrete):
    demo_path = simple_record(use_discrete, num_visual=1, num_vector=0)
    env = Simple1DEnvironment(
        [BRAIN_NAME],
        num_visual=1,
        num_vector=0,
        use_discrete=use_discrete,
        step_size=0.2,
    )
    override_vals = {
        "max_steps": 1000,
        "learning_rate": 3.0e-4,
        "behavioral_cloning": {"demo_path": demo_path, "strength": 1.0, "steps": 1000},
        "reward_signals": {
            "gail": {
                "strength": 1.0,
                "gamma": 0.99,
                "encoding_size": 32,
                "demo_path": demo_path,
            }
        },
    }
    config = generate_config(PPO_CONFIG, override_vals)
    _check_environment_trains(env, config, success_threshold=0.9)


@pytest.mark.parametrize("use_discrete", [True, False])
def test_gail_visual_sac(simple_record, use_discrete):
    demo_path = simple_record(use_discrete, num_visual=1, num_vector=0)
    env = Simple1DEnvironment(
        [BRAIN_NAME],
        num_visual=1,
        num_vector=0,
        use_discrete=use_discrete,
        step_size=0.2,
    )
    override_vals = {
        "max_steps": 500,
        "batch_size": 16,
        "learning_rate": 3.0e-4,
        "behavioral_cloning": {"demo_path": demo_path, "strength": 1.0, "steps": 1000},
        "reward_signals": {
            "gail": {
                "strength": 1.0,
                "gamma": 0.99,
                "encoding_size": 32,
                "demo_path": demo_path,
            }
        },
    }
    config = generate_config(SAC_CONFIG, override_vals)
    _check_environment_trains(env, config, success_threshold=0.9)

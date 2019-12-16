import math
import random
import tempfile
import pytest
import yaml
from typing import Dict
import numpy as np


from mlagents.trainers.trainer_controller import TrainerController
from mlagents.trainers.trainer_util import TrainerFactory
from mlagents_envs.base_env import (
    BaseEnv,
    AgentGroupSpec,
    BatchedStepResult,
    ActionType,
)
from mlagents.trainers.brain import BrainParameters
from mlagents.trainers.simple_env_manager import SimpleEnvManager
from mlagents.trainers.sampler_class import SamplerManager
from mlagents.trainers.stats import StatsReporter
from mlagents_envs.side_channel.float_properties_channel import FloatPropertiesChannel

BRAIN_NAME = __name__
OBS_SIZE = 1
STEP_SIZE = 0.1

TIME_PENALTY = 0.001
MIN_STEPS = int(1.0 / STEP_SIZE) + 1
SUCCESS_REWARD = 1.0 + MIN_STEPS * TIME_PENALTY


def clamp(x, min_val, max_val):
    return max(min_val, min(x, max_val))


class Simple1DEnvironment(BaseEnv):
    """
    Very simple "game" - the agent has a position on [-1, 1], gets a reward of 1 if it reaches 1, and a reward of -1 if
    it reaches -1. The position is incremented by the action amount (clamped to [-step_size, step_size]).
    """

    def __init__(self, use_discrete):
        super().__init__()
        self.discrete = use_discrete
        action_type = ActionType.DISCRETE if use_discrete else ActionType.CONTINUOUS
        self.group_spec = AgentGroupSpec(
            [(OBS_SIZE,)], action_type, (2,) if use_discrete else 1
        )
        # state
        self.position = 0.0
        self.step_count = 0
        self.random = random.Random(str(self.group_spec))
        self.goal = self.random.choice([-1, 1])
        self.action = None
        self.step_result = None

    def get_agent_groups(self):
        return [BRAIN_NAME]

    def get_agent_group_spec(self, name):
        return self.group_spec

    def set_action_for_agent(self, name, id, data):
        pass

    def set_actions(self, name, data):
        self.action = data

    def get_step_result(self, name):
        return self.step_result

    def step(self) -> None:
        assert self.action is not None

        if self.discrete:
            act = self.action[0][0]
            delta = 1 if act else -1
        else:
            delta = self.action[0][0]
        delta = clamp(delta, -STEP_SIZE, STEP_SIZE)
        self.position += delta
        self.position = clamp(self.position, -1, 1)
        self.step_count += 1
        done = self.position >= 1.0 or self.position <= -1.0
        if done:
            reward = SUCCESS_REWARD * self.position * self.goal
        else:
            reward = -TIME_PENALTY

        m_vector_obs = [np.ones((1, OBS_SIZE), dtype=np.float32) * self.goal]
        m_reward = np.array([reward], dtype=np.float32)
        m_done = np.array([done], dtype=np.bool)
        m_agent_id = np.array([0], dtype=np.int32)

        if done:
            self._reset_agent()

        self.step_result = BatchedStepResult(
            m_vector_obs, m_reward, m_done, m_done, m_agent_id, None
        )

    def _reset_agent(self):
        self.position = 0.0
        self.step_count = 0
        self.goal = self.random.choice([-1, 1])

    def reset(self) -> None:  # type: ignore
        self._reset_agent()

        m_vector_obs = [np.ones((1, OBS_SIZE), dtype=np.float32) * self.goal]
        m_reward = np.array([0], dtype=np.float32)
        m_done = np.array([False], dtype=np.bool)
        m_agent_id = np.array([0], dtype=np.int32)

        self.step_result = BatchedStepResult(
            m_vector_obs, m_reward, m_done, m_done, m_agent_id, None
        )

    @property
    def external_brains(self) -> Dict[str, BrainParameters]:
        return self._brains

    @property
    def reset_parameters(self) -> Dict[str, str]:
        return {}

    def close(self):
        pass


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
        summary_freq: 500
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


def _check_environment_trains(
    env, config, meta_curriculum=None, success_threshold=0.99
):
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
        print(tc._get_measure_vals())
        for brain_name, mean_reward in tc._get_measure_vals().items():
            assert not math.isnan(mean_reward)
            assert mean_reward > success_threshold


@pytest.mark.parametrize("use_discrete", [True, False])
def test_simple_ppo(use_discrete):
    env = Simple1DEnvironment(use_discrete=use_discrete)
    _check_environment_trains(env, PPO_CONFIG)


@pytest.mark.parametrize("use_discrete", [True, False])
def test_simple_sac(use_discrete):
    env = Simple1DEnvironment(use_discrete=use_discrete)
    _check_environment_trains(env, SAC_CONFIG)

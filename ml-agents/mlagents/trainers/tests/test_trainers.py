from unittest.mock import patch

import pytest

from mlagents.trainers.agent_processor import AgentManagerQueue
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.environment_parameter_manager import EnvironmentParameterManager
from mlagents.trainers.settings import RunOptions
from mlagents.trainers.tests import mock_brain as mb
from mlagents.trainers.tests.dummy_config import (
    create_observation_specs_with_shapes,
    ppo_dummy_config,
    poca_dummy_config,
    sac_dummy_config,
)
from mlagents.trainers.tests.mock_brain import make_fake_trajectory
from mlagents.trainers.trainer import TrainerFactory


@pytest.fixture
def ppo_config():
    return RunOptions(behaviors={"test_brain": ppo_dummy_config()})


@pytest.fixture
def sac_config():
    return RunOptions(behaviors={"test_brain": sac_dummy_config()})


@pytest.fixture
def poca_config():
    return RunOptions(behaviors={"test_brain": poca_dummy_config()})


def test_ppo_trainer_update_normalization(ppo_config):
    behavior_id_team0 = "test_brain?team=0"
    brain_name = BehaviorIdentifiers.from_name_behavior_id(behavior_id_team0).brain_name
    mock_specs = mb.setup_test_behavior_specs(
        True, False, vector_action_space=[2], vector_obs_space=1
    )
    base_config = ppo_config.behaviors
    output_path = "results_dir"
    train_model = True
    load_model = False
    seed = 42
    trainer_factory = TrainerFactory(
        trainer_config=base_config,
        output_path=output_path,
        train_model=train_model,
        load_model=load_model,
        seed=seed,
        param_manager=EnvironmentParameterManager(),
    )
    ppo_trainer = trainer_factory.generate(brain_name)
    parsed_behavior_id0 = BehaviorIdentifiers.from_name_behavior_id(behavior_id_team0)
    policy = ppo_trainer.create_policy(parsed_behavior_id0, mock_specs)
    ppo_trainer.add_policy(parsed_behavior_id0, policy)
    trajectory_queue0 = AgentManagerQueue(behavior_id_team0)
    ppo_trainer.subscribe_trajectory_queue(trajectory_queue0)
    time_horizon = 15
    trajectory = make_fake_trajectory(
        length=time_horizon,
        max_step_complete=True,
        observation_specs=create_observation_specs_with_shapes([(1,)]),
        action_spec=mock_specs.action_spec,
    )
    trajectory_queue0.put(trajectory)
    # mocking out update_normalization in both the policy and critic
    with patch(
        "mlagents.trainers.torch_entities.networks.ValueNetwork.update_normalization"
    ) as optimizer_update_normalization_mock, patch(
        "mlagents.trainers.torch_entities.networks.SimpleActor.update_normalization"
    ) as policy_update_normalization_mock:
        ppo_trainer.advance()
        optimizer_update_normalization_mock.assert_called_once()
        policy_update_normalization_mock.assert_called_once()


def test_sac_trainer_update_normalization(sac_config):
    behavior_id_team0 = "test_brain?team=0"
    brain_name = BehaviorIdentifiers.from_name_behavior_id(behavior_id_team0).brain_name
    mock_specs = mb.setup_test_behavior_specs(
        True, False, vector_action_space=[2], vector_obs_space=1
    )
    base_config = sac_config.behaviors
    output_path = "results_dir"
    train_model = True
    load_model = False
    seed = 42
    trainer_factory = TrainerFactory(
        trainer_config=base_config,
        output_path=output_path,
        train_model=train_model,
        load_model=load_model,
        seed=seed,
        param_manager=EnvironmentParameterManager(),
    )
    sac_trainer = trainer_factory.generate(brain_name)
    parsed_behavior_id0 = BehaviorIdentifiers.from_name_behavior_id(behavior_id_team0)
    policy = sac_trainer.create_policy(parsed_behavior_id0, mock_specs)
    sac_trainer.add_policy(parsed_behavior_id0, policy)
    trajectory_queue0 = AgentManagerQueue(behavior_id_team0)
    sac_trainer.subscribe_trajectory_queue(trajectory_queue0)
    time_horizon = 15
    trajectory = make_fake_trajectory(
        length=time_horizon,
        max_step_complete=True,
        observation_specs=create_observation_specs_with_shapes([(1,)]),
        action_spec=mock_specs.action_spec,
    )
    trajectory_queue0.put(trajectory)
    # mocking out update_normalization in both the policy and critic
    with patch(
        "mlagents.trainers.torch_entities.networks.ValueNetwork.update_normalization"
    ) as optimizer_update_normalization_mock, patch(
        "mlagents.trainers.torch_entities.networks.SimpleActor.update_normalization"
    ) as policy_update_normalization_mock:
        sac_trainer.advance()
        optimizer_update_normalization_mock.assert_called_once()
        policy_update_normalization_mock.assert_called_once()


def test_poca_trainer_update_normalization(poca_config):
    behavior_id_team0 = "test_brain?team=0"
    brain_name = BehaviorIdentifiers.from_name_behavior_id(behavior_id_team0).brain_name
    mock_specs = mb.setup_test_behavior_specs(
        True, False, vector_action_space=[2], vector_obs_space=1
    )
    base_config = poca_config.behaviors
    output_path = "results_dir"
    train_model = True
    load_model = False
    seed = 42
    trainer_factory = TrainerFactory(
        trainer_config=base_config,
        output_path=output_path,
        train_model=train_model,
        load_model=load_model,
        seed=seed,
        param_manager=EnvironmentParameterManager(),
    )
    poca_trainer = trainer_factory.generate(brain_name)
    parsed_behavior_id0 = BehaviorIdentifiers.from_name_behavior_id(behavior_id_team0)
    policy = poca_trainer.create_policy(parsed_behavior_id0, mock_specs)
    poca_trainer.add_policy(parsed_behavior_id0, policy)
    trajectory_queue0 = AgentManagerQueue(behavior_id_team0)
    poca_trainer.subscribe_trajectory_queue(trajectory_queue0)
    time_horizon = 15
    trajectory = make_fake_trajectory(
        length=time_horizon,
        max_step_complete=True,
        observation_specs=create_observation_specs_with_shapes([(1,)]),
        action_spec=mock_specs.action_spec,
    )
    trajectory_queue0.put(trajectory)
    # mocking out update_normalization in both the policy and critic
    with patch(
        "mlagents.trainers.poca.optimizer_torch.TorchPOCAOptimizer.POCAValueNetwork.update_normalization"
    ) as optimizer_update_normalization_mock, patch(
        "mlagents.trainers.torch_entities.networks.SimpleActor.update_normalization"
    ) as policy_update_normalization_mock:
        poca_trainer.advance()
        optimizer_update_normalization_mock.assert_called_once()
        policy_update_normalization_mock.assert_called_once()

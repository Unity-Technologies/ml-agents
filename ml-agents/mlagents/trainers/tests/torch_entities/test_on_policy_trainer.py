from pathlib import Path
from typing import Callable, Type

import numpy as np
import pytest

from mlagents.torch_utils import torch
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.buffer import RewardSignalUtil
from mlagents.trainers.poca.trainer import POCATrainer
from mlagents.trainers.ppo.trainer import PPOTrainer
from mlagents.trainers.settings import TrainerSettings
from mlagents.trainers.tests.dummy_config import ppo_dummy_config, poca_dummy_config
from mlagents.trainers.tests import mock_brain as mb
from mlagents.trainers.trainer.on_policy_trainer import OnPolicyTrainer


@pytest.mark.parametrize(
    "trainer_class,config_factory",
    [(PPOTrainer, ppo_dummy_config), (POCATrainer, poca_dummy_config)],
    ids=["ppo", "poca"],
)
def test_process_trajectory_preserves_value_estimates(
    trainer_class: Type[OnPolicyTrainer],
    config_factory: Callable[[], TrainerSettings],
    tmp_path: Path,
) -> None:
    trainer = trainer_class(
        "test_brain", 10, config_factory(), True, False, 0, str(tmp_path)
    )
    behavior_spec = mb.setup_test_behavior_specs(False, False)
    behavior_id = BehaviorIdentifiers.from_name_behavior_id("test_brain?team=0")
    policy = trainer.create_policy(behavior_id, behavior_spec)
    trainer.add_policy(behavior_id, policy)
    # Keep the old predictions distinct from the returns of the rewarded trajectory.
    with torch.no_grad():
        for parameter in trainer.optimizer.critic.parameters():
            parameter.zero_()
    trajectory = mb.make_fake_trajectory(
        length=2,
        observation_specs=behavior_spec.observation_specs,
        action_spec=behavior_spec.action_spec,
    )

    trainer._process_trajectory(trajectory)

    batch = trainer.update_buffer.make_mini_batch(0, 2)
    np.testing.assert_array_equal(
        batch[RewardSignalUtil.value_estimates_key("extrinsic")], [0.0, 0.0]
    )
    # Two rewards of 1, gamma=0.99 and lambda=0.95 with zero value estimates.
    np.testing.assert_allclose(
        batch[RewardSignalUtil.returns_key("extrinsic")], [1.9405, 1.0]
    )

import pytest
import os

import numpy as np

from mlagents_envs.logging_util import WARNING
from mlagents.trainers.ppo.optimizer_torch import TorchPPOOptimizer, PPOSettings
from mlagents.trainers.sac.optimizer_torch import TorchSACOptimizer, SACSettings
from mlagents.trainers.poca.optimizer_torch import TorchPOCAOptimizer, POCASettings
from mlagents.trainers.model_saver.torch_model_saver import TorchModelSaver
from mlagents.trainers.settings import (
    TrainerSettings,
    RewardSignalType,
    CuriositySettings,
    GAILSettings,
    RNDSettings,
)
from mlagents.trainers.tests.torch_entities.test_policy import create_policy_mock
from mlagents.trainers.tests.torch_entities.test_reward_providers.utils import (
    create_agent_buffer,
)


DEMO_PATH = (
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
    + "/test.demo"
)


@pytest.mark.parametrize(
    "optimizer",
    [
        (TorchPPOOptimizer, PPOSettings),
        (TorchSACOptimizer, SACSettings),
        (TorchPOCAOptimizer, POCASettings),
    ],
    ids=["ppo", "sac", "poca"],
)
def test_reward_provider_save(tmp_path, optimizer):
    OptimizerClass, HyperparametersClass = optimizer

    trainer_settings = TrainerSettings()
    trainer_settings.hyperparameters = HyperparametersClass()
    trainer_settings.reward_signals = {
        RewardSignalType.CURIOSITY: CuriositySettings(),
        RewardSignalType.GAIL: GAILSettings(demo_path=DEMO_PATH),
        RewardSignalType.RND: RNDSettings(),
    }
    policy = create_policy_mock(trainer_settings.network_settings, use_discrete=False)
    optimizer = OptimizerClass(policy, trainer_settings)

    # save at path 1
    path1 = os.path.join(tmp_path, "runid1")
    model_saver = TorchModelSaver(trainer_settings, path1)
    model_saver.register(policy)
    model_saver.register(optimizer)
    model_saver.initialize_or_load()
    policy.set_step(2000)
    model_saver.save_checkpoint("MockBrain", 2000)

    # create a new optimizer and policy
    optimizer2 = OptimizerClass(policy, trainer_settings)
    policy2 = create_policy_mock(trainer_settings.network_settings, use_discrete=False)

    # load weights
    model_saver2 = TorchModelSaver(trainer_settings, path1, load=True)
    model_saver2.register(policy2)
    model_saver2.register(optimizer2)
    model_saver2.initialize_or_load()  # This is to load the optimizers

    # assert the models have the same weights
    module_dict_1 = optimizer.get_modules()
    module_dict_2 = optimizer2.get_modules()
    assert "Module:GAIL" in module_dict_1
    assert "Module:GAIL" in module_dict_2
    assert "Module:Curiosity" in module_dict_1
    assert "Module:Curiosity" in module_dict_2
    assert "Module:RND-pred" in module_dict_1
    assert "Module:RND-pred" in module_dict_2
    assert "Module:RND-target" in module_dict_1
    assert "Module:RND-target" in module_dict_2
    for name, module1 in module_dict_1.items():
        assert name in module_dict_2
        module2 = module_dict_2[name]
        if hasattr(module1, "parameters"):
            for param1, param2 in zip(module1.parameters(), module2.parameters()):
                assert param1.data.ne(param2.data).sum() == 0

    # Run some rewards
    data = create_agent_buffer(policy.behavior_spec, 1)
    for reward_name in optimizer.reward_signals.keys():
        rp_1 = optimizer.reward_signals[reward_name]
        rp_2 = optimizer2.reward_signals[reward_name]
        assert np.array_equal(rp_1.evaluate(data), rp_2.evaluate(data))


@pytest.mark.parametrize(
    "optimizer",
    [
        (TorchPPOOptimizer, PPOSettings),
        (TorchSACOptimizer, SACSettings),
        (TorchPOCAOptimizer, POCASettings),
    ],
    ids=["ppo", "sac", "poca"],
)
def test_load_different_reward_provider(caplog, tmp_path, optimizer):
    OptimizerClass, HyperparametersClass = optimizer

    trainer_settings = TrainerSettings()
    trainer_settings.hyperparameters = HyperparametersClass()
    trainer_settings.reward_signals = {
        RewardSignalType.CURIOSITY: CuriositySettings(),
        RewardSignalType.RND: RNDSettings(),
    }

    policy = create_policy_mock(trainer_settings.network_settings, use_discrete=False)
    optimizer = OptimizerClass(policy, trainer_settings)

    # save at path 1
    path1 = os.path.join(tmp_path, "runid1")
    model_saver = TorchModelSaver(trainer_settings, path1)
    model_saver.register(policy)
    model_saver.register(optimizer)
    model_saver.initialize_or_load()
    assert len(optimizer.critic.value_heads.stream_names) == 2
    policy.set_step(2000)
    model_saver.save_checkpoint("MockBrain", 2000)

    trainer_settings2 = TrainerSettings()
    trainer_settings2.hyperparameters = HyperparametersClass()
    trainer_settings2.reward_signals = {
        RewardSignalType.GAIL: GAILSettings(demo_path=DEMO_PATH)
    }

    # create a new optimizer and policy
    policy2 = create_policy_mock(trainer_settings2.network_settings, use_discrete=False)
    optimizer2 = OptimizerClass(policy2, trainer_settings2)

    # load weights
    model_saver2 = TorchModelSaver(trainer_settings2, path1, load=True)
    model_saver2.register(policy2)
    model_saver2.register(optimizer2)
    assert len(optimizer2.critic.value_heads.stream_names) == 1
    model_saver2.initialize_or_load()  # This is to load the optimizers
    messages = [rec.message for rec in caplog.records if rec.levelno == WARNING]
    assert len(messages) > 0

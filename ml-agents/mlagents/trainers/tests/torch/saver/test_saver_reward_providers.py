import pytest
import os

from mlagents.trainers.ppo.optimizer_torch import TorchPPOOptimizer
from mlagents.trainers.sac.optimizer_torch import TorchSACOptimizer
from mlagents.trainers.saver.torch_saver import TorchSaver
from mlagents.trainers.settings import (
    TrainerSettings,
    RewardSignalType,
    CuriositySettings,
    GAILSettings,
    PPOSettings,
    SACSettings,
)
from mlagents.trainers.tests.torch.test_policy import create_policy_mock

DEMO_PATH = (
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
    + "/test.demo"
)


@pytest.mark.parametrize(
    "optimizer",
    [(TorchPPOOptimizer, PPOSettings), (TorchSACOptimizer, SACSettings)],
    ids=["ppo", "sac"],
)
def test_reward_provider_save(tmp_path, optimizer):
    OptimizerClass, HyperparametersClass = optimizer

    trainer_settings = TrainerSettings()
    trainer_settings.hyperparameters = HyperparametersClass()
    trainer_settings.reward_signals = {
        RewardSignalType.CURIOSITY: CuriositySettings(),
        RewardSignalType.GAIL: GAILSettings(demo_path=DEMO_PATH),
    }
    policy = create_policy_mock(trainer_settings, use_discrete=False)
    optimizer = OptimizerClass(policy, trainer_settings)

    # save at path 1
    path1 = os.path.join(tmp_path, "runid1")
    saver = TorchSaver(trainer_settings, path1)
    saver.register(policy)
    saver.register(optimizer)
    saver.initialize_or_load(policy)
    policy.set_step(2000)
    saver.save_checkpoint("MockBrain", 2000)

    # create a new optimizer and policy
    optimizer2 = OptimizerClass(policy, trainer_settings)
    policy2 = create_policy_mock(trainer_settings, use_discrete=False)

    # load weights
    saver2 = TorchSaver(trainer_settings, path1, load=True)
    saver2.register(policy2)
    saver2.register(optimizer2)
    saver2.initialize_or_load(policy2)
    saver2.initialize_or_load()  # This is to load the optimizers

    # assert the models have the same weights
    module_dict_1 = optimizer.get_modules()
    module_dict_2 = optimizer2.get_modules()
    assert "Module:GAIL" in module_dict_1
    assert "Module:GAIL" in module_dict_2
    for name, module1 in module_dict_1.items():
        assert name in module_dict_2
        module2 = module_dict_2[name]
        if hasattr(module1, "parameters"):
            for param1, param2 in zip(module1.parameters(), module2.parameters()):
                print(param1.data)
                print(param2.data)
                assert param1.data.ne(param2.data).sum() == 0

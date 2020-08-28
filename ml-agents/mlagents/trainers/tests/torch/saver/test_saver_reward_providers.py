import pytest
import os

from mlagents.trainers.ppo.optimizer_torch import TorchPPOOptimizer
from mlagents.trainers.sac.optimizer_torch import TorchSACOptimizer
from mlagents.trainers.model_saver.torch_model_saver import TorchModelSaver
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
    model_saver = TorchModelSaver(trainer_settings, path1)
    model_saver.register(policy)
    model_saver.register(optimizer)
    model_saver.initialize_or_load()
    policy.set_step(2000)
    model_saver.save_checkpoint("MockBrain", 2000)

    # create a new optimizer and policy
    optimizer2 = OptimizerClass(policy, trainer_settings)
    policy2 = create_policy_mock(trainer_settings, use_discrete=False)

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
    for name, module1 in module_dict_1.items():
        assert name in module_dict_2
        module2 = module_dict_2[name]
        if hasattr(module1, "parameters"):
            for param1, param2 in zip(module1.parameters(), module2.parameters()):
                assert param1.data.ne(param2.data).sum() == 0

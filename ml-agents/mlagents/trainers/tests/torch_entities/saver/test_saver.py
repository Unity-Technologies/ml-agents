from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
import pytest
from unittest import mock
import os

import numpy as np
from mlagents.torch_utils import torch, default_device
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.ppo.optimizer_torch import TorchPPOOptimizer, PPOSettings
from mlagents.trainers.sac.optimizer_torch import TorchSACOptimizer, SACSettings
from mlagents.trainers.poca.optimizer_torch import TorchPOCAOptimizer, POCASettings
from mlagents.trainers.model_saver.torch_model_saver import (
    TorchModelSaver,
    DEFAULT_CHECKPOINT_NAME,
)
from mlagents.trainers.settings import (
    TrainerSettings,
    NetworkSettings,
    EncoderType,
)
from mlagents.trainers.tests import mock_brain as mb
from mlagents.trainers.tests.torch_entities.test_policy import create_policy_mock
from mlagents.trainers.torch_entities.utils import ModelUtils


def test_register(tmp_path):
    trainer_params = TrainerSettings()
    model_saver = TorchModelSaver(trainer_params, tmp_path)

    opt = mock.Mock(spec=TorchPPOOptimizer)
    opt.get_modules = mock.Mock(return_value={})
    model_saver.register(opt)
    assert model_saver.policy is None

    trainer_params = TrainerSettings()
    policy = create_policy_mock(trainer_params.network_settings)
    opt.get_modules = mock.Mock(return_value={})
    model_saver.register(policy)
    assert model_saver.policy is not None


def test_load_save_policy(tmp_path):
    path1 = os.path.join(tmp_path, "runid1")
    path2 = os.path.join(tmp_path, "runid2")
    trainer_params = TrainerSettings()
    policy = create_policy_mock(trainer_params.network_settings)
    model_saver = TorchModelSaver(trainer_params, path1)
    model_saver.register(policy)
    model_saver.initialize_or_load(policy)
    policy.set_step(2000)

    mock_brain_name = "MockBrain"
    model_saver.save_checkpoint(mock_brain_name, 2000)
    assert len(os.listdir(tmp_path)) > 0

    # Try load from this path
    model_saver2 = TorchModelSaver(trainer_params, path1, load=True)
    policy2 = create_policy_mock(trainer_params.network_settings)
    model_saver2.register(policy2)
    model_saver2.initialize_or_load(policy2)
    _compare_two_policies(policy, policy2)
    assert policy2.get_current_step() == 2000

    # Try initialize from path 1
    trainer_params.init_path = os.path.join(path1, DEFAULT_CHECKPOINT_NAME)
    model_saver3 = TorchModelSaver(trainer_params, path2)
    policy3 = create_policy_mock(trainer_params.network_settings)
    model_saver3.register(policy3)
    model_saver3.initialize_or_load(policy3)
    _compare_two_policies(policy2, policy3)
    # Assert that the steps are 0.
    assert policy3.get_current_step() == 0


@pytest.mark.parametrize("vis_encode_type", ["resnet", "nature_cnn", "match3"])
def test_load_policy_different_hidden_units(tmp_path, vis_encode_type):
    path1 = os.path.join(tmp_path, "runid1")
    trainer_params = TrainerSettings()
    trainer_params.network_settings = NetworkSettings(
        hidden_units=12, vis_encode_type=EncoderType(vis_encode_type)
    )
    policy = create_policy_mock(trainer_params.network_settings, use_visual=True)
    conv_params = [mod for mod in policy.actor.parameters() if len(mod.shape) > 2]

    model_saver = TorchModelSaver(trainer_params, path1)
    model_saver.register(policy)
    model_saver.initialize_or_load(policy)
    policy.set_step(2000)

    mock_brain_name = "MockBrain"
    model_saver.save_checkpoint(mock_brain_name, 2000)

    # Try load from this path
    trainer_params2 = TrainerSettings()
    trainer_params2.network_settings = NetworkSettings(
        hidden_units=10, vis_encode_type=EncoderType(vis_encode_type)
    )
    model_saver2 = TorchModelSaver(trainer_params2, path1, load=True)
    policy2 = create_policy_mock(trainer_params2.network_settings, use_visual=True)
    conv_params2 = [mod for mod in policy2.actor.parameters() if len(mod.shape) > 2]
    # asserts convolutions have different parameters before load
    for conv1, conv2 in zip(conv_params, conv_params2):
        assert not torch.equal(conv1, conv2)
    # asserts layers still have different dimensions
    for mod1, mod2 in zip(policy.actor.parameters(), policy2.actor.parameters()):
        if mod1.shape[0] == 12:
            assert mod2.shape[0] == 10
    model_saver2.register(policy2)
    model_saver2.initialize_or_load(policy2)
    # asserts convolutions have same parameters after load
    for conv1, conv2 in zip(conv_params, conv_params2):
        assert torch.equal(conv1, conv2)
    # asserts layers still have different dimensions
    for mod1, mod2 in zip(policy.actor.parameters(), policy2.actor.parameters()):
        if mod1.shape[0] == 12:
            assert mod2.shape[0] == 10


@pytest.mark.parametrize(
    "optimizer",
    [
        (TorchPPOOptimizer, PPOSettings),
        (TorchSACOptimizer, SACSettings),
        (TorchPOCAOptimizer, POCASettings),
    ],
    ids=["ppo", "sac", "poca"],
)
def test_load_save_optimizer(tmp_path, optimizer):
    OptimizerClass, HyperparametersClass = optimizer

    trainer_settings = TrainerSettings()
    trainer_settings.hyperparameters = HyperparametersClass()
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
    policy2 = create_policy_mock(trainer_settings.network_settings, use_discrete=False)
    optimizer2 = OptimizerClass(policy2, trainer_settings)

    # load weights
    model_saver2 = TorchModelSaver(trainer_settings, path1, load=True)
    model_saver2.register(policy2)
    model_saver2.register(optimizer2)
    model_saver2.initialize_or_load()  # This is to load the optimizers

    # Compare the two optimizers
    _compare_two_optimizers(optimizer, optimizer2)


# TorchPolicy.evalute() returns log_probs instead of all_log_probs like tf does.
# resulting in indeterministic results for testing.
# So here use sample_actions instead.
def _compare_two_policies(policy1: TorchPolicy, policy2: TorchPolicy) -> None:
    """
    Make sure two policies have the same output for the same input.
    """
    policy1.actor = policy1.actor.to(default_device())
    policy2.actor = policy2.actor.to(default_device())

    decision_step, _ = mb.create_steps_from_behavior_spec(
        policy1.behavior_spec, num_agents=1
    )
    np_obs = decision_step.obs
    masks = policy1._extract_masks(decision_step)
    memories = torch.as_tensor(
        policy1.retrieve_memories(list(decision_step.agent_id))
    ).unsqueeze(0)
    tensor_obs = [ModelUtils.list_to_tensor(obs) for obs in np_obs]

    with torch.no_grad():
        _, stat_dict1, _ = policy1.actor.get_action_and_stats(
            tensor_obs, masks=masks, memories=memories
        )
        _, stat_dict2, _ = policy2.actor.get_action_and_stats(
            tensor_obs, masks=masks, memories=memories
        )
        log_probs1 = stat_dict1["log_probs"]
        log_probs2 = stat_dict2["log_probs"]
    np.testing.assert_array_equal(
        ModelUtils.to_numpy(log_probs1.all_discrete_tensor),
        ModelUtils.to_numpy(log_probs2.all_discrete_tensor),
    )


def _compare_two_optimizers(opt1: TorchOptimizer, opt2: TorchOptimizer) -> None:
    trajectory = mb.make_fake_trajectory(
        length=10,
        observation_specs=opt1.policy.behavior_spec.observation_specs,
        action_spec=opt1.policy.behavior_spec.action_spec,
        max_step_complete=True,
    )
    with torch.no_grad():
        _, opt1_val_out, _ = opt1.get_trajectory_value_estimates(
            trajectory.to_agentbuffer(), trajectory.next_obs, done=False
        )
        _, opt2_val_out, _ = opt2.get_trajectory_value_estimates(
            trajectory.to_agentbuffer(), trajectory.next_obs, done=False
        )

    for opt1_val, opt2_val in zip(opt1_val_out.values(), opt2_val_out.values()):
        np.testing.assert_array_equal(opt1_val, opt2_val)


@pytest.mark.parametrize("discrete", [True, False], ids=["discrete", "continuous"])
@pytest.mark.parametrize("visual", [True, False], ids=["visual", "vector"])
@pytest.mark.parametrize("rnn", [True, False], ids=["rnn", "no_rnn"])
def test_checkpoint_conversion(tmpdir, rnn, visual, discrete):
    dummy_config = TrainerSettings()
    model_path = os.path.join(tmpdir, "Mock_Brain")
    policy = create_policy_mock(
        dummy_config.network_settings,
        use_rnn=rnn,
        use_discrete=discrete,
        use_visual=visual,
    )
    trainer_params = TrainerSettings()
    model_saver = TorchModelSaver(trainer_params, model_path)
    model_saver.register(policy)
    model_saver.save_checkpoint("Mock_Brain", 100)
    assert os.path.isfile(model_path + "/Mock_Brain-100.onnx")

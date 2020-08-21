import pytest
from unittest import mock
import os

import numpy as np
import torch
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.ppo.optimizer_torch import TorchPPOOptimizer
from mlagents.trainers.saver.torch_saver import TorchSaver
from mlagents.trainers.settings import TrainerSettings
from mlagents.trainers.tests import mock_brain as mb
from mlagents.trainers.tests.torch.test_policy import create_policy_mock


def test_register(tmp_path):
    trainer_params = TrainerSettings()
    saver = TorchSaver(trainer_params, tmp_path)

    opt = mock.Mock(spec=TorchPPOOptimizer)
    opt.get_modules = mock.Mock(return_value={})
    saver.register(opt)
    assert saver.policy is None

    trainer_params = TrainerSettings()
    policy = create_policy_mock(trainer_params)
    opt.get_modules = mock.Mock(return_value={})
    saver.register(policy)
    assert saver.policy is not None


def test_load_save(tmp_path):
    path1 = os.path.join(tmp_path, "runid1")
    path2 = os.path.join(tmp_path, "runid2")
    trainer_params = TrainerSettings()
    policy = create_policy_mock(trainer_params)
    saver = TorchSaver(trainer_params, path1)
    saver.register(policy)
    saver.initialize_or_load(policy)
    policy.set_step(2000)

    mock_brain_name = "MockBrain"
    saver.save_checkpoint(mock_brain_name, 2000)
    assert len(os.listdir(tmp_path)) > 0

    # Try load from this path
    saver2 = TorchSaver(trainer_params, path1, load=True)
    policy2 = create_policy_mock(trainer_params)
    saver2.register(policy2)
    saver2.initialize_or_load(policy2)
    _compare_two_policies(policy, policy2)
    assert policy2.get_current_step() == 2000

    # Try initialize from path 1
    trainer_params.init_path = path1
    saver3 = TorchSaver(trainer_params, path2)
    policy3 = create_policy_mock(trainer_params)
    saver3.register(policy3)
    saver3.initialize_or_load(policy3)
    _compare_two_policies(policy2, policy3)
    # Assert that the steps are 0.
    assert policy3.get_current_step() == 0


# TorchPolicy.evalute() returns log_probs instead of all_log_probs like tf does.
# resulting in indeterministic results for testing.
# So here use sample_actions instead.
def _compare_two_policies(policy1: TorchPolicy, policy2: TorchPolicy) -> None:
    """
    Make sure two policies have the same output for the same input.
    """
    decision_step, _ = mb.create_steps_from_behavior_spec(
        policy1.behavior_spec, num_agents=1
    )
    vec_vis_obs, masks = policy1._split_decision_step(decision_step)
    vec_obs = [torch.as_tensor(vec_vis_obs.vector_observations)]
    vis_obs = [torch.as_tensor(vis_ob) for vis_ob in vec_vis_obs.visual_observations]
    memories = torch.as_tensor(
        policy1.retrieve_memories(list(decision_step.agent_id))
    ).unsqueeze(0)

    with torch.no_grad():
        _, log_probs1, _, _, _ = policy1.sample_actions(
            vec_obs, vis_obs, masks=masks, memories=memories, all_log_probs=True
        )
        _, log_probs2, _, _, _ = policy2.sample_actions(
            vec_obs, vis_obs, masks=masks, memories=memories, all_log_probs=True
        )

    np.testing.assert_array_equal(log_probs1, log_probs2)


@pytest.mark.parametrize("discrete", [True, False], ids=["discrete", "continuous"])
@pytest.mark.parametrize("visual", [True, False], ids=["visual", "vector"])
@pytest.mark.parametrize("rnn", [True, False], ids=["rnn", "no_rnn"])
def test_checkpoint_conversion(tmpdir, rnn, visual, discrete):
    dummy_config = TrainerSettings()
    model_path = os.path.join(tmpdir, "Mock_Brain")
    policy = create_policy_mock(
        dummy_config, use_rnn=rnn, use_discrete=discrete, use_visual=visual
    )
    trainer_params = TrainerSettings()
    saver = TorchSaver(trainer_params, model_path)
    saver.register(policy)
    saver.save_checkpoint("Mock_Brain", 100)
    assert os.path.isfile(model_path + "/Mock_Brain-100.onnx")

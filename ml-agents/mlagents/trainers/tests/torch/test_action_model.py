import pytest

from mlagents.torch_utils import torch
from mlagents.trainers.torch.action_model import ActionModel, DistInstances
from mlagents.trainers.torch.agent_action import AgentAction
from mlagents.trainers.torch.distributions import (
    GaussianDistInstance,
    CategoricalDistInstance,
)

from mlagents_envs.base_env import ActionSpec


def create_action_model(inp_size, act_size, deterministic=False):
    mask = torch.ones([1, act_size ** 2])
    action_spec = ActionSpec(act_size, tuple(act_size for _ in range(act_size)))
    action_model = ActionModel(inp_size, action_spec, deterministic=deterministic)
    return action_model, mask


def test_get_dists():
    inp_size = 4
    act_size = 2
    action_model, masks = create_action_model(inp_size, act_size)
    sample_inp = torch.ones((1, inp_size))
    dists = action_model._get_dists(sample_inp, masks=masks)
    assert isinstance(dists.continuous, GaussianDistInstance)
    assert len(dists.discrete) == 2
    for _dist in dists.discrete:
        assert isinstance(_dist, CategoricalDistInstance)


def test_sample_action():
    inp_size = 4
    act_size = 2
    action_model, masks = create_action_model(inp_size, act_size)
    sample_inp = torch.ones((1, inp_size))
    dists = action_model._get_dists(sample_inp, masks=masks)
    agent_action = action_model._sample_action(dists)
    assert agent_action.continuous_tensor.shape == (1, 2)
    assert len(agent_action.discrete_list) == 2
    for _disc in agent_action.discrete_list:
        assert _disc.shape == (1, 1)


def test_deterministic_sample_action():
    inp_size = 4
    act_size = 8
    action_model, masks = create_action_model(inp_size, act_size, deterministic=True)
    sample_inp = torch.ones((1, inp_size))
    dists = action_model._get_dists(sample_inp, masks=masks)
    agent_action1 = action_model._sample_action(dists)
    agent_action2 = action_model._sample_action(dists)
    agent_action3 = action_model._sample_action(dists)

    assert torch.equal(agent_action1.continuous_tensor, agent_action2.continuous_tensor)
    assert torch.equal(agent_action1.continuous_tensor, agent_action3.continuous_tensor)
    assert torch.equal(agent_action1.discrete_tensor, agent_action2.discrete_tensor)
    assert torch.equal(agent_action1.discrete_tensor, agent_action3.discrete_tensor)

    action_model, masks = create_action_model(inp_size, act_size, deterministic=False)
    sample_inp = torch.ones((1, inp_size))
    dists = action_model._get_dists(sample_inp, masks=masks)
    agent_action1 = action_model._sample_action(dists)
    agent_action2 = action_model._sample_action(dists)
    agent_action3 = action_model._sample_action(dists)

    assert not torch.equal(
        agent_action1.continuous_tensor, agent_action2.continuous_tensor
    )

    assert not torch.equal(
        agent_action1.continuous_tensor, agent_action3.continuous_tensor
    )

    chance_counter = 0
    if not torch.equal(agent_action1.discrete_tensor, agent_action2.discrete_tensor):
        chance_counter += 1
    if not torch.equal(agent_action1.discrete_tensor, agent_action3.discrete_tensor):
        chance_counter += 1
    if not torch.equal(agent_action2.discrete_tensor, agent_action3.discrete_tensor):
        chance_counter += 1

    assert chance_counter > 1


def test_get_probs_and_entropy():
    inp_size = 4
    act_size = 2
    action_model, masks = create_action_model(inp_size, act_size)

    _continuous_dist = GaussianDistInstance(torch.zeros((1, 2)), torch.ones((1, 2)))
    act_size = 2
    test_prob = torch.tensor([[1.0 - 0.1 * (act_size - 1)] + [0.1] * (act_size - 1)])
    _discrete_dist_list = [
        CategoricalDistInstance(test_prob),
        CategoricalDistInstance(test_prob),
    ]
    dist_tuple = DistInstances(_continuous_dist, _discrete_dist_list)

    agent_action = AgentAction(
        torch.zeros((1, 2)), [torch.tensor([0]), torch.tensor([1])]
    )

    log_probs, entropies = action_model._get_probs_and_entropy(agent_action, dist_tuple)

    assert log_probs.continuous_tensor.shape == (1, 2)
    assert len(log_probs.discrete_list) == 2
    for _disc in log_probs.discrete_list:
        assert _disc.shape == (1,)
    assert len(log_probs.all_discrete_list) == 2
    for _disc in log_probs.all_discrete_list:
        assert _disc.shape == (1, 2)

    for clp in log_probs.continuous_tensor[0].tolist():
        # Log prob of standard normal at 0
        assert clp == pytest.approx(-0.919, abs=0.01)

    assert log_probs.discrete_list[0] > log_probs.discrete_list[1]

    for ent, val in zip(entropies[0].tolist(), [1.4189, 0.6191, 0.6191]):
        assert ent == pytest.approx(val, abs=0.01)


def test_get_onnx_deterministic_tensors():
    inp_size = 4
    act_size = 2
    action_model, masks = create_action_model(inp_size, act_size)
    sample_inp = torch.ones((1, inp_size))
    out_tensors = action_model.get_action_out(sample_inp, masks=masks)
    (
        continuous_out,
        discrete_out,
        action_out_deprecated,
        deterministic_continuous_out,
        deterministic_discrete_out,
    ) = out_tensors
    assert continuous_out.shape == (1, 2)
    assert discrete_out.shape == (1, 2)
    assert deterministic_discrete_out.shape == (1, 2)
    assert deterministic_continuous_out.shape == (1, 2)

    # Second sampling from same distribution
    out_tensors2 = action_model.get_action_out(sample_inp, masks=masks)
    (
        continuous_out_2,
        discrete_out_2,
        action_out_2_deprecated,
        deterministic_continuous_out_2,
        deterministic_discrete_out_2,
    ) = out_tensors2
    assert ~torch.all(torch.eq(continuous_out, continuous_out_2))
    assert torch.all(
        torch.eq(deterministic_continuous_out, deterministic_continuous_out_2)
    )

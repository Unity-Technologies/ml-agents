import numpy as np
from mlagents.torch_utils import torch

from mlagents.trainers.buffer import AgentBuffer, BufferKey
from mlagents.trainers.torch_entities.agent_action import AgentAction


def test_agent_action_group_from_buffer():
    buff = AgentBuffer()
    # Create some actions
    for _ in range(3):
        buff[BufferKey.GROUP_CONTINUOUS_ACTION].append(
            3 * [np.ones((5,), dtype=np.float32)]
        )
        buff[BufferKey.GROUP_DISCRETE_ACTION].append(
            3 * [np.ones((4,), dtype=np.float32)]
        )
    # Some agents have died
    for _ in range(2):
        buff[BufferKey.GROUP_CONTINUOUS_ACTION].append(
            1 * [np.ones((5,), dtype=np.float32)]
        )
        buff[BufferKey.GROUP_DISCRETE_ACTION].append(
            1 * [np.ones((4,), dtype=np.float32)]
        )

    # Get the group actions, which will be a List of Lists of AgentAction, where each element is the same
    # length as the AgentBuffer but contains only one agent's obs. Dead agents are padded by
    # NaNs.
    gact = AgentAction.group_from_buffer(buff)
    # Agent 0 is full
    agent_0_act = gact[0]
    assert agent_0_act.continuous_tensor.shape == (buff.num_experiences, 5)
    assert agent_0_act.discrete_tensor.shape == (buff.num_experiences, 4)

    agent_1_act = gact[1]
    assert agent_1_act.continuous_tensor.shape == (buff.num_experiences, 5)
    assert agent_1_act.discrete_tensor.shape == (buff.num_experiences, 4)
    assert (agent_1_act.continuous_tensor[0:3] > 0).all()
    assert (agent_1_act.continuous_tensor[3:] == 0).all()
    assert (agent_1_act.discrete_tensor[0:3] > 0).all()
    assert (agent_1_act.discrete_tensor[3:] == 0).all()


def test_slice():
    # Both continuous and discrete
    aa = AgentAction(
        torch.tensor([[1.0], [1.0], [1.0]]),
        [torch.tensor([2, 1, 0]), torch.tensor([1, 2, 0])],
    )
    saa = aa.slice(0, 2)
    assert saa.continuous_tensor.shape == (2, 1)
    assert saa.discrete_tensor.shape == (2, 2)


def test_to_flat():
    # Both continuous and discrete
    aa = AgentAction(
        torch.tensor([[1.0, 1.0, 1.0]]), [torch.tensor([2]), torch.tensor([1])]
    )
    flattened_actions = aa.to_flat([3, 3])
    assert torch.eq(
        flattened_actions, torch.tensor([[1, 1, 1, 0, 0, 1, 0, 1, 0]])
    ).all()

    # Just continuous
    aa = AgentAction(torch.tensor([[1.0, 1.0, 1.0]]), None)
    flattened_actions = aa.to_flat([])
    assert torch.eq(flattened_actions, torch.tensor([1, 1, 1])).all()

    # Just discrete
    aa = AgentAction(torch.tensor([]), [torch.tensor([2]), torch.tensor([1])])
    flattened_actions = aa.to_flat([3, 3])
    assert torch.eq(flattened_actions, torch.tensor([0, 0, 1, 0, 1, 0])).all()

from typing import List, Optional, Tuple
import numpy as np

from mlagents.trainers.buffer import AgentBuffer, AgentBufferKey
from mlagents.trainers.torch_entities.action_log_probs import LogProbsTuple
from mlagents.trainers.trajectory import AgentStatus, Trajectory, AgentExperience
from mlagents_envs.base_env import (
    DecisionSteps,
    TerminalSteps,
    ObservationSpec,
    BehaviorSpec,
    ActionSpec,
    ActionTuple,
)
from mlagents.trainers.tests.dummy_config import create_observation_specs_with_shapes


def create_mock_steps(
    num_agents: int,
    observation_specs: List[ObservationSpec],
    action_spec: ActionSpec,
    done: bool = False,
    grouped: bool = False,
    agent_ids: Optional[List[int]] = None,
) -> Tuple[DecisionSteps, TerminalSteps]:
    """
    Creates a mock Tuple[DecisionSteps, TerminalSteps] with observations.
    Imitates constant vector/visual observations, rewards, dones, and agents.

    :int num_agents: Number of "agents" to imitate.
    :List observation_specs: A List of the observation specs in your steps
    :int action_spec: ActionSpec for the agent
    :bool done: Whether all the agents in the batch are done
    """
    obs_list = []
    for obs_spec in observation_specs:
        obs_list.append(np.ones((num_agents,) + obs_spec.shape, dtype=np.float32))
    action_mask = None
    if action_spec.is_discrete():
        action_mask = [
            np.array(num_agents * [action_size * [False]])
            for action_size in action_spec.discrete_branches  # type: ignore
        ]  # type: ignore

    reward = np.array(num_agents * [1.0], dtype=np.float32)
    interrupted = np.array(num_agents * [False], dtype=np.bool)
    if agent_ids is not None:
        agent_id = np.array(agent_ids, dtype=np.int32)
    else:
        agent_id = np.arange(num_agents, dtype=np.int32)
    _gid = 1 if grouped else 0
    group_id = np.array(num_agents * [_gid], dtype=np.int32)
    group_reward = np.array(num_agents * [0.0], dtype=np.float32)
    behavior_spec = BehaviorSpec(observation_specs, action_spec)
    if done:
        return (
            DecisionSteps.empty(behavior_spec),
            TerminalSteps(
                obs_list, reward, interrupted, agent_id, group_id, group_reward
            ),
        )
    else:
        return (
            DecisionSteps(
                obs_list, reward, agent_id, action_mask, group_id, group_reward
            ),
            TerminalSteps.empty(behavior_spec),
        )


def create_steps_from_behavior_spec(
    behavior_spec: BehaviorSpec, num_agents: int = 1
) -> Tuple[DecisionSteps, TerminalSteps]:
    return create_mock_steps(
        num_agents=num_agents,
        observation_specs=behavior_spec.observation_specs,
        action_spec=behavior_spec.action_spec,
    )


def make_fake_trajectory(
    length: int,
    observation_specs: List[ObservationSpec],
    action_spec: ActionSpec,
    max_step_complete: bool = False,
    memory_size: int = 10,
    num_other_agents_in_group: int = 0,
    group_reward: float = 0.0,
    is_terminal: bool = True,
    team_id: int = 0,
) -> Trajectory:
    """
    Makes a fake trajectory of length length. If max_step_complete,
    the trajectory is terminated by a max step rather than a done.
    """
    steps_list = []

    action_size = action_spec.discrete_size + action_spec.continuous_size
    for _i in range(length - 1):
        obs = []
        for obs_spec in observation_specs:
            obs.append(np.ones(obs_spec.shape, dtype=np.float32))
        reward = 1.0
        done = False
        action = ActionTuple(
            continuous=np.zeros(action_spec.continuous_size, dtype=np.float32),
            discrete=np.zeros(action_spec.discrete_size, dtype=np.int32),
        )
        action_probs = LogProbsTuple(
            continuous=np.ones(action_spec.continuous_size, dtype=np.float32),
            discrete=np.ones(action_spec.discrete_size, dtype=np.float32),
        )
        action_mask = (
            [
                [False for _ in range(branch)]
                for branch in action_spec.discrete_branches
            ]  # type: ignore
            if action_spec.is_discrete()
            else None
        )
        if action_spec.is_discrete():
            prev_action = np.ones(action_size, dtype=np.int32)
        else:
            prev_action = np.ones(action_size, dtype=np.float32)

        max_step = False
        memory = np.ones(memory_size, dtype=np.float32)
        agent_id = "test_agent"
        behavior_id = "test_brain?team=" + str(team_id)
        group_status = []
        for _ in range(num_other_agents_in_group):
            group_status.append(AgentStatus(obs, reward, action, done))
        experience = AgentExperience(
            obs=obs,
            reward=reward,
            done=done,
            action=action,
            action_probs=action_probs,
            action_mask=action_mask,
            prev_action=prev_action,
            interrupted=max_step,
            memory=memory,
            group_status=group_status,
            group_reward=group_reward,
        )
        steps_list.append(experience)
    obs = []
    for obs_spec in observation_specs:
        obs.append(np.ones(obs_spec.shape, dtype=np.float32))
    last_group_status = []
    for _ in range(num_other_agents_in_group):
        last_group_status.append(
            AgentStatus(obs, reward, action, not max_step_complete and is_terminal)
        )
    last_experience = AgentExperience(
        obs=obs,
        reward=reward,
        done=not max_step_complete and is_terminal,
        action=action,
        action_probs=action_probs,
        action_mask=action_mask,
        prev_action=prev_action,
        interrupted=max_step_complete,
        memory=memory,
        group_status=last_group_status,
        group_reward=group_reward,
    )
    steps_list.append(last_experience)
    return Trajectory(
        steps=steps_list,
        agent_id=agent_id,
        behavior_id=behavior_id,
        next_obs=obs,
        next_group_obs=[obs] * num_other_agents_in_group,
    )


def copy_buffer_fields(
    buffer: AgentBuffer, src_key: AgentBufferKey, dst_keys: List[AgentBufferKey]
) -> None:
    for dst_key in dst_keys:
        buffer[dst_key] = buffer[src_key]


def simulate_rollout(
    length: int,
    behavior_spec: BehaviorSpec,
    memory_size: int = 10,
    exclude_key_list: List[str] = None,
    num_other_agents_in_group: int = 0,
) -> AgentBuffer:
    trajectory = make_fake_trajectory(
        length,
        behavior_spec.observation_specs,
        action_spec=behavior_spec.action_spec,
        memory_size=memory_size,
        num_other_agents_in_group=num_other_agents_in_group,
    )
    buffer = trajectory.to_agentbuffer()
    # If a key_list was given, remove those keys
    if exclude_key_list:
        for key in exclude_key_list:
            if key in buffer:
                buffer.pop(key)
    return buffer


def setup_test_behavior_specs(
    use_discrete=True, use_visual=False, vector_action_space=2, vector_obs_space=8
):
    if use_discrete:
        action_spec = ActionSpec.create_discrete(tuple(vector_action_space))
    else:
        action_spec = ActionSpec.create_continuous(vector_action_space)
    observation_shapes = [(84, 84, 3)] * int(use_visual) + [(vector_obs_space,)]
    obs_spec = create_observation_specs_with_shapes(observation_shapes)
    behavior_spec = BehaviorSpec(obs_spec, action_spec)
    return behavior_spec


def create_mock_3dball_behavior_specs():
    return setup_test_behavior_specs(
        False, False, vector_action_space=2, vector_obs_space=8
    )


def create_mock_pushblock_behavior_specs():
    return setup_test_behavior_specs(
        True, False, vector_action_space=7, vector_obs_space=70
    )


def create_mock_banana_behavior_specs():
    return setup_test_behavior_specs(
        True, True, vector_action_space=[3, 3, 3, 2], vector_obs_space=0
    )

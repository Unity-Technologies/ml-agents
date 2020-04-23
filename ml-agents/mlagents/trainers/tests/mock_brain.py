from unittest import mock
from typing import List, Tuple
import numpy as np

from mlagents.trainers.brain import CameraResolution, BrainParameters
from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.trajectory import Trajectory, AgentExperience
from mlagents_envs.base_env import (
    DecisionSteps,
    TerminalSteps,
    BehaviorSpec,
    ActionType,
)


def create_mock_brainparams(
    number_visual_observations=0,
    vector_action_space_type="continuous",
    vector_observation_space_size=3,
    vector_action_space_size=None,
):
    """
    Creates a mock BrainParameters object with parameters.
    """
    # Avoid using mutable object as default param
    if vector_action_space_size is None:
        vector_action_space_size = [2]
    mock_brain = mock.Mock()
    mock_brain.return_value.number_visual_observations = number_visual_observations
    mock_brain.return_value.vector_action_space_type = vector_action_space_type
    mock_brain.return_value.vector_observation_space_size = (
        vector_observation_space_size
    )
    camrez = CameraResolution(height=84, width=84, num_channels=3)
    mock_brain.return_value.camera_resolutions = [camrez] * number_visual_observations
    mock_brain.return_value.vector_action_space_size = vector_action_space_size
    mock_brain.return_value.brain_name = "MockBrain"
    return mock_brain()


def create_mock_steps(
    num_agents: int = 1,
    num_vector_observations: int = 0,
    num_vis_observations: int = 0,
    action_shape: List[int] = None,
    discrete: bool = False,
    done: bool = False,
) -> Tuple[DecisionSteps, TerminalSteps]:
    """
    Creates a mock Tuple[DecisionSteps, TerminalSteps] with observations.
    Imitates constant vector/visual observations, rewards, dones, and agents.

    :int num_agents: Number of "agents" to imitate.
    :int num_vector_observations: Number of "observations" in your observation space
    :int num_vis_observations: Number of "observations" in your observation space
    :int num_vector_acts: Number of actions in your action space
    :bool discrete: Whether or not action space is discrete
    :bool done: Whether all the agents in the batch are done
    """
    if action_shape is None:
        action_shape = [2]

    obs_list = []
    for _ in range(num_vis_observations):
        obs_list.append(np.ones((num_agents, 84, 84, 3), dtype=np.float32))
    if num_vector_observations > 1:
        obs_list.append(
            np.array(num_agents * [num_vector_observations * [1]], dtype=np.float32)
        )
    action_mask = None
    if discrete:
        action_mask = [
            np.array(num_agents * [action_size * [False]])
            for action_size in action_shape
        ]

    reward = np.array(num_agents * [1.0], dtype=np.float32)
    max_step = np.array(num_agents * [False], dtype=np.bool)
    agent_id = np.arange(num_agents, dtype=np.int32)
    behavior_spec = BehaviorSpec(
        [(84, 84, 3)] * num_vis_observations + [(num_vector_observations, 0, 0)],
        ActionType.DISCRETE if discrete else ActionType.CONTINUOUS,
        action_shape if discrete else action_shape[0],
    )
    if done:
        return (
            DecisionSteps.empty(behavior_spec),
            TerminalSteps(obs_list, reward, max_step, agent_id),
        )
    else:
        return (
            DecisionSteps(obs_list, reward, agent_id, action_mask),
            TerminalSteps.empty(behavior_spec),
        )


def create_steps_from_brainparams(
    brain_params: BrainParameters, num_agents: int = 1
) -> Tuple[DecisionSteps, TerminalSteps]:
    return create_mock_steps(
        num_agents=num_agents,
        num_vector_observations=brain_params.vector_observation_space_size,
        num_vis_observations=brain_params.number_visual_observations,
        action_shape=brain_params.vector_action_space_size,
        discrete=brain_params.vector_action_space_type == "discrete",
    )


def make_fake_trajectory(
    length: int,
    max_step_complete: bool = False,
    vec_obs_size: int = 1,
    num_vis_obs: int = 1,
    action_space: List[int] = None,
    memory_size: int = 10,
    is_discrete: bool = True,
) -> Trajectory:
    """
    Makes a fake trajectory of length length. If max_step_complete,
    the trajectory is terminated by a max step rather than a done.
    """
    if action_space is None:
        action_space = [2]
    steps_list = []
    for _i in range(length - 1):
        obs = []
        for _j in range(num_vis_obs):
            obs.append(np.ones((84, 84, 3), dtype=np.float32))
        obs.append(np.ones(vec_obs_size, dtype=np.float32))
        reward = 1.0
        done = False
        if is_discrete:
            action_size = len(action_space)
            action_probs = np.ones(np.sum(action_space), dtype=np.float32)
        else:
            action_size = action_space[0]
            action_probs = np.ones((action_size), dtype=np.float32)
        action = np.zeros(action_size, dtype=np.float32)
        action_pre = np.zeros(action_size, dtype=np.float32)
        action_mask = (
            [[False for _ in range(branch)] for branch in action_space]
            if is_discrete
            else None
        )
        prev_action = np.ones(action_size, dtype=np.float32)
        max_step = False
        memory = np.ones(memory_size, dtype=np.float32)
        agent_id = "test_agent"
        behavior_id = "test_brain"
        experience = AgentExperience(
            obs=obs,
            reward=reward,
            done=done,
            action=action,
            action_probs=action_probs,
            action_pre=action_pre,
            action_mask=action_mask,
            prev_action=prev_action,
            max_step=max_step,
            memory=memory,
        )
        steps_list.append(experience)
    last_experience = AgentExperience(
        obs=obs,
        reward=reward,
        done=not max_step_complete,
        action=action,
        action_probs=action_probs,
        action_pre=action_pre,
        action_mask=action_mask,
        prev_action=prev_action,
        max_step=max_step_complete,
        memory=memory,
    )
    steps_list.append(last_experience)
    return Trajectory(
        steps=steps_list, agent_id=agent_id, behavior_id=behavior_id, next_obs=obs
    )


def simulate_rollout(
    length: int,
    brain_params: BrainParameters,
    memory_size: int = 10,
    exclude_key_list: List[str] = None,
) -> AgentBuffer:
    vec_obs_size = brain_params.vector_observation_space_size
    num_vis_obs = brain_params.number_visual_observations
    action_space = brain_params.vector_action_space_size
    is_discrete = brain_params.vector_action_space_type == "discrete"

    trajectory = make_fake_trajectory(
        length,
        vec_obs_size=vec_obs_size,
        num_vis_obs=num_vis_obs,
        action_space=action_space,
        memory_size=memory_size,
        is_discrete=is_discrete,
    )
    buffer = trajectory.to_agentbuffer()
    # If a key_list was given, remove those keys
    if exclude_key_list:
        for key in exclude_key_list:
            if key in buffer:
                buffer.pop(key)
    return buffer


def setup_mock_brain(
    use_discrete,
    use_visual,
    discrete_action_space=None,
    vector_action_space=None,
    vector_obs_space=8,
):
    # defaults
    discrete_action_space = (
        [3, 3, 3, 2] if discrete_action_space is None else discrete_action_space
    )
    vector_action_space = [2] if vector_action_space is None else vector_action_space

    if not use_visual:
        mock_brain = create_mock_brainparams(
            vector_action_space_type="discrete" if use_discrete else "continuous",
            vector_action_space_size=discrete_action_space
            if use_discrete
            else vector_action_space,
            vector_observation_space_size=vector_obs_space,
        )
    else:
        mock_brain = create_mock_brainparams(
            vector_action_space_type="discrete" if use_discrete else "continuous",
            vector_action_space_size=discrete_action_space
            if use_discrete
            else vector_action_space,
            vector_observation_space_size=0,
            number_visual_observations=1,
        )
    return mock_brain


def create_mock_3dball_brain():
    mock_brain = create_mock_brainparams(
        vector_action_space_type="continuous",
        vector_action_space_size=[2],
        vector_observation_space_size=8,
    )
    mock_brain.brain_name = "Ball3DBrain"
    return mock_brain


def create_mock_pushblock_brain():
    mock_brain = create_mock_brainparams(
        vector_action_space_type="discrete",
        vector_action_space_size=[7],
        vector_observation_space_size=70,
    )
    mock_brain.brain_name = "PushblockLearning"
    return mock_brain


def create_mock_banana_brain():
    mock_brain = create_mock_brainparams(
        number_visual_observations=1,
        vector_action_space_type="discrete",
        vector_action_space_size=[3, 3, 3, 2],
        vector_observation_space_size=0,
    )
    return mock_brain


def make_brain_parameters(
    discrete_action: bool = False,
    visual_inputs: int = 0,
    brain_name: str = "RealFakeBrain",
    vec_obs_size: int = 6,
) -> BrainParameters:
    resolutions = [
        CameraResolution(width=30, height=40, num_channels=3)
        for _ in range(visual_inputs)
    ]

    return BrainParameters(
        vector_observation_space_size=vec_obs_size,
        camera_resolutions=resolutions,
        vector_action_space_size=[2],
        vector_action_descriptions=["", ""],
        vector_action_space_type=int(not discrete_action),
        brain_name=brain_name,
    )

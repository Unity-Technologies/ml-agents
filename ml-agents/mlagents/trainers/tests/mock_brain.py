import unittest.mock as mock
import numpy as np

from mlagents.envs.brain import CameraResolution, BrainParameters
from mlagents.trainers.buffer import Buffer


def create_mock_brainparams(
    number_visual_observations=0,
    num_stacked_vector_observations=1,
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
    mock_brain.return_value.num_stacked_vector_observations = (
        num_stacked_vector_observations
    )
    mock_brain.return_value.vector_action_space_type = vector_action_space_type
    mock_brain.return_value.vector_observation_space_size = (
        vector_observation_space_size
    )
    camrez = CameraResolution(height=84, width=84, num_channels=3)
    mock_brain.return_value.camera_resolutions = [camrez] * number_visual_observations
    mock_brain.return_value.vector_action_space_size = vector_action_space_size
    mock_brain.return_value.brain_name = "MockBrain"
    return mock_brain()


def create_mock_braininfo(
    num_agents=1,
    num_vector_observations=0,
    num_vis_observations=0,
    num_vector_acts=2,
    discrete=False,
    num_discrete_branches=1,
):
    """
    Creates a mock BrainInfo with observations. Imitates constant
    vector/visual observations, rewards, dones, and agents.

    :int num_agents: Number of "agents" to imitate in your BrainInfo values.
    :int num_vector_observations: Number of "observations" in your observation space
    :int num_vis_observations: Number of "observations" in your observation space
    :int num_vector_acts: Number of actions in your action space
    :bool discrete: Whether or not action space is discrete
    """
    mock_braininfo = mock.Mock()

    mock_braininfo.return_value.visual_observations = num_vis_observations * [
        np.ones((num_agents, 84, 84, 3))
    ]
    mock_braininfo.return_value.vector_observations = np.array(
        num_agents * [num_vector_observations * [1]]
    )
    if discrete:
        mock_braininfo.return_value.previous_vector_actions = np.array(
            num_agents * [num_discrete_branches * [0.5]]
        )
        mock_braininfo.return_value.action_masks = np.array(
            num_agents * [num_vector_acts * [1.0]]
        )
    else:
        mock_braininfo.return_value.previous_vector_actions = np.array(
            num_agents * [num_vector_acts * [0.5]]
        )
    mock_braininfo.return_value.memories = np.ones((num_agents, 8))
    mock_braininfo.return_value.rewards = num_agents * [1.0]
    mock_braininfo.return_value.local_done = num_agents * [False]
    mock_braininfo.return_value.text_observations = num_agents * [""]
    mock_braininfo.return_value.previous_text_actions = num_agents * [""]
    mock_braininfo.return_value.max_reached = num_agents * [100]
    mock_braininfo.return_value.action_masks = num_agents * [num_vector_acts * [1.0]]
    mock_braininfo.return_value.agents = range(0, num_agents)
    return mock_braininfo()


def setup_mock_unityenvironment(mock_env, mock_brain, mock_braininfo):
    """
    Takes a mock UnityEnvironment and adds the appropriate properties, defined by the mock
    BrainParameters and BrainInfo.

    :Mock mock_env: A mock UnityEnvironment, usually empty.
    :Mock mock_brain: A mock Brain object that specifies the params of this environment.
    :Mock mock_braininfo: A mock BrainInfo object that will be returned at each step and reset.
    """
    brain_name = mock_brain.brain_name
    mock_env.return_value.academy_name = "MockAcademy"
    mock_env.return_value.brains = {brain_name: mock_brain}
    mock_env.return_value.external_brain_names = [brain_name]
    mock_env.return_value.reset.return_value = {brain_name: mock_braininfo}
    mock_env.return_value.step.return_value = {brain_name: mock_braininfo}


def simulate_rollout(env, policy, buffer_init_samples, exclude_key_list=None):
    brain_info_list = []
    for i in range(buffer_init_samples):
        brain_info_list.append(env.step()[env.external_brain_names[0]])
    buffer = create_buffer(brain_info_list, policy.brain, policy.sequence_length)
    # If a key_list was given, remove those keys
    if exclude_key_list:
        for key in exclude_key_list:
            if key in buffer.update_buffer:
                buffer.update_buffer.pop(key)
    return buffer


def create_buffer(brain_infos, brain_params, sequence_length, memory_size=8):
    buffer = Buffer()
    # Make a buffer
    for idx, experience in enumerate(brain_infos):
        if idx > len(brain_infos) - 2:
            break
        current_brain_info = brain_infos[idx]
        next_brain_info = brain_infos[idx + 1]
        buffer[0].last_brain_info = current_brain_info
        buffer[0]["done"].append(next_brain_info.local_done[0])
        buffer[0]["rewards"].append(next_brain_info.rewards[0])
        for i in range(brain_params.number_visual_observations):
            buffer[0]["visual_obs%d" % i].append(
                current_brain_info.visual_observations[i][0]
            )
            buffer[0]["next_visual_obs%d" % i].append(
                current_brain_info.visual_observations[i][0]
            )
        if brain_params.vector_observation_space_size > 0:
            buffer[0]["vector_obs"].append(current_brain_info.vector_observations[0])
            buffer[0]["next_vector_in"].append(
                current_brain_info.vector_observations[0]
            )
        buffer[0]["actions"].append(next_brain_info.previous_vector_actions[0])
        buffer[0]["prev_action"].append(current_brain_info.previous_vector_actions[0])
        buffer[0]["masks"].append(1.0)
        buffer[0]["advantages"].append(1.0)
        if brain_params.vector_action_space_type == "discrete":
            buffer[0]["action_probs"].append(
                np.ones(sum(brain_params.vector_action_space_size))
            )
        else:
            buffer[0]["action_probs"].append(np.ones(buffer[0]["actions"][0].shape))
        buffer[0]["actions_pre"].append(np.ones(buffer[0]["actions"][0].shape))
        buffer[0]["random_normal_epsilon"].append(
            np.ones(buffer[0]["actions"][0].shape)
        )
        buffer[0]["action_mask"].append(
            np.ones(np.sum(brain_params.vector_action_space_size))
        )
        buffer[0]["memory"].append(np.ones(memory_size))

    buffer.append_update_buffer(0, batch_size=None, training_length=sequence_length)
    return buffer


def setup_mock_env_and_brains(
    mock_env,
    use_discrete,
    use_visual,
    num_agents=12,
    discrete_action_space=[3, 3, 3, 2],
    vector_action_space=[2],
    vector_obs_space=8,
):
    if not use_visual:
        mock_brain = create_mock_brainparams(
            vector_action_space_type="discrete" if use_discrete else "continuous",
            vector_action_space_size=discrete_action_space
            if use_discrete
            else vector_action_space,
            vector_observation_space_size=vector_obs_space,
        )
        mock_braininfo = create_mock_braininfo(
            num_agents=num_agents,
            num_vector_observations=vector_obs_space,
            num_vector_acts=sum(
                discrete_action_space if use_discrete else vector_action_space
            ),
            discrete=use_discrete,
            num_discrete_branches=len(discrete_action_space),
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
        mock_braininfo = create_mock_braininfo(
            num_agents=num_agents,
            num_vis_observations=1,
            num_vector_acts=sum(
                discrete_action_space if use_discrete else vector_action_space
            ),
            discrete=use_discrete,
            num_discrete_branches=len(discrete_action_space),
        )
    setup_mock_unityenvironment(mock_env, mock_brain, mock_braininfo)
    env = mock_env()
    return env, mock_brain, mock_braininfo


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
    stack: bool = True,
    brain_name: str = "RealFakeBrain",
    vec_obs_size: int = 3,
) -> BrainParameters:
    resolutions = [
        CameraResolution(width=30, height=40, num_channels=3)
        for _ in range(visual_inputs)
    ]

    return BrainParameters(
        vector_observation_space_size=vec_obs_size,
        num_stacked_vector_observations=2 if stack else 1,
        camera_resolutions=resolutions,
        vector_action_space_size=[2],
        vector_action_descriptions=["", ""],
        vector_action_space_type=int(not discrete_action),
        brain_name=brain_name,
    )

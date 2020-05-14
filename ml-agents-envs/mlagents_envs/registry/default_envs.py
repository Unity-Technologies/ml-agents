from mlagents_envs.registry.unity_env_registry import UnityEnvRegistry
from mlagents_envs.registry.remote_registry_entry import RemoteRegistryEntry


registry = UnityEnvRegistry()

# 3dBall
registry.register(
    RemoteRegistryEntry(
        identifier="3DBall",
        expected_reward=100,
        description="""
## 3DBall: 3D Balance Ball
- Set-up: A balance-ball task, where the agent balances the ball on it's head.
- Goal: The agent must balance the ball on it's head for as long as possible.
- Agents: The environment contains 12 agents of the same kind, all using the
  same Behavior Parameters.
- Agent Reward Function:
  - +0.1 for every step the ball remains on it's head.
  - -1.0 if the ball falls off.
- Behavior Parameters:
  - Vector Observation space: 8 variables corresponding to rotation of the agent
    cube, and position and velocity of ball.
  - Vector Observation space (Hard Version): 5 variables corresponding to
    rotation of the agent cube and position of ball.
  - Vector Action space: (Continuous) Size of 2, with one value corresponding to
    X-rotation, and the other to Z-rotation.
  - Visual Observations: None.
- Float Properties: Three
  - scale: Specifies the scale of the ball in the 3 dimensions (equal across the
    three dimensions)
    - Default: 1
    - Recommended Minimum: 0.2
    - Recommended Maximum: 5
  - gravity: Magnitude of gravity
    - Default: 9.81
    - Recommended Minimum: 4
    - Recommended Maximum: 105
  - mass: Specifies mass of the ball
    - Default: 1
    - Recommended Minimum: 0.1
    - Recommended Maximum: 20
- Benchmark Mean Reward: 100
        """,
        linux_url=None,
        darwin_url="https://storage.googleapis.com/mlagents-test-environments/1.0.0/darwin/3DBall.zip",  # noqa: E501
        win_url=None,
    )
)

# Basic
registry.register(
    RemoteRegistryEntry(
        identifier="Basic",
        expected_reward=0.93,
        description="""
## Basic
- Set-up: A linear movement task where the agent must move left or right to
  rewarding states.
- Goal: Move to the most reward state.
- Agents: The environment contains one agent.
- Agent Reward Function:
  - -0.01 at each step
  - +0.1 for arriving at suboptimal state.
  - +1.0 for arriving at optimal state.
- Behavior Parameters:
  - Vector Observation space: One variable corresponding to current state.
  - Vector Action space: (Discrete) Two possible actions (Move left, move
    right).
  - Visual Observations: None
- Float Properties: None
- Benchmark Mean Reward: 0.93
        """,
        linux_url=None,
        darwin_url="https://storage.googleapis.com/mlagents-test-environments/1.0.0/darwin/Basic.zip",  # noqa: E501
        win_url=None,
    )
)

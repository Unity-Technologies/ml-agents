# Unity ML-Agents Python Low Level API

The `mlagents` Python package contains two components: a low level API which
allows you to interact directly with a Unity Environment (`mlagents_envs`) and
an entry point to train (`mlagents-learn`) which allows you to train agents in
Unity Environments using our implementations of reinforcement learning or
imitation learning. This document describes how to use the `mlagents_envs` API.
For information on using `mlagents-learn`, see [here](Training-ML-Agents.md).

The Python Low Level API can be used to interact directly with your Unity
learning environment. As such, it can serve as the basis for developing and
evaluating new learning algorithms.

## mlagents_envs

The ML-Agents Toolkit Low Level API is a Python API for controlling the
simulation loop of an environment or game built with Unity. This API is used by
the training algorithms inside the ML-Agent Toolkit, but you can also write your
own Python programs using this API.

The key objects in the Python API include:

- **UnityEnvironment** — the main interface between the Unity application and
  your code. Use UnityEnvironment to start and control a simulation or training
  session.
- **BehaviorName** - is a string that identifies a behavior in the simulation.
- **AgentId** - is an `int` that serves as unique identifier for Agents in the
  simulation.
- **DecisionSteps** — contains the data from Agents belonging to the same
  "Behavior" in the simulation, such as observations and rewards. Only Agents
  that requested a decision since the last call to `env.step()` are in the
  DecisionSteps object.
- **TerminalSteps** — contains the data from Agents belonging to the same
  "Behavior" in the simulation, such as observations and rewards. Only Agents
  whose episode ended since the last call to `env.step()` are in the
  TerminalSteps object.
- **BehaviorSpec** — describes the shape of the observation data inside
  DecisionSteps and TerminalSteps as well as the expected action shapes.

These classes are all defined in the
[base_env](../ml-agents-envs/mlagents_envs/base_env.py) script.

An Agent "Behavior" is a group of Agents identified by a `BehaviorName` that
share the same observations and action types (described in their
`BehaviorSpec`). You can think about Agent Behavior as a group of agents that
will share the same policy. All Agents with the same behavior have the same goal
and reward signals.

To communicate with an Agent in a Unity environment from a Python program, the
Agent in the simulation must have `Behavior Parameters` set to communicate. You
must set the `Behavior Type` to `Default` and give it a `Behavior Name`.

_Notice: Currently communication between Unity and Python takes place over an
open socket without authentication. As such, please make sure that the network
where training takes place is secure. This will be addressed in a future
release._

## Loading a Unity Environment

Python-side communication happens through `UnityEnvironment` which is located in
[`environment.py`](../ml-agents-envs/mlagents_envs/environment.py). To load a
Unity environment from a built binary file, put the file in the same directory
as `envs`. For example, if the filename of your Unity environment is `3DBall`,
in python, run:

```python
from mlagents_envs.environment import UnityEnvironment
# This is a non-blocking call that only loads the environment.
env = UnityEnvironment(file_name="3DBall", seed=1, side_channels=[])
# Start interacting with the evironment.
env.reset()
behavior_names = env.behavior_spec.keys()
...
```
**NOTE:** Please read [Interacting with a Unity Environment](#interacting-with-a-unity-environment)
to read more about how you can interact with the Unity environment from Python.

- `file_name` is the name of the environment binary (located in the root
  directory of the python project).
- `worker_id` indicates which port to use for communication with the
  environment. For use in parallel training regimes such as A3C.
- `seed` indicates the seed to use when generating random numbers during the
  training process. In environments which are deterministic, setting the seed
  enables reproducible experimentation by ensuring that the environment and
  trainers utilize the same random seed.
- `side_channels` provides a way to exchange data with the Unity simulation that
  is not related to the reinforcement learning loop. For example: configurations
  or properties. More on them in the
  [Modifying the environment from Python](Python-API.md#modifying-the-environment-from-python)
  section.

If you want to directly interact with the Editor, you need to use
`file_name=None`, then press the **Play** button in the Editor when the message
_"Start training by pressing the Play button in the Unity Editor"_ is displayed
on the screen

### Interacting with a Unity Environment

#### The BaseEnv interface

A `BaseEnv` has the following methods:

- **Reset : `env.reset()`** Sends a signal to reset the environment. Returns
  None.
- **Step : `env.step()`** Sends a signal to step the environment. Returns None.
  Note that a "step" for Python does not correspond to either Unity `Update` nor
  `FixedUpdate`. When `step()` or `reset()` is called, the Unity simulation will
  move forward until an Agent in the simulation needs a input from Python to
  act.
- **Close : `env.close()`** Sends a shutdown signal to the environment and
  terminates the communication.
- **Behavior Specs : `env.behavior_specs`** Returns a Mapping of
  `BehaviorName` to `BehaviorSpec` objects (read only).
  A `BehaviorSpec` contains information such as the observation shapes, the
  action type (multi-discrete or continuous) and the action shape. Note that
  the `BehaviorSpec` for a specific group is fixed throughout the simulation.
  The number of entries in the Mapping can change over time in the simulation
  if new Agent behaviors are created in the simulation.
- **Get Steps : `env.get_steps(behavior_name: str)`** Returns a tuple
  `DecisionSteps, TerminalSteps` corresponding to the behavior_name given as
  input. The `DecisionSteps` contains information about the state of the agents
  **that need an action this step** and have the behavior behavior_name. The
  `TerminalSteps` contains information about the state of the agents **whose
  episode ended** and have the behavior behavior_name. Both `DecisionSteps` and
  `TerminalSteps` contain information such as the observations, the rewards and
  the agent identifiers. `DecisionSteps` also contains action masks for the next
  action while `TerminalSteps` contains the reason for termination (did the
  Agent reach its maximum step and was interrupted). The data is in `np.array`
  of which the first dimension is always the number of agents note that the
  number of agents is not guaranteed to remain constant during the simulation
  and it is not unusual to have either `DecisionSteps` or `TerminalSteps`
  contain no Agents at all.
- **Set Actions :`env.set_actions(behavior_name: str, action: np.array)`** Sets
  the actions for a whole agent group. `action` is a 2D `np.array` of
  `dtype=np.int32` in the discrete action case and `dtype=np.float32` in the
  continuous action case. The first dimension of `action` is the number of
  agents that requested a decision since the last call to `env.step()`. The
  second dimension is the number of discrete actions in multi-discrete action
  type and the number of actions in continuous action type.
- **Set Action for Agent :
  `env.set_action_for_agent(agent_group: str, agent_id: int, action: np.array)`**
  Sets the action for a specific Agent in an agent group. `agent_group` is the
  name of the group the Agent belongs to and `agent_id` is the integer
  identifier of the Agent. Action is a 1D array of type `dtype=np.int32` and
  size equal to the number of discrete actions in multi-discrete action type and
  of type `dtype=np.float32` and size equal to the number of actions in
  continuous action type.

**Note:** If no action is provided for an agent group between two calls to
`env.step()` then the default action will be all zeros (in either discrete or
continuous action space)

#### DecisionSteps and DecisionStep

`DecisionSteps` (with `s`) contains information about a whole batch of Agents
while `DecisionStep` (no `s`) only contains information about a single Agent.

A `DecisionSteps` has the following fields :

- `obs` is a list of numpy arrays observations collected by the group of agent.
  The first dimension of the array corresponds to the batch size of the group
  (number of agents requesting a decision since the last call to `env.step()`).
- `reward` is a float vector of length batch size. Corresponds to the rewards
  collected by each agent since the last simulation step.
- `agent_id` is an int vector of length batch size containing unique identifier
  for the corresponding Agent. This is used to track Agents across simulation
  steps.
- `action_mask` is an optional list of two dimensional array of booleans. Only
  available in multi-discrete action space type. Each array corresponds to an
  action branch. The first dimension of each array is the batch size and the
  second contains a mask for each action of the branch. If true, the action is
  not available for the agent during this simulation step.

It also has the two following methods:

- `len(DecisionSteps)` Returns the number of agents requesting a decision since
  the last call to `env.step()`.
- `DecisionSteps[agent_id]` Returns a `DecisionStep` for the Agent with the
  `agent_id` unique identifier.

A `DecisionStep` has the following fields:

- `obs` is a list of numpy arrays observations collected by the agent. (Each
  array has one less dimension than the arrays in `DecisionSteps`)
- `reward` is a float. Corresponds to the rewards collected by the agent since
  the last simulation step.
- `agent_id` is an int and an unique identifier for the corresponding Agent.
- `action_mask` is an optional list of one dimensional array of booleans. Only
  available in multi-discrete action space type. Each array corresponds to an
  action branch. Each array contains a mask for each action of the branch. If
  true, the action is not available for the agent during this simulation step.

#### TerminalSteps and TerminalStep

Similarly to `DecisionSteps` and `DecisionStep`, `TerminalSteps` (with `s`)
contains information about a whole batch of Agents while `TerminalStep` (no `s`)
only contains information about a single Agent.

A `TerminalSteps` has the following fields :

- `obs` is a list of numpy arrays observations collected by the group of agent.
  The first dimension of the array corresponds to the batch size of the group
  (number of agents requesting a decision since the last call to `env.step()`).
- `reward` is a float vector of length batch size. Corresponds to the rewards
  collected by each agent since the last simulation step.
- `agent_id` is an int vector of length batch size containing unique identifier
  for the corresponding Agent. This is used to track Agents across simulation
  steps.
 - `interrupted` is an array of booleans of length batch size. Is true if the
 associated Agent was interrupted since the last decision step. For example,
 if the Agent reached the maximum number of steps for the episode.

It also has the two following methods:

- `len(TerminalSteps)` Returns the number of agents requesting a decision since
  the last call to `env.step()`.
- `TerminalSteps[agent_id]` Returns a `TerminalStep` for the Agent with the
  `agent_id` unique identifier.

A `TerminalStep` has the following fields:

- `obs` is a list of numpy arrays observations collected by the agent. (Each
  array has one less dimension than the arrays in `TerminalSteps`)
- `reward` is a float. Corresponds to the rewards collected by the agent since
  the last simulation step.
- `agent_id` is an int and an unique identifier for the corresponding Agent.
 - `interrupted` is a bool. Is true if the Agent was interrupted since the last
 decision step. For example, if the Agent reached the maximum number of steps for
 the episode.

#### BehaviorSpec

An Agent behavior can either have discrete or continuous actions. To check which
type it is, use `spec.is_action_discrete()` or `spec.is_action_continuous()` to
see which one it is. If discrete, the action tensors are expected to be
`np.int32`. If continuous, the actions are expected to be `np.float32`.

A `BehaviorSpec` has the following fields :

- `observation_shapes` is a List of Tuples of int : Each Tuple corresponds to an
  observation's dimensions (without the number of agents dimension). The shape
  tuples have the same ordering as the ordering of the DecisionSteps,
  DecisionStep, TerminalSteps and TerminalStep.
- `action_type` is the type of data of the action. it can be discrete or
  continuous. If discrete, the action tensors are expected to be `np.int32`. If
  continuous, the actions are expected to be `np.float32`.
- `action_size` is an `int` corresponding to the expected dimension of the
  action array.
  - In continuous action space it is the number of floats that constitute the
    action.
  - In discrete action space (same as multi-discrete) it corresponds to the
    number of branches (the number of independent actions)
- `discrete_action_branches` is a Tuple of int only for discrete action space.
  Each int corresponds to the number of different options for each branch of the
  action. For example : In a game direction input (no movement, left, right) and
  jump input (no jump, jump) there will be two branches (direction and jump),
  the first one with 3 options and the second with 2 options. (`action_size = 2`
  and `discrete_action_branches = (3,2,)`)

### Communicating additional information with the Environment

In addition to the means of communicating between Unity and python described
above, we also provide methods for sharing agent-agnostic information. These
additional methods are referred to as side channels. ML-Agents includes two
ready-made side channels, described below. It is also possible to create custom
side channels to communicate any additional data between a Unity environment and
Python. Instructions for creating custom side channels can be found
[here](Custom-SideChannels.md).

Side channels exist as separate classes which are instantiated, and then passed
as list to the `side_channels` argument of the constructor of the
`UnityEnvironment` class.

```python
channel = MyChannel()

env = UnityEnvironment(side_channels = [channel])
```

**Note** : A side channel will only send/receive messages when `env.step` or
`env.reset()` is called.

#### EngineConfigurationChannel

The `EngineConfiguration` side channel allows you to modify the time-scale,
resolution, and graphics quality of the environment. This can be useful for
adjusting the environment to perform better during training, or be more
interpretable during inference.

`EngineConfigurationChannel` has two methods :

- `set_configuration_parameters` which takes the following arguments:
  - `width`: Defines the width of the display. (Must be set alongside height)
  - `height`: Defines the height of the display. (Must be set alongside width)
  - `quality_level`: Defines the quality level of the simulation.
  - `time_scale`: Defines the multiplier for the deltatime in the simulation. If
    set to a higher value, time will pass faster in the simulation but the
    physics may perform unpredictably.
  - `target_frame_rate`: Instructs simulation to try to render at a specified
    frame rate.
  - `capture_frame_rate` Instructs the simulation to consider time between
    updates to always be constant, regardless of the actual frame rate.
- `set_configuration` with argument config which is an `EngineConfig` NamedTuple
  object.

For example, the following code would adjust the time-scale of the simulation to
be 2x realtime.

```python
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

channel = EngineConfigurationChannel()

env = UnityEnvironment(side_channels=[channel])

channel.set_configuration_parameters(time_scale = 2.0)

i = env.reset()
...
```

#### EnvironmentParameters

The `EnvironmentParameters` will allow you to get and set pre-defined numerical
values in the environment. This can be useful for adjusting environment-specific
settings, or for reading non-agent related information from the environment. You
can call `get_property` and `set_property` on the side channel to read and write
properties.

`EnvironmentParametersChannel` has one methods:

- `set_float_parameter` Sets a float parameter in the Unity Environment.
  - key: The string identifier of the property.
  - value: The float value of the property.

```python
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

channel = EnvironmentParametersChannel()

env = UnityEnvironment(side_channels=[channel])

channel.set_float_parameter("parameter_1", 2.0)

i = env.reset()
...
```

Once a property has been modified in Python, you can access it in C# after the
next call to `step` as follows:

```csharp
var envParameters = Academy.Instance.EnvironmentParameters;
float property1 = envParameters.GetWithDefault("parameter_1", 0.0f);
```

#### Custom side channels

For information on how to make custom side channels for sending additional data
types, see the documentation [here](Custom-SideChannels.md).

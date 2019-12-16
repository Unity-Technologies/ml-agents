# Unity ML-Agents Python Low Level API

The `mlagents` Python package contains two components: a low level API which
allows you to interact directly with a Unity Environment (`mlagents_envs`) and
an entry point to train (`mlagents-learn`) which allows you to train agents in
Unity Environments using our implementations of reinforcement learning or
imitation learning.

You can use the Python Low Level API to interact directly with your learning
environment, and use it to develop new learning algorithms.

## mlagents_envs

The ML-Agents Toolkit Low Level API is a Python API for controlling the simulation
loop of an environment or game built with Unity. This API is used by the
training algorithms inside the ML-Agent Toolkit, but you can also write your own
Python programs using this API. Go [here](../notebooks/getting-started.ipynb)
for a Jupyter Notebook walking through the functionality of the API.

The key objects in the Python API include:

- **UnityEnvironment** — the main interface between the Unity application and
  your code. Use UnityEnvironment to start and control a simulation or training
  session.
- **BatchedStepResult** — contains the data from Agents belonging to the same
  "AgentGroup" in the simulation, such as observations and rewards.
- **AgentGroupSpec** — describes the shape of the data inside a BatchedStepResult.
  For example, provides the dimensions of the observations of a group.

These classes are all defined in the [base_env](../ml-agents-envs/mlagents_envs/base_env.py)
script.

An Agent Group is a group of Agents identified by a string name that share the same
observations and action types. You can think about Agent Group as a group of agents
that will share the same policy or behavior. All Agents in a group have the same goal
and reward signals.

To communicate with an Agent in a Unity environment from a Python program, the
Agent in the simulation must have `Behavior Parameters` set to communicate. You
must set the `Behavior Type` to `Default` and give it a `Behavior Name`.

__Note__: The `Behavior Name` corresponds to the Agent Group name on Python.

_Notice: Currently communication between Unity and Python takes place over an
open socket without authentication. As such, please make sure that the network
where training takes place is secure. This will be addressed in a future
release._

## Loading a Unity Environment

Python-side communication happens through `UnityEnvironment` which is located in
[`environment.py`](../ml-agents-envs/mlagents_envs/environment.py). To load
a Unity environment from a built binary file, put the file in the same directory
as `envs`. For example, if the filename of your Unity environment is 3DBall.app, in python, run:

```python
from mlagents_envs.environment import UnityEnvironment
env = UnityEnvironment(file_name="3DBall", base_port=5005, seed=1, side_channels=[])
```

- `file_name` is the name of the environment binary (located in the root
  directory of the python project).
- `worker_id` indicates which port to use for communication with the
  environment. For use in parallel training regimes such as A3C.
- `seed` indicates the seed to use when generating random numbers during the
  training process. In environments which do not involve physics calculations,
  setting the seed enables reproducible experimentation by ensuring that the
  environment and trainers utilize the same random seed.
- `side_channels` provides a way to exchange data with the Unity simulation that
  is not related to the reinforcement learning loop. For example: configurations
  or properties. More on them in the [Modifying the environment from Python](Python-API.md#modifying-the-environment-from-python) section.

If you want to directly interact with the Editor, you need to use
`file_name=None`, then press the :arrow_forward: button in the Editor when the
message _"Start training by pressing the Play button in the Unity Editor"_ is
displayed on the screen

### Interacting with a Unity Environment

#### The BaseEnv interface

A `BaseEnv` has the following methods:

 - **Reset : `env.reset()`** Sends a signal to reset the environment. Returns None.
 - **Step : `env.step()`** Sends a signal to step the environment. Returns None.
   Note that a "step" for Python does not correspond to either Unity `Update` nor
   `FixedUpdate`. When `step()` or `reset()` is called, the Unity simulation will
   move forward until an Agent in the simulation needs a input from Python to act.
 - **Close : `env.close()`** Sends a shutdown signal to the environment and terminates
   the communication.
 - **Get Agent Group Names : `env.get_agent_groups()`** Returns a list of agent group ids.
   Note that the number of groups can change over time in the simulation if new
   agent groups are created in the simulation.
 - **Get Agent Group Spec : `env.get_agent_group_spec(agent_group: str)`** Returns
   the `AgentGroupSpec` corresponding to the agent_group given as input. An
   `AgentGroupSpec` contains information such as the observation shapes, the action
   type (multi-discrete or continuous) and the action shape. Note that the `AgentGroupSpec`
   for a specific group is fixed throughout the simulation.
 - **Get Batched Step Result for Agent Group : `env.get_step_result(agent_group: str)`**
   Returns a `BatchedStepResult` corresponding to the agent_group given as input.
   A `BatchedStepResult` contains information about the state of the agents in a group
   such as the observations, the rewards, the done flags and the agent identifiers. The
   data is in `np.array` of which the first dimension is always the number of agents which
   requested a decision in the simulation since the last call to `env.step()` note that the
   number of agents is not guaranteed to remain constant during the simulation.
 - **Set Actions for Agent Group :`env.set_actions(agent_group: str, action: np.array)`**
   Sets the actions for a whole agent group. `action` is a 2D `np.array` of `dtype=np.int32`
   in the discrete action case and `dtype=np.float32` in the continuous action case.
   The first dimension of `action` is the number of agents that requested a decision
   since the last call to `env.step()`. The second dimension is the number of discrete actions
   in multi-discrete action type and the number of actions in continuous action type.
 - **Set Action for Agent : `env.set_action_for_agent(agent_group: str, agent_id: int, action: np.array)`**
   Sets the action for a specific Agent in an agent group. `agent_group` is the name of the
   group the Agent belongs to and `agent_id` is the integer identifier of the Agent. Action
   is a 1D array of type `dtype=np.int32` and size equal to the number of discrete actions
   in multi-discrete action type and of type `dtype=np.float32` and size equal to the number
   of actions in continuous action type.


__Note:__ If no action is provided for an agent group between two calls to `env.step()` then
the default action will be all zeros (in either discrete or continuous action space)
#### BathedStepResult and StepResult

A `BatchedStepResult` has the following fields :

 - `obs` is a list of numpy arrays observations collected by the group of
 agent. The first dimension of the array corresponds to the batch size of
 the group (number of agents requesting a decision since the last call to
 `env.step()`).
 - `reward` is a float vector of length batch size. Corresponds to the
 rewards collected by each agent since the last simulation step.
 - `done` is an array of booleans of length batch size. Is true if the
 associated Agent was terminated during the last simulation step.
 - `max_step` is an array of booleans of length batch size. Is true if the
 associated Agent reached its maximum number of steps during the last
 simulation step.
 - `agent_id` is an int vector of length batch size containing unique
 identifier for the corresponding Agent. This is used to track Agents
 across simulation steps.
 - `action_mask` is an optional list of two dimensional array of booleans.
 Only available in multi-discrete action space type.
 Each array corresponds to an action branch. The first dimension of each
 array is the batch size and the second contains a mask for each action of
 the branch. If true, the action is not available for the agent during
 this simulation step.

It also has the two following methods:

 - `n_agents()` Returns the number of agents requesting a decision since
 the last call to `env.step()`
 - `get_agent_step_result(agent_id: int)` Returns a `StepResult`
 for the Agent with the `agent_id` unique identifier.

A `StepResult` has the following fields:

 - `obs` is a list of numpy arrays observations collected by the group of
 agent. (Each array has one less dimension than the arrays in `BatchedStepResult`)
 - `reward` is a float. Corresponds to the rewards collected by the agent
 since the last simulation step.
 - `done` is a bool. Is true if the Agent was terminated during the last
 simulation step.
 - `max_step` is a bool. Is true if the Agent reached its maximum number of
 steps during the last simulation step.
 - `agent_id` is an int and an unique identifier for the corresponding Agent.
 - `action_mask` is an optional list of one dimensional array of booleans.
 Only available in multi-discrete action space type.
 Each array corresponds to an action branch. Each array contains a mask
 for each action of the branch. If true, the action is not available for
 the agent during this simulation step.

#### AgentGroupSpec

An Agent group can either have discrete or continuous actions. To check which type
it is, use `spec.is_action_discrete()` or `spec.is_action_continuous()` to see
which one it is. If discrete, the action tensors are expected to be `np.int32`. If
continuous, the actions are expected to be `np.float32`.

An `AgentGroupSpec` has the following fields :

 - `observation_shapes` is a List of Tuples of int : Each Tuple corresponds
 to an observation's dimensions (without the number of agents dimension).
 The shape tuples have the same ordering as the ordering of the
 BatchedStepResult and StepResult.
 - `action_type` is the type of data of the action. it can be discrete or
 continuous. If discrete, the action tensors are expected to be `np.int32`. If
 continuous, the actions are expected to be `np.float32`.
 - `action_size` is an `int` corresponding to the expected dimension of the action
 array.
   - In continuous action space it is the number of floats that constitute the action.
   - In discrete action space (same as multi-discrete) it corresponds to the
   number of branches (the number of independent actions)
 - `discrete_action_branches` is a Tuple of int only for discrete action space. Each int
 corresponds to the number of different options for each branch of the action.
 For example : In a game direction input (no movement, left, right) and jump input
 (no jump, jump) there will be two branches (direction and jump), the first one with 3
 options and the second with 2 options. (`action_size = 2` and
 `discrete_action_branches = (3,2,)`)


### Modifying the environment from Python
The Environment can be modified by using side channels to send data to the
environment. When creating the environment, pass a list of side channels as
`side_channels` argument to the constructor.

__Note__ : A side channel will only send/receive messages when `env.step` is
called.

#### EngineConfigurationChannel
An `EngineConfiguration` will allow you to modify the time scale and graphics quality of the Unity engine.
`EngineConfigurationChannel` has two methods :

 * `set_configuration_parameters` with arguments
   * width: Defines the width of the display. Default 80.
   * height: Defines the height of the display. Default 80.
   * quality_level: Defines the quality level of the simulation. Default 1.
   * time_scale: Defines the multiplier for the deltatime in the simulation. If set to a higher value, time will pass faster in the simulation but the physics might break. Default 20.
   *  target_frame_rate: Instructs simulation to try to render at a specified frame rate. Default -1.
 * `set_configuration` with argument config which is an `EngineConfig`
 NamedTuple object.

For example :
```python
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

channel = EngineConfigurationChannel()

env = UnityEnvironment(base_port = 5004, side_channels = [channel])

channel.set_configuration_parameters(time_scale = 2.0)

i = env.reset()
...
```

#### FloatPropertiesChannel
A `FloatPropertiesChannel` will allow you to get and set float properties
in the environment. You can call get_property and set_property on the
side channel to read and write properties.
`FloatPropertiesChannel` has three methods:

 * `set_property` Sets a property in the Unity Environment.
  * key: The string identifier of the property.
  * value: The float value of the property.
 * `get_property` Gets a property in the Unity Environment. If the property was not found, will return None.
  * key: The string identifier of the property.
 * `list_properties` Returns a list of all the string identifiers of the properties

```python
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.float_properties_channel import FloatPropertiesChannel

channel = FloatPropertiesChannel()

env = UnityEnvironment(base_port = 5004, side_channels = [channel])

channel.set_property("parameter_1", 2.0)

i = env.reset()
...
```

Once a property has been modified in Python, you can access it in C# after the next call to `step` as follows:

```csharp
var academy = FindObjectOfType<Academy>();
var sharedProperties = academy.FloatProperties;
float property1 = sharedProperties.GetPropertyWithDefault("parameter_1", 0.0f);
```

# Table of Contents

* [mlagents\_envs.base\_env](#mlagents_envs.base_env)
  * [DecisionStep](#mlagents_envs.base_env.DecisionStep)
  * [DecisionSteps](#mlagents_envs.base_env.DecisionSteps)
    * [agent\_id\_to\_index](#mlagents_envs.base_env.DecisionSteps.agent_id_to_index)
    * [\_\_getitem\_\_](#mlagents_envs.base_env.DecisionSteps.__getitem__)
    * [empty](#mlagents_envs.base_env.DecisionSteps.empty)
  * [TerminalStep](#mlagents_envs.base_env.TerminalStep)
  * [TerminalSteps](#mlagents_envs.base_env.TerminalSteps)
    * [agent\_id\_to\_index](#mlagents_envs.base_env.TerminalSteps.agent_id_to_index)
    * [\_\_getitem\_\_](#mlagents_envs.base_env.TerminalSteps.__getitem__)
    * [empty](#mlagents_envs.base_env.TerminalSteps.empty)
  * [ActionTuple](#mlagents_envs.base_env.ActionTuple)
    * [discrete\_dtype](#mlagents_envs.base_env.ActionTuple.discrete_dtype)
  * [ActionSpec](#mlagents_envs.base_env.ActionSpec)
    * [is\_discrete](#mlagents_envs.base_env.ActionSpec.is_discrete)
    * [is\_continuous](#mlagents_envs.base_env.ActionSpec.is_continuous)
    * [discrete\_size](#mlagents_envs.base_env.ActionSpec.discrete_size)
    * [empty\_action](#mlagents_envs.base_env.ActionSpec.empty_action)
    * [random\_action](#mlagents_envs.base_env.ActionSpec.random_action)
    * [create\_continuous](#mlagents_envs.base_env.ActionSpec.create_continuous)
    * [create\_discrete](#mlagents_envs.base_env.ActionSpec.create_discrete)
    * [create\_hybrid](#mlagents_envs.base_env.ActionSpec.create_hybrid)
  * [DimensionProperty](#mlagents_envs.base_env.DimensionProperty)
    * [UNSPECIFIED](#mlagents_envs.base_env.DimensionProperty.UNSPECIFIED)
    * [NONE](#mlagents_envs.base_env.DimensionProperty.NONE)
    * [TRANSLATIONAL\_EQUIVARIANCE](#mlagents_envs.base_env.DimensionProperty.TRANSLATIONAL_EQUIVARIANCE)
    * [VARIABLE\_SIZE](#mlagents_envs.base_env.DimensionProperty.VARIABLE_SIZE)
  * [ObservationType](#mlagents_envs.base_env.ObservationType)
    * [DEFAULT](#mlagents_envs.base_env.ObservationType.DEFAULT)
    * [GOAL\_SIGNAL](#mlagents_envs.base_env.ObservationType.GOAL_SIGNAL)
  * [ObservationSpec](#mlagents_envs.base_env.ObservationSpec)
  * [BehaviorSpec](#mlagents_envs.base_env.BehaviorSpec)
  * [BaseEnv](#mlagents_envs.base_env.BaseEnv)
    * [step](#mlagents_envs.base_env.BaseEnv.step)
    * [reset](#mlagents_envs.base_env.BaseEnv.reset)
    * [close](#mlagents_envs.base_env.BaseEnv.close)
    * [behavior\_specs](#mlagents_envs.base_env.BaseEnv.behavior_specs)
    * [set\_actions](#mlagents_envs.base_env.BaseEnv.set_actions)
    * [set\_action\_for\_agent](#mlagents_envs.base_env.BaseEnv.set_action_for_agent)
    * [get\_steps](#mlagents_envs.base_env.BaseEnv.get_steps)
* [mlagents\_envs.environment](#mlagents_envs.environment)
  * [UnityEnvironment](#mlagents_envs.environment.UnityEnvironment)
    * [\_\_init\_\_](#mlagents_envs.environment.UnityEnvironment.__init__)
    * [close](#mlagents_envs.environment.UnityEnvironment.close)
* [mlagents\_envs.registry](#mlagents_envs.registry)
* [mlagents\_envs.registry.unity\_env\_registry](#mlagents_envs.registry.unity_env_registry)
  * [UnityEnvRegistry](#mlagents_envs.registry.unity_env_registry.UnityEnvRegistry)
    * [register](#mlagents_envs.registry.unity_env_registry.UnityEnvRegistry.register)
    * [register\_from\_yaml](#mlagents_envs.registry.unity_env_registry.UnityEnvRegistry.register_from_yaml)
    * [clear](#mlagents_envs.registry.unity_env_registry.UnityEnvRegistry.clear)
    * [\_\_getitem\_\_](#mlagents_envs.registry.unity_env_registry.UnityEnvRegistry.__getitem__)
* [mlagents\_envs.side\_channel](#mlagents_envs.side_channel)
* [mlagents\_envs.side\_channel.raw\_bytes\_channel](#mlagents_envs.side_channel.raw_bytes_channel)
  * [RawBytesChannel](#mlagents_envs.side_channel.raw_bytes_channel.RawBytesChannel)
    * [on\_message\_received](#mlagents_envs.side_channel.raw_bytes_channel.RawBytesChannel.on_message_received)
    * [get\_and\_clear\_received\_messages](#mlagents_envs.side_channel.raw_bytes_channel.RawBytesChannel.get_and_clear_received_messages)
    * [send\_raw\_data](#mlagents_envs.side_channel.raw_bytes_channel.RawBytesChannel.send_raw_data)
* [mlagents\_envs.side\_channel.outgoing\_message](#mlagents_envs.side_channel.outgoing_message)
  * [OutgoingMessage](#mlagents_envs.side_channel.outgoing_message.OutgoingMessage)
    * [\_\_init\_\_](#mlagents_envs.side_channel.outgoing_message.OutgoingMessage.__init__)
    * [write\_bool](#mlagents_envs.side_channel.outgoing_message.OutgoingMessage.write_bool)
    * [write\_int32](#mlagents_envs.side_channel.outgoing_message.OutgoingMessage.write_int32)
    * [write\_float32](#mlagents_envs.side_channel.outgoing_message.OutgoingMessage.write_float32)
    * [write\_float32\_list](#mlagents_envs.side_channel.outgoing_message.OutgoingMessage.write_float32_list)
    * [write\_string](#mlagents_envs.side_channel.outgoing_message.OutgoingMessage.write_string)
    * [set\_raw\_bytes](#mlagents_envs.side_channel.outgoing_message.OutgoingMessage.set_raw_bytes)
* [mlagents\_envs.side\_channel.engine\_configuration\_channel](#mlagents_envs.side_channel.engine_configuration_channel)
  * [EngineConfigurationChannel](#mlagents_envs.side_channel.engine_configuration_channel.EngineConfigurationChannel)
    * [on\_message\_received](#mlagents_envs.side_channel.engine_configuration_channel.EngineConfigurationChannel.on_message_received)
    * [set\_configuration\_parameters](#mlagents_envs.side_channel.engine_configuration_channel.EngineConfigurationChannel.set_configuration_parameters)
    * [set\_configuration](#mlagents_envs.side_channel.engine_configuration_channel.EngineConfigurationChannel.set_configuration)
* [mlagents\_envs.side\_channel.side\_channel\_manager](#mlagents_envs.side_channel.side_channel_manager)
  * [SideChannelManager](#mlagents_envs.side_channel.side_channel_manager.SideChannelManager)
    * [process\_side\_channel\_message](#mlagents_envs.side_channel.side_channel_manager.SideChannelManager.process_side_channel_message)
    * [generate\_side\_channel\_messages](#mlagents_envs.side_channel.side_channel_manager.SideChannelManager.generate_side_channel_messages)
* [mlagents\_envs.side\_channel.stats\_side\_channel](#mlagents_envs.side_channel.stats_side_channel)
  * [StatsSideChannel](#mlagents_envs.side_channel.stats_side_channel.StatsSideChannel)
    * [on\_message\_received](#mlagents_envs.side_channel.stats_side_channel.StatsSideChannel.on_message_received)
    * [get\_and\_reset\_stats](#mlagents_envs.side_channel.stats_side_channel.StatsSideChannel.get_and_reset_stats)
* [mlagents\_envs.side\_channel.incoming\_message](#mlagents_envs.side_channel.incoming_message)
  * [IncomingMessage](#mlagents_envs.side_channel.incoming_message.IncomingMessage)
    * [\_\_init\_\_](#mlagents_envs.side_channel.incoming_message.IncomingMessage.__init__)
    * [read\_bool](#mlagents_envs.side_channel.incoming_message.IncomingMessage.read_bool)
    * [read\_int32](#mlagents_envs.side_channel.incoming_message.IncomingMessage.read_int32)
    * [read\_float32](#mlagents_envs.side_channel.incoming_message.IncomingMessage.read_float32)
    * [read\_float32\_list](#mlagents_envs.side_channel.incoming_message.IncomingMessage.read_float32_list)
    * [read\_string](#mlagents_envs.side_channel.incoming_message.IncomingMessage.read_string)
    * [get\_raw\_bytes](#mlagents_envs.side_channel.incoming_message.IncomingMessage.get_raw_bytes)
* [mlagents\_envs.side\_channel.float\_properties\_channel](#mlagents_envs.side_channel.float_properties_channel)
  * [FloatPropertiesChannel](#mlagents_envs.side_channel.float_properties_channel.FloatPropertiesChannel)
    * [on\_message\_received](#mlagents_envs.side_channel.float_properties_channel.FloatPropertiesChannel.on_message_received)
    * [set\_property](#mlagents_envs.side_channel.float_properties_channel.FloatPropertiesChannel.set_property)
    * [get\_property](#mlagents_envs.side_channel.float_properties_channel.FloatPropertiesChannel.get_property)
    * [list\_properties](#mlagents_envs.side_channel.float_properties_channel.FloatPropertiesChannel.list_properties)
    * [get\_property\_dict\_copy](#mlagents_envs.side_channel.float_properties_channel.FloatPropertiesChannel.get_property_dict_copy)
* [mlagents\_envs.side\_channel.environment\_parameters\_channel](#mlagents_envs.side_channel.environment_parameters_channel)
  * [EnvironmentParametersChannel](#mlagents_envs.side_channel.environment_parameters_channel.EnvironmentParametersChannel)
    * [set\_float\_parameter](#mlagents_envs.side_channel.environment_parameters_channel.EnvironmentParametersChannel.set_float_parameter)
    * [set\_uniform\_sampler\_parameters](#mlagents_envs.side_channel.environment_parameters_channel.EnvironmentParametersChannel.set_uniform_sampler_parameters)
    * [set\_gaussian\_sampler\_parameters](#mlagents_envs.side_channel.environment_parameters_channel.EnvironmentParametersChannel.set_gaussian_sampler_parameters)
    * [set\_multirangeuniform\_sampler\_parameters](#mlagents_envs.side_channel.environment_parameters_channel.EnvironmentParametersChannel.set_multirangeuniform_sampler_parameters)
* [mlagents\_envs.side\_channel.side\_channel](#mlagents_envs.side_channel.side_channel)
  * [SideChannel](#mlagents_envs.side_channel.side_channel.SideChannel)
    * [queue\_message\_to\_send](#mlagents_envs.side_channel.side_channel.SideChannel.queue_message_to_send)
    * [on\_message\_received](#mlagents_envs.side_channel.side_channel.SideChannel.on_message_received)
    * [channel\_id](#mlagents_envs.side_channel.side_channel.SideChannel.channel_id)

<a name="mlagents_envs.base_env"></a>
# mlagents\_envs.base\_env

Python Environment API for the ML-Agents Toolkit
The aim of this API is to expose Agents evolving in a simulation
to perform reinforcement learning on.
This API supports multi-agent scenarios and groups similar Agents (same
observations, actions spaces and behavior) together. These groups of Agents are
identified by their BehaviorName.
For performance reasons, the data of each group of agents is processed in a
batched manner. Agents are identified by a unique AgentId identifier that
allows tracking of Agents across simulation steps. Note that there is no
guarantee that the number or order of the Agents in the state will be
consistent across simulation steps.
A simulation steps corresponds to moving the simulation forward until at least
one agent in the simulation sends its observations to Python again. Since
Agents can request decisions at different frequencies, a simulation step does
not necessarily correspond to a fixed simulation time increment.

<a name="mlagents_envs.base_env.DecisionStep"></a>
## DecisionStep Objects

```python
class DecisionStep(NamedTuple)
```

Contains the data a single Agent collected since the last
simulation step.
 - obs is a list of numpy arrays observations collected by the agent.
 - reward is a float. Corresponds to the rewards collected by the agent
 since the last simulation step.
 - agent_id is an int and an unique identifier for the corresponding Agent.
 - action_mask is an optional list of one dimensional array of booleans.
 Only available when using multi-discrete actions.
 Each array corresponds to an action branch. Each array contains a mask
 for each action of the branch. If true, the action is not available for
 the agent during this simulation step.

<a name="mlagents_envs.base_env.DecisionSteps"></a>
## DecisionSteps Objects

```python
class DecisionSteps(Mapping)
```

Contains the data a batch of similar Agents collected since the last
simulation step. Note that all Agents do not necessarily have new
information to send at each simulation step. Therefore, the ordering of
agents and the batch size of the DecisionSteps are not fixed across
simulation steps.
 - obs is a list of numpy arrays observations collected by the batch of
 agent. Each obs has one extra dimension compared to DecisionStep: the
 first dimension of the array corresponds to the batch size of the batch.
 - reward is a float vector of length batch size. Corresponds to the
 rewards collected by each agent since the last simulation step.
 - agent_id is an int vector of length batch size containing unique
 identifier for the corresponding Agent. This is used to track Agents
 across simulation steps.
 - action_mask is an optional list of two dimensional array of booleans.
 Only available when using multi-discrete actions.
 Each array corresponds to an action branch. The first dimension of each
 array is the batch size and the second contains a mask for each action of
 the branch. If true, the action is not available for the agent during
 this simulation step.

<a name="mlagents_envs.base_env.DecisionSteps.agent_id_to_index"></a>
#### agent\_id\_to\_index

```python
 | @property
 | agent_id_to_index() -> Dict[AgentId, int]
```

**Returns**:

A Dict that maps agent_id to the index of those agents in
this DecisionSteps.

<a name="mlagents_envs.base_env.DecisionSteps.__getitem__"></a>
#### \_\_getitem\_\_

```python
 | __getitem__(agent_id: AgentId) -> DecisionStep
```

returns the DecisionStep for a specific agent.

**Arguments**:

- `agent_id`: The id of the agent

**Returns**:

The DecisionStep

<a name="mlagents_envs.base_env.DecisionSteps.empty"></a>
#### empty

```python
 | @staticmethod
 | empty(spec: "BehaviorSpec") -> "DecisionSteps"
```

Returns an empty DecisionSteps.

**Arguments**:

- `spec`: The BehaviorSpec for the DecisionSteps

<a name="mlagents_envs.base_env.TerminalStep"></a>
## TerminalStep Objects

```python
class TerminalStep(NamedTuple)
```

Contains the data a single Agent collected when its episode ended.
 - obs is a list of numpy arrays observations collected by the agent.
 - reward is a float. Corresponds to the rewards collected by the agent
 since the last simulation step.
 - interrupted is a bool. Is true if the Agent was interrupted since the last
 decision step. For example, if the Agent reached the maximum number of steps for
 the episode.
 - agent_id is an int and an unique identifier for the corresponding Agent.

<a name="mlagents_envs.base_env.TerminalSteps"></a>
## TerminalSteps Objects

```python
class TerminalSteps(Mapping)
```

Contains the data a batch of Agents collected when their episode
terminated. All Agents present in the TerminalSteps have ended their
episode.
 - obs is a list of numpy arrays observations collected by the batch of
 agent. Each obs has one extra dimension compared to DecisionStep: the
 first dimension of the array corresponds to the batch size of the batch.
 - reward is a float vector of length batch size. Corresponds to the
 rewards collected by each agent since the last simulation step.
 - interrupted is an array of booleans of length batch size. Is true if the
 associated Agent was interrupted since the last decision step. For example, if the
 Agent reached the maximum number of steps for the episode.
 - agent_id is an int vector of length batch size containing unique
 identifier for the corresponding Agent. This is used to track Agents
 across simulation steps.

<a name="mlagents_envs.base_env.TerminalSteps.agent_id_to_index"></a>
#### agent\_id\_to\_index

```python
 | @property
 | agent_id_to_index() -> Dict[AgentId, int]
```

**Returns**:

A Dict that maps agent_id to the index of those agents in
this TerminalSteps.

<a name="mlagents_envs.base_env.TerminalSteps.__getitem__"></a>
#### \_\_getitem\_\_

```python
 | __getitem__(agent_id: AgentId) -> TerminalStep
```

returns the TerminalStep for a specific agent.

**Arguments**:

- `agent_id`: The id of the agent

**Returns**:

obs, reward, done, agent_id and optional action mask for a
specific agent

<a name="mlagents_envs.base_env.TerminalSteps.empty"></a>
#### empty

```python
 | @staticmethod
 | empty(spec: "BehaviorSpec") -> "TerminalSteps"
```

Returns an empty TerminalSteps.

**Arguments**:

- `spec`: The BehaviorSpec for the TerminalSteps

<a name="mlagents_envs.base_env.ActionTuple"></a>
## ActionTuple Objects

```python
class ActionTuple(_ActionTupleBase)
```

An object whose fields correspond to actions of different types.
Continuous and discrete actions are numpy arrays of type float32 and
int32, respectively and are type checked on construction.
Dimensions are of (n_agents, continuous_size) and (n_agents, discrete_size),
respectively. Note, this also holds when continuous or discrete size is
zero.

<a name="mlagents_envs.base_env.ActionTuple.discrete_dtype"></a>
#### discrete\_dtype

```python
 | @property
 | discrete_dtype() -> np.dtype
```

The dtype of a discrete action.

<a name="mlagents_envs.base_env.ActionSpec"></a>
## ActionSpec Objects

```python
class ActionSpec(NamedTuple)
```

A NamedTuple containing utility functions and information about the action spaces
for a group of Agents under the same behavior.
- num_continuous_actions is an int corresponding to the number of floats which
constitute the action.
- discrete_branch_sizes is a Tuple of int where each int corresponds to
the number of discrete actions available to the agent on an independent action branch.

<a name="mlagents_envs.base_env.ActionSpec.is_discrete"></a>
#### is\_discrete

```python
 | is_discrete() -> bool
```

Returns true if this Behavior uses discrete actions

<a name="mlagents_envs.base_env.ActionSpec.is_continuous"></a>
#### is\_continuous

```python
 | is_continuous() -> bool
```

Returns true if this Behavior uses continuous actions

<a name="mlagents_envs.base_env.ActionSpec.discrete_size"></a>
#### discrete\_size

```python
 | @property
 | discrete_size() -> int
```

Returns a an int corresponding to the number of discrete branches.

<a name="mlagents_envs.base_env.ActionSpec.empty_action"></a>
#### empty\_action

```python
 | empty_action(n_agents: int) -> ActionTuple
```

Generates ActionTuple corresponding to an empty action (all zeros)
for a number of agents.

**Arguments**:

- `n_agents`: The number of agents that will have actions generated

<a name="mlagents_envs.base_env.ActionSpec.random_action"></a>
#### random\_action

```python
 | random_action(n_agents: int) -> ActionTuple
```

Generates ActionTuple corresponding to a random action (either discrete
or continuous) for a number of agents.

**Arguments**:

- `n_agents`: The number of agents that will have actions generated

<a name="mlagents_envs.base_env.ActionSpec.create_continuous"></a>
#### create\_continuous

```python
 | @staticmethod
 | create_continuous(continuous_size: int) -> "ActionSpec"
```

Creates an ActionSpec that is homogenously continuous

<a name="mlagents_envs.base_env.ActionSpec.create_discrete"></a>
#### create\_discrete

```python
 | @staticmethod
 | create_discrete(discrete_branches: Tuple[int]) -> "ActionSpec"
```

Creates an ActionSpec that is homogenously discrete

<a name="mlagents_envs.base_env.ActionSpec.create_hybrid"></a>
#### create\_hybrid

```python
 | @staticmethod
 | create_hybrid(continuous_size: int, discrete_branches: Tuple[int]) -> "ActionSpec"
```

Creates a hybrid ActionSpace

<a name="mlagents_envs.base_env.DimensionProperty"></a>
## DimensionProperty Objects

```python
class DimensionProperty(IntFlag)
```

The dimension property of a dimension of an observation.

<a name="mlagents_envs.base_env.DimensionProperty.UNSPECIFIED"></a>
#### UNSPECIFIED

No properties specified.

<a name="mlagents_envs.base_env.DimensionProperty.NONE"></a>
#### NONE

No Property of the observation in that dimension. Observation can be processed with
Fully connected networks.

<a name="mlagents_envs.base_env.DimensionProperty.TRANSLATIONAL_EQUIVARIANCE"></a>
#### TRANSLATIONAL\_EQUIVARIANCE

Means it is suitable to do a convolution in this dimension.

<a name="mlagents_envs.base_env.DimensionProperty.VARIABLE_SIZE"></a>
#### VARIABLE\_SIZE

Means that there can be a variable number of observations in this dimension.
The observations are unordered.

<a name="mlagents_envs.base_env.ObservationType"></a>
## ObservationType Objects

```python
class ObservationType(Enum)
```

An Enum which defines the type of information carried in the observation
of the agent.

<a name="mlagents_envs.base_env.ObservationType.DEFAULT"></a>
#### DEFAULT

Observation information is generic.

<a name="mlagents_envs.base_env.ObservationType.GOAL_SIGNAL"></a>
#### GOAL\_SIGNAL

Observation contains goal information for current task.

<a name="mlagents_envs.base_env.ObservationSpec"></a>
## ObservationSpec Objects

```python
class ObservationSpec(NamedTuple)
```

A NamedTuple containing information about the observation of Agents.
- shape is a Tuple of int : It corresponds to the shape of
an observation's dimensions.
- dimension_property is a Tuple of DimensionProperties flag, one flag for each
dimension.
- observation_type is an enum of ObservationType.

<a name="mlagents_envs.base_env.BehaviorSpec"></a>
## BehaviorSpec Objects

```python
class BehaviorSpec(NamedTuple)
```

A NamedTuple containing information about the observation and action
spaces for a group of Agents under the same behavior.
- observation_specs is a List of ObservationSpec NamedTuple containing
information about the information of the Agent's observations such as their shapes.
The order of the ObservationSpec is the same as the order of the observations of an
agent.
- action_spec is an ActionSpec NamedTuple.

<a name="mlagents_envs.base_env.BaseEnv"></a>
## BaseEnv Objects

```python
class BaseEnv(ABC)
```

<a name="mlagents_envs.base_env.BaseEnv.step"></a>
#### step

```python
 | @abstractmethod
 | step() -> None
```

Signals the environment that it must move the simulation forward
by one step.

<a name="mlagents_envs.base_env.BaseEnv.reset"></a>
#### reset

```python
 | @abstractmethod
 | reset() -> None
```

Signals the environment that it must reset the simulation.

<a name="mlagents_envs.base_env.BaseEnv.close"></a>
#### close

```python
 | @abstractmethod
 | close() -> None
```

Signals the environment that it must close.

<a name="mlagents_envs.base_env.BaseEnv.behavior_specs"></a>
#### behavior\_specs

```python
 | @property
 | @abstractmethod
 | behavior_specs() -> MappingType[str, BehaviorSpec]
```

Returns a Mapping from behavior names to behavior specs.
Agents grouped under the same behavior name have the same action and
observation specs, and are expected to behave similarly in the
environment.
Note that new keys can be added to this mapping as new policies are instantiated.

<a name="mlagents_envs.base_env.BaseEnv.set_actions"></a>
#### set\_actions

```python
 | @abstractmethod
 | set_actions(behavior_name: BehaviorName, action: ActionTuple) -> None
```

Sets the action for all of the agents in the simulation for the next
step. The Actions must be in the same order as the order received in
the DecisionSteps.

**Arguments**:

- `behavior_name`: The name of the behavior the agents are part of
- `action`: ActionTuple tuple of continuous and/or discrete action.
Actions are np.arrays with dimensions  (n_agents, continuous_size) and
(n_agents, discrete_size), respectively.

<a name="mlagents_envs.base_env.BaseEnv.set_action_for_agent"></a>
#### set\_action\_for\_agent

```python
 | @abstractmethod
 | set_action_for_agent(behavior_name: BehaviorName, agent_id: AgentId, action: ActionTuple) -> None
```

Sets the action for one of the agents in the simulation for the next
step.

**Arguments**:

- `behavior_name`: The name of the behavior the agent is part of
- `agent_id`: The id of the agent the action is set for
- `action`: ActionTuple tuple of continuous and/or discrete action
Actions are np.arrays with dimensions  (1, continuous_size) and
(1, discrete_size), respectively. Note, this initial dimensions of 1 is because
this action is meant for a single agent.

<a name="mlagents_envs.base_env.BaseEnv.get_steps"></a>
#### get\_steps

```python
 | @abstractmethod
 | get_steps(behavior_name: BehaviorName) -> Tuple[DecisionSteps, TerminalSteps]
```

Retrieves the steps of the agents that requested a step in the
simulation.

**Arguments**:

- `behavior_name`: The name of the behavior the agents are part of

**Returns**:

A tuple containing :
- A DecisionSteps NamedTuple containing the observations,
the rewards, the agent ids and the action masks for the Agents
of the specified behavior. These Agents need an action this step.
- A TerminalSteps NamedTuple containing the observations,
rewards, agent ids and interrupted flags of the agents that had their
episode terminated last step.

<a name="mlagents_envs.environment"></a>
# mlagents\_envs.environment

<a name="mlagents_envs.environment.UnityEnvironment"></a>
## UnityEnvironment Objects

```python
class UnityEnvironment(BaseEnv)
```

<a name="mlagents_envs.environment.UnityEnvironment.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(file_name: Optional[str] = None, worker_id: int = 0, base_port: Optional[int] = None, seed: int = 0, no_graphics: bool = False, no_graphics_monitor: bool = False, timeout_wait: int = 60, additional_args: Optional[List[str]] = None, side_channels: Optional[List[SideChannel]] = None, log_folder: Optional[str] = None, num_areas: int = 1)
```

Starts a new unity environment and establishes a connection with the environment.
Notice: Currently communication between Unity and Python takes place over an open socket without authentication.
Ensure that the network where training takes place is secure.

:string file_name: Name of Unity environment binary. :int base_port: Baseline port number to connect to Unity
environment over. worker_id increments over this. If no environment is specified (i.e. file_name is None),
the DEFAULT_EDITOR_PORT will be used. :int worker_id: Offset from base_port. Used for training multiple
environments simultaneously. :bool no_graphics: Whether to run the Unity simulator in no-graphics mode :bool
no_graphics_monitor: Whether to run the main worker in graphics mode, with the remaining in no-graphics mode
:int timeout_wait: Time (in seconds) to wait for connection from environment. :list args: Addition Unity
command line arguments :list side_channels: Additional side channel for no-rl communication with Unity :str
log_folder: Optional folder to write the Unity Player log file into.  Requires absolute path.

<a name="mlagents_envs.environment.UnityEnvironment.close"></a>
#### close

```python
 | close()
```

Sends a shutdown signal to the unity environment, and closes the socket connection.

<a name="mlagents_envs.registry"></a>
# mlagents\_envs.registry

<a name="mlagents_envs.registry.unity_env_registry"></a>
# mlagents\_envs.registry.unity\_env\_registry

<a name="mlagents_envs.registry.unity_env_registry.UnityEnvRegistry"></a>
## UnityEnvRegistry Objects

```python
class UnityEnvRegistry(Mapping)
```

### UnityEnvRegistry
Provides a library of Unity environments that can be launched without the need
of downloading the Unity Editor.
The UnityEnvRegistry implements a Map, to access an entry of the Registry, use:
```python
registry = UnityEnvRegistry()
entry = registry[<environment_identifyier>]
```
An entry has the following properties :
 * `identifier` : Uniquely identifies this environment
 * `expected_reward` : Corresponds to the reward an agent must obtained for the task
 to be considered completed.
 * `description` : A human readable description of the environment.

To launch a Unity environment from a registry entry, use the `make` method:
```python
registry = UnityEnvRegistry()
env = registry[<environment_identifyier>].make()
```

<a name="mlagents_envs.registry.unity_env_registry.UnityEnvRegistry.register"></a>
#### register

```python
 | register(new_entry: BaseRegistryEntry) -> None
```

Registers a new BaseRegistryEntry to the registry. The
BaseRegistryEntry.identifier value will be used as indexing key.
If two are more environments are registered under the same key, the most
recentry added will replace the others.

<a name="mlagents_envs.registry.unity_env_registry.UnityEnvRegistry.register_from_yaml"></a>
#### register\_from\_yaml

```python
 | register_from_yaml(path_to_yaml: str) -> None
```

Registers the environments listed in a yaml file (either local or remote). Note
that the entries are registered lazily: the registration will only happen when
an environment is accessed.
The yaml file must have the following format :
```yaml
environments:
- <identifier of the first environment>:
    expected_reward: <expected reward of the environment>
    description: | <a multi line description of the environment>
      <continued multi line description>
    linux_url: <The url for the Linux executable zip file>
    darwin_url: <The url for the OSX executable zip file>
    win_url: <The url for the Windows executable zip file>

- <identifier of the second environment>:
    expected_reward: <expected reward of the environment>
    description: | <a multi line description of the environment>
      <continued multi line description>
    linux_url: <The url for the Linux executable zip file>
    darwin_url: <The url for the OSX executable zip file>
    win_url: <The url for the Windows executable zip file>

- ...
```

**Arguments**:

- `path_to_yaml`: A local path or url to the yaml file

<a name="mlagents_envs.registry.unity_env_registry.UnityEnvRegistry.clear"></a>
#### clear

```python
 | clear() -> None
```

Deletes all entries in the registry.

<a name="mlagents_envs.registry.unity_env_registry.UnityEnvRegistry.__getitem__"></a>
#### \_\_getitem\_\_

```python
 | __getitem__(identifier: str) -> BaseRegistryEntry
```

Returns the BaseRegistryEntry with the provided identifier. BaseRegistryEntry
can then be used to make a Unity Environment.

**Arguments**:

- `identifier`: The identifier of the BaseRegistryEntry

**Returns**:

The associated BaseRegistryEntry

<a name="mlagents_envs.side_channel"></a>
# mlagents\_envs.side\_channel

<a name="mlagents_envs.side_channel.raw_bytes_channel"></a>
# mlagents\_envs.side\_channel.raw\_bytes\_channel

<a name="mlagents_envs.side_channel.raw_bytes_channel.RawBytesChannel"></a>
## RawBytesChannel Objects

```python
class RawBytesChannel(SideChannel)
```

This is an example of what the SideChannel for raw bytes exchange would
look like. Is meant to be used for general research purpose.

<a name="mlagents_envs.side_channel.raw_bytes_channel.RawBytesChannel.on_message_received"></a>
#### on\_message\_received

```python
 | on_message_received(msg: IncomingMessage) -> None
```

Is called by the environment to the side channel. Can be called
multiple times per step if multiple messages are meant for that
SideChannel.

<a name="mlagents_envs.side_channel.raw_bytes_channel.RawBytesChannel.get_and_clear_received_messages"></a>
#### get\_and\_clear\_received\_messages

```python
 | get_and_clear_received_messages() -> List[bytes]
```

returns a list of bytearray received from the environment.

<a name="mlagents_envs.side_channel.raw_bytes_channel.RawBytesChannel.send_raw_data"></a>
#### send\_raw\_data

```python
 | send_raw_data(data: bytearray) -> None
```

Queues a message to be sent by the environment at the next call to
step.

<a name="mlagents_envs.side_channel.outgoing_message"></a>
# mlagents\_envs.side\_channel.outgoing\_message

<a name="mlagents_envs.side_channel.outgoing_message.OutgoingMessage"></a>
## OutgoingMessage Objects

```python
class OutgoingMessage()
```

Utility class for forming the message that is written to a SideChannel.
All data is written in little-endian format using the struct module.

<a name="mlagents_envs.side_channel.outgoing_message.OutgoingMessage.__init__"></a>
#### \_\_init\_\_

```python
 | __init__()
```

Create an OutgoingMessage with an empty buffer.

<a name="mlagents_envs.side_channel.outgoing_message.OutgoingMessage.write_bool"></a>
#### write\_bool

```python
 | write_bool(b: bool) -> None
```

Append a boolean value.

<a name="mlagents_envs.side_channel.outgoing_message.OutgoingMessage.write_int32"></a>
#### write\_int32

```python
 | write_int32(i: int) -> None
```

Append an integer value.

<a name="mlagents_envs.side_channel.outgoing_message.OutgoingMessage.write_float32"></a>
#### write\_float32

```python
 | write_float32(f: float) -> None
```

Append a float value. It will be truncated to 32-bit precision.

<a name="mlagents_envs.side_channel.outgoing_message.OutgoingMessage.write_float32_list"></a>
#### write\_float32\_list

```python
 | write_float32_list(float_list: List[float]) -> None
```

Append a list of float values. They will be truncated to 32-bit precision.

<a name="mlagents_envs.side_channel.outgoing_message.OutgoingMessage.write_string"></a>
#### write\_string

```python
 | write_string(s: str) -> None
```

Append a string value. Internally, it will be encoded to ascii, and the
encoded length will also be written to the message.

<a name="mlagents_envs.side_channel.outgoing_message.OutgoingMessage.set_raw_bytes"></a>
#### set\_raw\_bytes

```python
 | set_raw_bytes(buffer: bytearray) -> None
```

Set the internal buffer to a new bytearray. This will overwrite any existing data.

**Arguments**:

- `buffer`:

**Returns**:



<a name="mlagents_envs.side_channel.engine_configuration_channel"></a>
# mlagents\_envs.side\_channel.engine\_configuration\_channel

<a name="mlagents_envs.side_channel.engine_configuration_channel.EngineConfigurationChannel"></a>
## EngineConfigurationChannel Objects

```python
class EngineConfigurationChannel(SideChannel)
```

This is the SideChannel for engine configuration exchange. The data in the
engine configuration is as follows :
 - int width;
 - int height;
 - int qualityLevel;
 - float timeScale;
 - int targetFrameRate;
 - int captureFrameRate;

<a name="mlagents_envs.side_channel.engine_configuration_channel.EngineConfigurationChannel.on_message_received"></a>
#### on\_message\_received

```python
 | on_message_received(msg: IncomingMessage) -> None
```

Is called by the environment to the side channel. Can be called
multiple times per step if multiple messages are meant for that
SideChannel.
Note that Python should never receive an engine configuration from
Unity

<a name="mlagents_envs.side_channel.engine_configuration_channel.EngineConfigurationChannel.set_configuration_parameters"></a>
#### set\_configuration\_parameters

```python
 | set_configuration_parameters(width: Optional[int] = None, height: Optional[int] = None, quality_level: Optional[int] = None, time_scale: Optional[float] = None, target_frame_rate: Optional[int] = None, capture_frame_rate: Optional[int] = None) -> None
```

Sets the engine configuration. Takes as input the configurations of the
engine.

**Arguments**:

- `width`: Defines the width of the display. (Must be set alongside height)
- `height`: Defines the height of the display. (Must be set alongside width)
- `quality_level`: Defines the quality level of the simulation.
- `time_scale`: Defines the multiplier for the deltatime in the
simulation. If set to a higher value, time will pass faster in the
simulation but the physics might break.
- `target_frame_rate`: Instructs simulation to try to render at a
specified frame rate.
- `capture_frame_rate`: Instructs the simulation to consider time between
updates to always be constant, regardless of the actual frame rate.

<a name="mlagents_envs.side_channel.engine_configuration_channel.EngineConfigurationChannel.set_configuration"></a>
#### set\_configuration

```python
 | set_configuration(config: EngineConfig) -> None
```

Sets the engine configuration. Takes as input an EngineConfig.

<a name="mlagents_envs.side_channel.side_channel_manager"></a>
# mlagents\_envs.side\_channel.side\_channel\_manager

<a name="mlagents_envs.side_channel.side_channel_manager.SideChannelManager"></a>
## SideChannelManager Objects

```python
class SideChannelManager()
```

<a name="mlagents_envs.side_channel.side_channel_manager.SideChannelManager.process_side_channel_message"></a>
#### process\_side\_channel\_message

```python
 | process_side_channel_message(data: bytes) -> None
```

Separates the data received from Python into individual messages for each
registered side channel and calls on_message_received on them.

**Arguments**:

- `data`: The packed message sent by Unity

<a name="mlagents_envs.side_channel.side_channel_manager.SideChannelManager.generate_side_channel_messages"></a>
#### generate\_side\_channel\_messages

```python
 | generate_side_channel_messages() -> bytearray
```

Gathers the messages that the registered side channels will send to Unity
and combines them into a single message ready to be sent.

<a name="mlagents_envs.side_channel.stats_side_channel"></a>
# mlagents\_envs.side\_channel.stats\_side\_channel

<a name="mlagents_envs.side_channel.stats_side_channel.StatsSideChannel"></a>
## StatsSideChannel Objects

```python
class StatsSideChannel(SideChannel)
```

Side channel that receives (string, float) pairs from the environment, so that they can eventually
be passed to a StatsReporter.

<a name="mlagents_envs.side_channel.stats_side_channel.StatsSideChannel.on_message_received"></a>
#### on\_message\_received

```python
 | on_message_received(msg: IncomingMessage) -> None
```

Receive the message from the environment, and save it for later retrieval.

**Arguments**:

- `msg`:

**Returns**:



<a name="mlagents_envs.side_channel.stats_side_channel.StatsSideChannel.get_and_reset_stats"></a>
#### get\_and\_reset\_stats

```python
 | get_and_reset_stats() -> EnvironmentStats
```

Returns the current stats, and resets the internal storage of the stats.

**Returns**:



<a name="mlagents_envs.side_channel.incoming_message"></a>
# mlagents\_envs.side\_channel.incoming\_message

<a name="mlagents_envs.side_channel.incoming_message.IncomingMessage"></a>
## IncomingMessage Objects

```python
class IncomingMessage()
```

Utility class for reading the message written to a SideChannel.
Values must be read in the order they were written.

<a name="mlagents_envs.side_channel.incoming_message.IncomingMessage.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(buffer: bytes, offset: int = 0)
```

Create a new IncomingMessage from the bytes.

<a name="mlagents_envs.side_channel.incoming_message.IncomingMessage.read_bool"></a>
#### read\_bool

```python
 | read_bool(default_value: bool = False) -> bool
```

Read a boolean value from the message buffer.

**Arguments**:

- `default_value`: Default value to use if the end of the message is reached.

**Returns**:

The value read from the message, or the default value if the end was reached.

<a name="mlagents_envs.side_channel.incoming_message.IncomingMessage.read_int32"></a>
#### read\_int32

```python
 | read_int32(default_value: int = 0) -> int
```

Read an integer value from the message buffer.

**Arguments**:

- `default_value`: Default value to use if the end of the message is reached.

**Returns**:

The value read from the message, or the default value if the end was reached.

<a name="mlagents_envs.side_channel.incoming_message.IncomingMessage.read_float32"></a>
#### read\_float32

```python
 | read_float32(default_value: float = 0.0) -> float
```

Read a float value from the message buffer.

**Arguments**:

- `default_value`: Default value to use if the end of the message is reached.

**Returns**:

The value read from the message, or the default value if the end was reached.

<a name="mlagents_envs.side_channel.incoming_message.IncomingMessage.read_float32_list"></a>
#### read\_float32\_list

```python
 | read_float32_list(default_value: List[float] = None) -> List[float]
```

Read a list of float values from the message buffer.

**Arguments**:

- `default_value`: Default value to use if the end of the message is reached.

**Returns**:

The value read from the message, or the default value if the end was reached.

<a name="mlagents_envs.side_channel.incoming_message.IncomingMessage.read_string"></a>
#### read\_string

```python
 | read_string(default_value: str = "") -> str
```

Read a string value from the message buffer.

**Arguments**:

- `default_value`: Default value to use if the end of the message is reached.

**Returns**:

The value read from the message, or the default value if the end was reached.

<a name="mlagents_envs.side_channel.incoming_message.IncomingMessage.get_raw_bytes"></a>
#### get\_raw\_bytes

```python
 | get_raw_bytes() -> bytes
```

Get a copy of the internal bytes used by the message.

<a name="mlagents_envs.side_channel.float_properties_channel"></a>
# mlagents\_envs.side\_channel.float\_properties\_channel

<a name="mlagents_envs.side_channel.float_properties_channel.FloatPropertiesChannel"></a>
## FloatPropertiesChannel Objects

```python
class FloatPropertiesChannel(SideChannel)
```

This is the SideChannel for float properties shared with Unity.
You can modify the float properties of an environment with the commands
set_property, get_property and list_properties.

<a name="mlagents_envs.side_channel.float_properties_channel.FloatPropertiesChannel.on_message_received"></a>
#### on\_message\_received

```python
 | on_message_received(msg: IncomingMessage) -> None
```

Is called by the environment to the side channel. Can be called
multiple times per step if multiple messages are meant for that
SideChannel.

<a name="mlagents_envs.side_channel.float_properties_channel.FloatPropertiesChannel.set_property"></a>
#### set\_property

```python
 | set_property(key: str, value: float) -> None
```

Sets a property in the Unity Environment.

**Arguments**:

- `key`: The string identifier of the property.
- `value`: The float value of the property.

<a name="mlagents_envs.side_channel.float_properties_channel.FloatPropertiesChannel.get_property"></a>
#### get\_property

```python
 | get_property(key: str) -> Optional[float]
```

Gets a property in the Unity Environment. If the property was not
found, will return None.

**Arguments**:

- `key`: The string identifier of the property.

**Returns**:

The float value of the property or None.

<a name="mlagents_envs.side_channel.float_properties_channel.FloatPropertiesChannel.list_properties"></a>
#### list\_properties

```python
 | list_properties() -> List[str]
```

Returns a list of all the string identifiers of the properties
currently present in the Unity Environment.

<a name="mlagents_envs.side_channel.float_properties_channel.FloatPropertiesChannel.get_property_dict_copy"></a>
#### get\_property\_dict\_copy

```python
 | get_property_dict_copy() -> Dict[str, float]
```

Returns a copy of the float properties.

**Returns**:



<a name="mlagents_envs.side_channel.environment_parameters_channel"></a>
# mlagents\_envs.side\_channel.environment\_parameters\_channel

<a name="mlagents_envs.side_channel.environment_parameters_channel.EnvironmentParametersChannel"></a>
## EnvironmentParametersChannel Objects

```python
class EnvironmentParametersChannel(SideChannel)
```

This is the SideChannel for sending environment parameters to Unity.
You can send parameters to an environment with the command
set_float_parameter.

<a name="mlagents_envs.side_channel.environment_parameters_channel.EnvironmentParametersChannel.set_float_parameter"></a>
#### set\_float\_parameter

```python
 | set_float_parameter(key: str, value: float) -> None
```

Sets a float environment parameter in the Unity Environment.

**Arguments**:

- `key`: The string identifier of the parameter.
- `value`: The float value of the parameter.

<a name="mlagents_envs.side_channel.environment_parameters_channel.EnvironmentParametersChannel.set_uniform_sampler_parameters"></a>
#### set\_uniform\_sampler\_parameters

```python
 | set_uniform_sampler_parameters(key: str, min_value: float, max_value: float, seed: int) -> None
```

Sets a uniform environment parameter sampler.

**Arguments**:

- `key`: The string identifier of the parameter.
- `min_value`: The minimum of the sampling distribution.
- `max_value`: The maximum of the sampling distribution.
- `seed`: The random seed to initialize the sampler.

<a name="mlagents_envs.side_channel.environment_parameters_channel.EnvironmentParametersChannel.set_gaussian_sampler_parameters"></a>
#### set\_gaussian\_sampler\_parameters

```python
 | set_gaussian_sampler_parameters(key: str, mean: float, st_dev: float, seed: int) -> None
```

Sets a gaussian environment parameter sampler.

**Arguments**:

- `key`: The string identifier of the parameter.
- `mean`: The mean of the sampling distribution.
- `st_dev`: The standard deviation of the sampling distribution.
- `seed`: The random seed to initialize the sampler.

<a name="mlagents_envs.side_channel.environment_parameters_channel.EnvironmentParametersChannel.set_multirangeuniform_sampler_parameters"></a>
#### set\_multirangeuniform\_sampler\_parameters

```python
 | set_multirangeuniform_sampler_parameters(key: str, intervals: List[Tuple[float, float]], seed: int) -> None
```

Sets a multirangeuniform environment parameter sampler.

**Arguments**:

- `key`: The string identifier of the parameter.
- `intervals`: The lists of min and max that define each uniform distribution.
- `seed`: The random seed to initialize the sampler.

<a name="mlagents_envs.side_channel.side_channel"></a>
# mlagents\_envs.side\_channel.side\_channel

<a name="mlagents_envs.side_channel.side_channel.SideChannel"></a>
## SideChannel Objects

```python
class SideChannel(ABC)
```

The side channel just get access to a bytes buffer that will be shared
between C# and Python. For example, We will create a specific side channel
for properties that will be a list of string (fixed size) to float number,
that can be modified by both C# and Python. All side channels are passed
to the Env object at construction.

<a name="mlagents_envs.side_channel.side_channel.SideChannel.queue_message_to_send"></a>
#### queue\_message\_to\_send

```python
 | queue_message_to_send(msg: OutgoingMessage) -> None
```

Queues a message to be sent by the environment at the next call to
step.

<a name="mlagents_envs.side_channel.side_channel.SideChannel.on_message_received"></a>
#### on\_message\_received

```python
 | @abstractmethod
 | on_message_received(msg: IncomingMessage) -> None
```

Is called by the environment to the side channel. Can be called
multiple times per step if multiple messages are meant for that
SideChannel.

<a name="mlagents_envs.side_channel.side_channel.SideChannel.channel_id"></a>
#### channel\_id

```python
 | @property
 | channel_id() -> uuid.UUID
```

**Returns**:

The type of side channel used. Will influence how the data is
processed in the environment.

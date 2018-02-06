# Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`enum `[`BrainType`](#Brain_8cs_1a255eedae6b63fb43e45485914ff30288)            | 
`enum `[`StateType`](#Brain_8cs_1a1615968a92950438f6e67a28e9d56e5c)            | 
`enum `[`ExternalCommand`](#Communicator_8cs_1a2056e4a11471702052b3ee067b38e355)            | 
`enum `[`MonitorType`](#Monitor_8cs_1ac69ce5a28782fdc01f7ba5c236cd5f72)            | The type of monitor the information must be displayed in. <slider> corresponds to a slingle rectangle which width is given by a float between -1 and 1. (green is positive, red is negative) <hist> corresponds to n vertical sliders. <text> is a text field. <bar> is a rectangle of fixed length to represent the proportions of a list of floats.
`class `[`Academy`](#classAcademy) | Create a child class to implement [InitializeAcademy()](#classAcademy_1ab6a884f7a70c4dce4432077d716e886c), [AcademyStep()](#classAcademy_1aec20943228af90888c11a796b9e27777) and [AcademyReset()](#classAcademy_1a0872c23a338aebf18b22ce40d3f72c39). The child class script must be attached to an empty game object in your scene, and there can only be one such object within the scene.
`class `[`Agent`](#classAgent) | Generic functions for parent [Agent](#classAgent) class. Contains all logic for Brain-Agent communication and Agent-Environment interaction.
`class `[`ExternalCommunicator::AgentMessage`](#classExternalCommunicator_1_1AgentMessage) | 
`class `[`Brain`](#classBrain) | Contains all high-level [Brain](#classBrain) logic. Add this component to an empty GameObject in your scene and drag this GameObject into your [Academy](#classAcademy) to make it a child in the hierarchy. Contains a set of CoreBrains, which each correspond to a different method for deciding actions.
`class `[`BrainParameters`](#classBrainParameters) | Should be modified via the Editor Inspector. Defines brain-specific parameters
`class `[`CoreBrainExternal`](#classCoreBrainExternal) | [CoreBrain](#interfaceCoreBrain) which decides actions via communication with an external system such as Python.
`class `[`CoreBrainHeuristic`](#classCoreBrainHeuristic) | [CoreBrain](#interfaceCoreBrain) which decides actions using developer-provided [Decision.cs](#Decision_8cs) script.
`class `[`CoreBrainInternal`](#classCoreBrainInternal) | [CoreBrain](#interfaceCoreBrain) which decides actions using internally embedded TensorFlow model.
`class `[`CoreBrainPlayer`](#classCoreBrainPlayer) | [CoreBrain](#interfaceCoreBrain) which decides actions using Player input.
`class `[`ExternalCommunicator`](#classExternalCommunicator) | Responsible for communication with Python API.
`class `[`Monitor`](#classMonitor) | [Monitor](#classMonitor) is used to display information. Use the log function to add information to your monitor.
`class `[`ExternalCommunicator::ResetParametersMessage`](#classExternalCommunicator_1_1ResetParametersMessage) | 
`class `[`ScreenConfiguration`](#classScreenConfiguration) | 
`class `[`ExternalCommunicator::StepMessage`](#classExternalCommunicator_1_1StepMessage) | 
`class `[`UnityAgentsException`](#classUnityAgentsException) | Contains exceptions specific to ML-Agents.
`struct `[`AcademyParameters`](#structAcademyParameters) | [AcademyParameters](#structAcademyParameters) is a structure containing basic information about the training environment.
`struct `[`CoreBrainPlayer::ContinuousPlayerAction`](#structCoreBrainPlayer_1_1ContinuousPlayerAction) | 
`struct `[`CoreBrainPlayer::DiscretePlayerAction`](#structCoreBrainPlayer_1_1DiscretePlayerAction) | 
`struct `[`Monitor::DisplayValue`](#structMonitor_1_1DisplayValue) | 
`struct `[`Academy::ResetParameter`](#structAcademy_1_1ResetParameter) | 
`struct `[`resolution`](#structresolution) | Only need to be modified in the brain's inpector. Defines what is the resolution of the camera
`struct `[`CoreBrainInternal::TensorFlowAgentPlaceholder`](#structCoreBrainInternal_1_1TensorFlowAgentPlaceholder) | 

## Members

#### `enum `[`BrainType`](#Brain_8cs_1a255eedae6b63fb43e45485914ff30288) 

 Values                         | Descriptions                                
--------------------------------|---------------------------------------------
Player            | 
Heuristic            | 
External            | 

#### `enum `[`StateType`](#Brain_8cs_1a1615968a92950438f6e67a28e9d56e5c) 

 Values                         | Descriptions                                
--------------------------------|---------------------------------------------
discrete            | 
continuous            | 

#### `enum `[`ExternalCommand`](#Communicator_8cs_1a2056e4a11471702052b3ee067b38e355) 

 Values                         | Descriptions                                
--------------------------------|---------------------------------------------
STEP            | 
RESET            | 
QUIT            | 

#### `enum `[`MonitorType`](#Monitor_8cs_1ac69ce5a28782fdc01f7ba5c236cd5f72) 

 Values                         | Descriptions                                
--------------------------------|---------------------------------------------
slider            | 
hist            | 
text            | 
bar            | 

The type of monitor the information must be displayed in. <slider> corresponds to a slingle rectangle which width is given by a float between -1 and 1. (green is positive, red is negative) <hist> corresponds to n vertical sliders. <text> is a text field. <bar> is a rectangle of fixed length to represent the proportions of a list of floats.

# class `Academy` 

```
class Academy
  : public MonoBehaviour
```  

Create a child class to implement [InitializeAcademy()](#classAcademy_1ab6a884f7a70c4dce4432077d716e886c), [AcademyStep()](#classAcademy_1aec20943228af90888c11a796b9e27777) and [AcademyReset()](#classAcademy_1a0872c23a338aebf18b22ce40d3f72c39). The child class script must be attached to an empty game object in your scene, and there can only be one such object within the scene.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public bool `[`isInference`](#classAcademy_1a18e0a2416caec639df9dd58b044e4811) | Do not modify : If true, the [Academy](#classAcademy) will use inference settings.
`public Dictionary< string, float > `[`resetParameters`](#classAcademy_1a1f320e419c5cdf7a00008b742edaba21) | Contains a mapping from parameter names to float values.
`public bool `[`done`](#classAcademy_1a1d39aac66e12dae50a24cd7a9100ef33) | The done flag of the [Academy](#classAcademy).
`public int `[`episodeCount`](#classAcademy_1a8a4ebf2a02144b615981f8e17d386856) | Increments each time the environment is reset.
`public int `[`currentStep`](#classAcademy_1a61a56b55f026894815c3a945bd459a93) | Increments each time a step is taken in the environment. Is reset to 0 during [AcademyReset()](#classAcademy_1a0872c23a338aebf18b22ce40d3f72c39).
`public `[`Communicator`](#interfaceCommunicator)` `[`communicator`](#classAcademy_1a1caff0851123d68f1bdf3083bef7b00b) | Do not modify : pointer to the communicator currently in use by the [Academy](#classAcademy).
`public inline virtual void `[`InitializeAcademy`](#classAcademy_1ab6a884f7a70c4dce4432077d716e886c)`()` | Environment specific initialization.
`public inline virtual void `[`AcademyStep`](#classAcademy_1aec20943228af90888c11a796b9e27777)`()` | Environment specific step logic.
`public inline virtual void `[`AcademyReset`](#classAcademy_1a0872c23a338aebf18b22ce40d3f72c39)`()` | Environment specific reset logic.

## Members

#### `public bool `[`isInference`](#classAcademy_1a18e0a2416caec639df9dd58b044e4811) 

Do not modify : If true, the [Academy](#classAcademy) will use inference settings.

#### `public Dictionary< string, float > `[`resetParameters`](#classAcademy_1a1f320e419c5cdf7a00008b742edaba21) 

Contains a mapping from parameter names to float values.

You can specify the Default Reset Parameters in the Inspector of the [Academy](#classAcademy). You can modify these parameters when training with an External brain by passing a config dictionary at reset. Reference resetParameters in your [AcademyReset()](#classAcademy_1a0872c23a338aebf18b22ce40d3f72c39) or [AcademyStep()](#classAcademy_1aec20943228af90888c11a796b9e27777) to modify elements in your environment at reset time.

#### `public bool `[`done`](#classAcademy_1a1d39aac66e12dae50a24cd7a9100ef33) 

The done flag of the [Academy](#classAcademy).

When set to true, the [Academy](#classAcademy) will call [AcademyReset()](#classAcademy_1a0872c23a338aebf18b22ce40d3f72c39) instead of [AcademyStep()](#classAcademy_1aec20943228af90888c11a796b9e27777) at step time. If true, all agents done flags will be set to true.

#### `public int `[`episodeCount`](#classAcademy_1a8a4ebf2a02144b615981f8e17d386856) 

Increments each time the environment is reset.

#### `public int `[`currentStep`](#classAcademy_1a61a56b55f026894815c3a945bd459a93) 

Increments each time a step is taken in the environment. Is reset to 0 during [AcademyReset()](#classAcademy_1a0872c23a338aebf18b22ce40d3f72c39).

#### `public `[`Communicator`](#interfaceCommunicator)` `[`communicator`](#classAcademy_1a1caff0851123d68f1bdf3083bef7b00b) 

Do not modify : pointer to the communicator currently in use by the [Academy](#classAcademy).

#### `public inline virtual void `[`InitializeAcademy`](#classAcademy_1ab6a884f7a70c4dce4432077d716e886c)`()` 

Environment specific initialization.

Implemented in environment-specific child class. This method is called once when the environment is loaded.

#### `public inline virtual void `[`AcademyStep`](#classAcademy_1aec20943228af90888c11a796b9e27777)`()` 

Environment specific step logic.

Implemented in environment-specific child class. This method is called at every step.

#### `public inline virtual void `[`AcademyReset`](#classAcademy_1a0872c23a338aebf18b22ce40d3f72c39)`()` 

Environment specific reset logic.

Implemented in environment-specific child class. This method is called everytime the [Academy](#classAcademy) resets (when the global done flag is set to true).

# class `Agent` 

```
class Agent
  : public MonoBehaviour
```  

Generic functions for parent [Agent](#classAgent) class. Contains all logic for Brain-Agent communication and Agent-Environment interaction.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public `[`Brain`](#classBrain)` `[`brain`](#classAgent_1a4b858e0e212cea18f48976438a427ee9) | The brain that will control this agent.
`public List< Camera > `[`observations`](#classAgent_1a24856c17a4490e50f785122294fb1f28) | The list of the cameras the [Agent](#classAgent) uses as observations.
`public int `[`maxStep`](#classAgent_1a68defcf610c5210aa9a2b73fb4de6fbe) | The number of steps the agent takes before being done.
`public bool `[`resetOnDone`](#classAgent_1a1e06d6fe173ee1f37c3b7a9af8050195) | Determines the behaviour of the [Agent](#classAgent) when done.
`public float `[`reward`](#classAgent_1ab18e03604d8452d6d86f55873e19732a) | Describes the reward for the given step of the agent.
`public bool `[`done`](#classAgent_1a1d39aac66e12dae50a24cd7a9100ef33) | Whether or not the agent is done.
`public float `[`value`](#classAgent_1a17956fe0129d3d4c94ebc06cfef2ad82) | The current value estimate of the agent.
`public float `[`CumulativeReward`](#classAgent_1ad3630c1282a126eff197831adc99d5da) | Do not modify: This keeps track of the cumulative reward.
`public int `[`stepCounter`](#classAgent_1a58b9fe0d8bdc4ddc63eb62dcd3413b10) | Do not modify: This keeps track of the number of steps taken by the agent each episode.
`public float [] `[`agentStoredAction`](#classAgent_1a119acc1ae838d33b732153ef331e192f) | Do not modify: This keeps track of the last actions decided by the brain.
`public float [] `[`memory`](#classAgent_1ae330b283cc0ba651a3de35c80b181f9d) | Do not modify directly: This is used by the brain to store information about the previous states of the agent.
`public int `[`id`](#classAgent_1a7441ef0865bcb3db9b8064dd7375c1ea) | Do not modify : This is the unique Identifier each agent receives at initialization. It is used by the brain to identify the agent.
`public inline void `[`GiveBrain`](#classAgent_1a88d3a6e3aafb5d7bc127c3a208b31e2d)`(`[`Brain`](#classBrain)` b)` | When GiveBrain is called, the agent unsubscribes from its previous brain and subscribes to the one passed in argument. Use this method to provide a brain to the agent via script. Do not modify brain directly. 
`public inline void `[`RemoveBrain`](#classAgent_1a828f2f5311370d683d225c85533c009a)`()` | When RemoveBrain is called, the agent unsubscribes from its brain.
`public inline virtual void `[`InitializeAgent`](#classAgent_1a0d65cb2bf6fd9e49d87468583db3baa1)`()` | Initialize the agent with this method.
`public inline virtual List< float > `[`CollectState`](#classAgent_1aa12eea45ac8b5c6ff8cb5528c99ddb12)`()` | Collect the states of the agent with this method.
`public inline virtual void `[`AgentStep`](#classAgent_1a3b9ac9f7b99d090f9ec4a77c7080442f)`(float [] action)` | Defines agent-specific behavior at every step depending on the action.
`public inline virtual void `[`AgentOnDone`](#classAgent_1a1744571e645b75c4e15b11c1020199a3)`()` | Defines agent-specific behaviour when done.
`public inline virtual void `[`AgentReset`](#classAgent_1acbb8642634d325629d4e45c8b319610e)`()` | Defines agent-specific reset logic.
`public inline void `[`Reset`](#classAgent_1a372de693ad40b3f42839c8ec6ac845f4)`()` | Do not modify : Is used by the brain to reset the agent.
`public inline float `[`CollectReward`](#classAgent_1a44b938d1435797d71d09728ccd24b20c)`()` | Do not modify : Is used by the brain to collect rewards.
`public inline void `[`SetCumulativeReward`](#classAgent_1ae0a595afe26558461c02430791731cb0)`()` | 
`public inline bool `[`CollectDone`](#classAgent_1a43fc0419767aafbcb336102a983bf8f6)`()` | Do not modify : Is used by the brain to collect done.
`public inline void `[`UpdateAction`](#classAgent_1a13d484f047228c4d7d2ff8060739baea)`(float [] a)` | Do not modify : Is used by the brain give new action to the agent.
`public inline void `[`Step`](#classAgent_1a1020ec8c5c31463c0b7e7f2101902e3a)`()` | Do not modify : Is used by the brain to make the agent perform a step.
`public inline void `[`ResetReward`](#classAgent_1a2ebb156dac7038fe138c910df69300c9)`()` | Do not modify : Is used by the brain to reset the Reward.

## Members

#### `public `[`Brain`](#classBrain)` `[`brain`](#classAgent_1a4b858e0e212cea18f48976438a427ee9) 

The brain that will control this agent.

Use the inspector to drag the desired brain gameObject into the [Brain](#classBrain) field

#### `public List< Camera > `[`observations`](#classAgent_1a24856c17a4490e50f785122294fb1f28) 

The list of the cameras the [Agent](#classAgent) uses as observations.

These cameras will be used to generate the observations

#### `public int `[`maxStep`](#classAgent_1a68defcf610c5210aa9a2b73fb4de6fbe) 

The number of steps the agent takes before being done.

If set to 0, the agent can only be set to done via a script. If set to any positive integer, the agent will be set to done after that many steps each episode.

#### `public bool `[`resetOnDone`](#classAgent_1a1e06d6fe173ee1f37c3b7a9af8050195) 

Determines the behaviour of the [Agent](#classAgent) when done.

If true, the agent will reset when done. If not, the agent will remain done, and no longer take actions.

#### `public float `[`reward`](#classAgent_1ab18e03604d8452d6d86f55873e19732a) 

Describes the reward for the given step of the agent.

It is reset to 0 at the beginning of every step. Modify in [AgentStep()](#classAgent_1a3b9ac9f7b99d090f9ec4a77c7080442f). Should be set to positive to reinforcement desired behavior, and set to a negative value to punish undesireable behavior. Additionally, the magnitude of the reward should not exceed 1.0

#### `public bool `[`done`](#classAgent_1a1d39aac66e12dae50a24cd7a9100ef33) 

Whether or not the agent is done.

Set to true when the agent has acted in some way which ends the episode for the given agent.

#### `public float `[`value`](#classAgent_1a17956fe0129d3d4c94ebc06cfef2ad82) 

The current value estimate of the agent.

When using an External brain, you can pass value estimates to the agent at every step using env.Step(actions, values). If AgentMonitor is attached to the [Agent](#classAgent), this value will be displayed.

#### `public float `[`CumulativeReward`](#classAgent_1ad3630c1282a126eff197831adc99d5da) 

Do not modify: This keeps track of the cumulative reward.

#### `public int `[`stepCounter`](#classAgent_1a58b9fe0d8bdc4ddc63eb62dcd3413b10) 

Do not modify: This keeps track of the number of steps taken by the agent each episode.

#### `public float [] `[`agentStoredAction`](#classAgent_1a119acc1ae838d33b732153ef331e192f) 

Do not modify: This keeps track of the last actions decided by the brain.

#### `public float [] `[`memory`](#classAgent_1ae330b283cc0ba651a3de35c80b181f9d) 

Do not modify directly: This is used by the brain to store information about the previous states of the agent.

#### `public int `[`id`](#classAgent_1a7441ef0865bcb3db9b8064dd7375c1ea) 

Do not modify : This is the unique Identifier each agent receives at initialization. It is used by the brain to identify the agent.

#### `public inline void `[`GiveBrain`](#classAgent_1a88d3a6e3aafb5d7bc127c3a208b31e2d)`(`[`Brain`](#classBrain)` b)` 

When GiveBrain is called, the agent unsubscribes from its previous brain and subscribes to the one passed in argument. Use this method to provide a brain to the agent via script. Do not modify brain directly. 
#### Parameters
* `b` The [Brain](#classBrain) component the agent will subscribe to.

#### `public inline void `[`RemoveBrain`](#classAgent_1a828f2f5311370d683d225c85533c009a)`()` 

When RemoveBrain is called, the agent unsubscribes from its brain.

Use this method to give a brain to an agent via script. Do not modify brain directly. If an agent does not have a brain, it will not update its actions.

#### `public inline virtual void `[`InitializeAgent`](#classAgent_1a0d65cb2bf6fd9e49d87468583db3baa1)`()` 

Initialize the agent with this method.

Must be implemented in agent-specific child class. This method called only once when the agent is enabled.

#### `public inline virtual List< float > `[`CollectState`](#classAgent_1aa12eea45ac8b5c6ff8cb5528c99ddb12)`()` 

Collect the states of the agent with this method.

Must be implemented in agent-specific child class. This method called at every step and collects the state of the agent. The lenght of the output must be the same length as the state size field in the brain parameters of the brain the agent subscribes to. Note : The order of the elements in the state list is important. 
#### Returns
state A list of floats corresponding to the state of the agent.

#### `public inline virtual void `[`AgentStep`](#classAgent_1a3b9ac9f7b99d090f9ec4a77c7080442f)`(float [] action)` 

Defines agent-specific behavior at every step depending on the action.

Must be implemented in agent-specific child class. Note: If your state is discrete, you need to convert your state into a list of float with length 1. 
#### Parameters
* `action` The action the agent receives from the brain.

#### `public inline virtual void `[`AgentOnDone`](#classAgent_1a1744571e645b75c4e15b11c1020199a3)`()` 

Defines agent-specific behaviour when done.

Must be implemented in agent-specific child class. Is called when the [Agent](#classAgent) is done if ResetOneDone is false. The agent will remain done. You can use this method to remove the agent from the scene.

#### `public inline virtual void `[`AgentReset`](#classAgent_1acbb8642634d325629d4e45c8b319610e)`()` 

Defines agent-specific reset logic.

Must be implemented in agent-specific child class. Is called when the academy is done. Is called when the [Agent](#classAgent) is done if ResetOneDone is true.

#### `public inline void `[`Reset`](#classAgent_1a372de693ad40b3f42839c8ec6ac845f4)`()` 

Do not modify : Is used by the brain to reset the agent.

#### `public inline float `[`CollectReward`](#classAgent_1a44b938d1435797d71d09728ccd24b20c)`()` 

Do not modify : Is used by the brain to collect rewards.

#### `public inline void `[`SetCumulativeReward`](#classAgent_1ae0a595afe26558461c02430791731cb0)`()` 

#### `public inline bool `[`CollectDone`](#classAgent_1a43fc0419767aafbcb336102a983bf8f6)`()` 

Do not modify : Is used by the brain to collect done.

#### `public inline void `[`UpdateAction`](#classAgent_1a13d484f047228c4d7d2ff8060739baea)`(float [] a)` 

Do not modify : Is used by the brain give new action to the agent.

#### `public inline void `[`Step`](#classAgent_1a1020ec8c5c31463c0b7e7f2101902e3a)`()` 

Do not modify : Is used by the brain to make the agent perform a step.

#### `public inline void `[`ResetReward`](#classAgent_1a2ebb156dac7038fe138c910df69300c9)`()` 

Do not modify : Is used by the brain to reset the Reward.

# class `ExternalCommunicator::AgentMessage` 

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------

## Members

# class `Brain` 

```
class Brain
  : public MonoBehaviour
```  

Contains all high-level [Brain](#classBrain) logic. Add this component to an empty GameObject in your scene and drag this GameObject into your [Academy](#classAcademy) to make it a child in the hierarchy. Contains a set of CoreBrains, which each correspond to a different method for deciding actions.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public `[`BrainParameters`](#classBrainParameters)` `[`brainParameters`](#classBrain_1af7cfbe29c1b7ba9a4526f2856bc6ee92) | Defines brain specific parameters such as the state size.
`public `[`BrainType`](#Brain_8cs_1a255eedae6b63fb43e45485914ff30288)` `[`brainType`](#classBrain_1aa1ed80b695a6e1aefe60dd60c83cb98a) | Defines what is the type of the brain : External / Internal / Player / Heuristic.
`public Dictionary< int, `[`Agent`](#classAgent)` > `[`agents`](#classBrain_1aa311c15de5dc5443ca2e30c1763ec5d3) | Keeps track of the agents which subscribe to this brain.
`public `[`CoreBrain`](#interfaceCoreBrain)` `[`coreBrain`](#classBrain_1a5210e4ac17ce2a40d14b7fdbd34bb622) | Reference to the current [CoreBrain](#interfaceCoreBrain) used by the brain.
`public inline void `[`UpdateCoreBrains`](#classBrain_1acb5334497f21afed62517071bad8ce3c)`()` | Ensures the brain has an up to date array of coreBrains.
`public inline void `[`InitializeBrain`](#classBrain_1aeea8b88c1be97eb6264264c511da5f70)`()` | This is called by the [Academy](#classAcademy) at the start of the environemnt.
`public inline Dictionary< int, List< float > > `[`CollectStates`](#classBrain_1a6c1ca613b7a3fd791e5e4470de119717)`()` | Collects the states of all the agents which subscribe to this brain and returns a dictionary {id -> state}
`public inline Dictionary< int, List< Camera > > `[`CollectObservations`](#classBrain_1a8a902498767de86643c120c10fd068b2)`()` | Collects the observations of all the agents which subscribe to this brain and returns a dictionary {id -> Camera}
`public inline Dictionary< int, float > `[`CollectRewards`](#classBrain_1a4ea4a8cd9ed6aade194536985078661e)`()` | Collects the rewards of all the agents which subscribe to this brain and returns a dictionary {id -> reward}
`public inline Dictionary< int, bool > `[`CollectDones`](#classBrain_1af24fb4ac478f9ee82c08a0ff6d329359)`()` | Collects the done flag of all the agents which subscribe to this brain and returns a dictionary {id -> done}
`public inline Dictionary< int, float[]> `[`CollectActions`](#classBrain_1a145f1a881758d4d8b8964149c587f099)`()` | Collects the actions of all the agents which subscribe to this brain and returns a dictionary {id -> action}
`public inline Dictionary< int, float[]> `[`CollectMemories`](#classBrain_1a3af30140d5c88d865363d926e698086c)`()` | Collects the memories of all the agents which subscribe to this brain and returns a dictionary {id -> memories}
`public inline void `[`SendMemories`](#classBrain_1af1b6f1a61af3c7a5515f476aa6f81be9)`(Dictionary< int, float[]> memories)` | Takes a dictionary {id -> memories} and sends the memories to the corresponding agents
`public inline void `[`SendActions`](#classBrain_1a27e58afa69122e13c3ab71d59c3dfe15)`(Dictionary< int, float[]> actions)` | Takes a dictionary {id -> actions} and sends the actions to the corresponding agents
`public inline void `[`SendValues`](#classBrain_1aa8b410b58574932c058ba4cee54912cb)`(Dictionary< int, float > values)` | Takes a dictionary {id -> values} and sends the values to the corresponding agents
`public inline void `[`SendDone`](#classBrain_1ad63b07252e5eafc009449f00ea50bd3c)`()` | Sets all the agents which subscribe to the brain to done.
`public inline void `[`SendState`](#classBrain_1a9889ae5b144e333020db71636f20eeb4)`()` | Uses coreBrain to call SendState on the [CoreBrain](#interfaceCoreBrain).
`public inline void `[`DecideAction`](#classBrain_1a219ee30dceed5e480b7559643919892f)`()` | Uses coreBrain to call decideAction on the [CoreBrain](#interfaceCoreBrain).
`public inline void `[`Step`](#classBrain_1a1020ec8c5c31463c0b7e7f2101902e3a)`()` | Is used by the [Academy](#classAcademy) to send a step message to all the agents which are not done.
`public inline void `[`ResetIfDone`](#classBrain_1a31decddae0e49ae2f778eeca8c175b49)`()` | Is used by the [Academy](#classAcademy) to reset the agents if they are done.
`public inline void `[`Reset`](#classBrain_1a372de693ad40b3f42839c8ec6ac845f4)`()` | Is used by the [Academy](#classAcademy) to reset all agents.
`public inline void `[`ResetDoneAndReward`](#classBrain_1a9702db49c0590484872659926938dc4b)`()` | Is used by the [Academy](#classAcademy) reset the done flag and the rewards of the agents that subscribe to the brain.
`public inline Texture2D `[`ObservationToTex`](#classBrain_1a739f3ab44ba8c9ad91721e20dcd61893)`(Camera camera,int width,int height)` | Contains logic for coverting a camera component into a Texture2D.
`public inline List< float[,,,]> `[`GetObservationMatrixList`](#classBrain_1a4abc92e27f34e5f3391a59b298a7c03f)`(List< int > agent_keys)` | Contains logic to convert the agent's cameras into observation list (as list of float arrays)

## Members

#### `public `[`BrainParameters`](#classBrainParameters)` `[`brainParameters`](#classBrain_1af7cfbe29c1b7ba9a4526f2856bc6ee92) 

Defines brain specific parameters such as the state size.

#### `public `[`BrainType`](#Brain_8cs_1a255eedae6b63fb43e45485914ff30288)` `[`brainType`](#classBrain_1aa1ed80b695a6e1aefe60dd60c83cb98a) 

Defines what is the type of the brain : External / Internal / Player / Heuristic.

#### `public Dictionary< int, `[`Agent`](#classAgent)` > `[`agents`](#classBrain_1aa311c15de5dc5443ca2e30c1763ec5d3) 

Keeps track of the agents which subscribe to this brain.

#### `public `[`CoreBrain`](#interfaceCoreBrain)` `[`coreBrain`](#classBrain_1a5210e4ac17ce2a40d14b7fdbd34bb622) 

Reference to the current [CoreBrain](#interfaceCoreBrain) used by the brain.

#### `public inline void `[`UpdateCoreBrains`](#classBrain_1acb5334497f21afed62517071bad8ce3c)`()` 

Ensures the brain has an up to date array of coreBrains.

Is called when the inspector is modified and into InitializeBrain. If the brain gameObject was just created, it generates a list of coreBrains (one for each brainType)

#### `public inline void `[`InitializeBrain`](#classBrain_1aeea8b88c1be97eb6264264c511da5f70)`()` 

This is called by the [Academy](#classAcademy) at the start of the environemnt.

#### `public inline Dictionary< int, List< float > > `[`CollectStates`](#classBrain_1a6c1ca613b7a3fd791e5e4470de119717)`()` 

Collects the states of all the agents which subscribe to this brain and returns a dictionary {id -> state}

#### `public inline Dictionary< int, List< Camera > > `[`CollectObservations`](#classBrain_1a8a902498767de86643c120c10fd068b2)`()` 

Collects the observations of all the agents which subscribe to this brain and returns a dictionary {id -> Camera}

#### `public inline Dictionary< int, float > `[`CollectRewards`](#classBrain_1a4ea4a8cd9ed6aade194536985078661e)`()` 

Collects the rewards of all the agents which subscribe to this brain and returns a dictionary {id -> reward}

#### `public inline Dictionary< int, bool > `[`CollectDones`](#classBrain_1af24fb4ac478f9ee82c08a0ff6d329359)`()` 

Collects the done flag of all the agents which subscribe to this brain and returns a dictionary {id -> done}

#### `public inline Dictionary< int, float[]> `[`CollectActions`](#classBrain_1a145f1a881758d4d8b8964149c587f099)`()` 

Collects the actions of all the agents which subscribe to this brain and returns a dictionary {id -> action}

#### `public inline Dictionary< int, float[]> `[`CollectMemories`](#classBrain_1a3af30140d5c88d865363d926e698086c)`()` 

Collects the memories of all the agents which subscribe to this brain and returns a dictionary {id -> memories}

#### `public inline void `[`SendMemories`](#classBrain_1af1b6f1a61af3c7a5515f476aa6f81be9)`(Dictionary< int, float[]> memories)` 

Takes a dictionary {id -> memories} and sends the memories to the corresponding agents

#### `public inline void `[`SendActions`](#classBrain_1a27e58afa69122e13c3ab71d59c3dfe15)`(Dictionary< int, float[]> actions)` 

Takes a dictionary {id -> actions} and sends the actions to the corresponding agents

#### `public inline void `[`SendValues`](#classBrain_1aa8b410b58574932c058ba4cee54912cb)`(Dictionary< int, float > values)` 

Takes a dictionary {id -> values} and sends the values to the corresponding agents

#### `public inline void `[`SendDone`](#classBrain_1ad63b07252e5eafc009449f00ea50bd3c)`()` 

Sets all the agents which subscribe to the brain to done.

#### `public inline void `[`SendState`](#classBrain_1a9889ae5b144e333020db71636f20eeb4)`()` 

Uses coreBrain to call SendState on the [CoreBrain](#interfaceCoreBrain).

#### `public inline void `[`DecideAction`](#classBrain_1a219ee30dceed5e480b7559643919892f)`()` 

Uses coreBrain to call decideAction on the [CoreBrain](#interfaceCoreBrain).

#### `public inline void `[`Step`](#classBrain_1a1020ec8c5c31463c0b7e7f2101902e3a)`()` 

Is used by the [Academy](#classAcademy) to send a step message to all the agents which are not done.

#### `public inline void `[`ResetIfDone`](#classBrain_1a31decddae0e49ae2f778eeca8c175b49)`()` 

Is used by the [Academy](#classAcademy) to reset the agents if they are done.

#### `public inline void `[`Reset`](#classBrain_1a372de693ad40b3f42839c8ec6ac845f4)`()` 

Is used by the [Academy](#classAcademy) to reset all agents.

#### `public inline void `[`ResetDoneAndReward`](#classBrain_1a9702db49c0590484872659926938dc4b)`()` 

Is used by the [Academy](#classAcademy) reset the done flag and the rewards of the agents that subscribe to the brain.

#### `public inline Texture2D `[`ObservationToTex`](#classBrain_1a739f3ab44ba8c9ad91721e20dcd61893)`(Camera camera,int width,int height)` 

Contains logic for coverting a camera component into a Texture2D.

#### `public inline List< float[,,,]> `[`GetObservationMatrixList`](#classBrain_1a4abc92e27f34e5f3391a59b298a7c03f)`(List< int > agent_keys)` 

Contains logic to convert the agent's cameras into observation list (as list of float arrays)

# class `BrainParameters` 

Should be modified via the Editor Inspector. Defines brain-specific parameters

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public int `[`stateSize`](#classBrainParameters_1aa06a68068b689cff9f919a2c299c9a89) | If continuous : The length of the float vector that represents the state <br/> If discrete : The number of possible values the state can take.
`public int `[`actionSize`](#classBrainParameters_1a7beffb549bba1786053f64c4bd947553) | If continuous : The length of the float vector that represents the action <br/> If discrete : The number of possible values the action can take.
`public int `[`memorySize`](#classBrainParameters_1ae63baadad611e205e7e5e486a1e6f769) | The length of the float vector that holds the memory for the agent.
`public `[`resolution`](#structresolution)` [] `[`cameraResolutions`](#classBrainParameters_1a0c28c04d3172f0ec2f56c75ad02b8a1c) | The list of observation resolutions for the brain.
`public string [] `[`actionDescriptions`](#classBrainParameters_1a3d35713b59c63910426cbac64c025a95) | The list of strings describing what the actions correpond to.
`public `[`StateType`](#Brain_8cs_1a1615968a92950438f6e67a28e9d56e5c)` `[`actionSpaceType`](#classBrainParameters_1a7d22a1f4228f7536f0a662be5b95ad4c) | Defines if the action is discrete or continuous.
`public `[`StateType`](#Brain_8cs_1a1615968a92950438f6e67a28e9d56e5c)` `[`stateSpaceType`](#classBrainParameters_1a976711473e5aac7e916def617390a04c) | Defines if the state is discrete or continuous.

## Members

#### `public int `[`stateSize`](#classBrainParameters_1aa06a68068b689cff9f919a2c299c9a89) 

If continuous : The length of the float vector that represents the state 
 If discrete : The number of possible values the state can take.

#### `public int `[`actionSize`](#classBrainParameters_1a7beffb549bba1786053f64c4bd947553) 

If continuous : The length of the float vector that represents the action 
 If discrete : The number of possible values the action can take.

#### `public int `[`memorySize`](#classBrainParameters_1ae63baadad611e205e7e5e486a1e6f769) 

The length of the float vector that holds the memory for the agent.

#### `public `[`resolution`](#structresolution)` [] `[`cameraResolutions`](#classBrainParameters_1a0c28c04d3172f0ec2f56c75ad02b8a1c) 

The list of observation resolutions for the brain.

#### `public string [] `[`actionDescriptions`](#classBrainParameters_1a3d35713b59c63910426cbac64c025a95) 

The list of strings describing what the actions correpond to.

#### `public `[`StateType`](#Brain_8cs_1a1615968a92950438f6e67a28e9d56e5c)` `[`actionSpaceType`](#classBrainParameters_1a7d22a1f4228f7536f0a662be5b95ad4c) 

Defines if the action is discrete or continuous.

#### `public `[`StateType`](#Brain_8cs_1a1615968a92950438f6e67a28e9d56e5c)` `[`stateSpaceType`](#classBrainParameters_1a976711473e5aac7e916def617390a04c) 

Defines if the state is discrete or continuous.

# class `CoreBrainExternal` 

```
class CoreBrainExternal
  : public ScriptableObject
  : public CoreBrain
```  

[CoreBrain](#interfaceCoreBrain) which decides actions via communication with an external system such as Python.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public `[`Brain`](#classBrain)` `[`brain`](#classCoreBrainExternal_1a4b858e0e212cea18f48976438a427ee9) | Reference to the brain that uses this [CoreBrainExternal](#classCoreBrainExternal)
`public inline void `[`SetBrain`](#classCoreBrainExternal_1a31c47e67f914756f0781eb58060315e2)`(`[`Brain`](#classBrain)` b)` | Creates the reference to the brain.
`public inline void `[`InitializeCoreBrain`](#classCoreBrainExternal_1aa5486bda9e98ad6105f7bdc3566c92f0)`()` | Generates the communicator for the [Academy](#classAcademy) if none was present and subscribe to [ExternalCommunicator](#classExternalCommunicator) if it was present.
`public inline void `[`DecideAction`](#classCoreBrainExternal_1a219ee30dceed5e480b7559643919892f)`()` | Uses the communicator to retrieve the actions, memories and values and sends them to the agents
`public inline void `[`SendState`](#classCoreBrainExternal_1a9889ae5b144e333020db71636f20eeb4)`()` | Uses the communicator to send the states, observations, rewards and dones outside of Unity
`public inline void `[`OnInspector`](#classCoreBrainExternal_1a72e8439d02e4fac663b4f9edc40944a8)`()` | Nothing needs to appear in the inspector.

## Members

#### `public `[`Brain`](#classBrain)` `[`brain`](#classCoreBrainExternal_1a4b858e0e212cea18f48976438a427ee9) 

Reference to the brain that uses this [CoreBrainExternal](#classCoreBrainExternal)

#### `public inline void `[`SetBrain`](#classCoreBrainExternal_1a31c47e67f914756f0781eb58060315e2)`(`[`Brain`](#classBrain)` b)` 

Creates the reference to the brain.

#### `public inline void `[`InitializeCoreBrain`](#classCoreBrainExternal_1aa5486bda9e98ad6105f7bdc3566c92f0)`()` 

Generates the communicator for the [Academy](#classAcademy) if none was present and subscribe to [ExternalCommunicator](#classExternalCommunicator) if it was present.

#### `public inline void `[`DecideAction`](#classCoreBrainExternal_1a219ee30dceed5e480b7559643919892f)`()` 

Uses the communicator to retrieve the actions, memories and values and sends them to the agents

#### `public inline void `[`SendState`](#classCoreBrainExternal_1a9889ae5b144e333020db71636f20eeb4)`()` 

Uses the communicator to send the states, observations, rewards and dones outside of Unity

#### `public inline void `[`OnInspector`](#classCoreBrainExternal_1a72e8439d02e4fac663b4f9edc40944a8)`()` 

Nothing needs to appear in the inspector.

# class `CoreBrainHeuristic` 

```
class CoreBrainHeuristic
  : public ScriptableObject
  : public CoreBrain
```  

[CoreBrain](#interfaceCoreBrain) which decides actions using developer-provided [Decision.cs](#Decision_8cs) script.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public `[`Brain`](#classBrain)` `[`brain`](#classCoreBrainHeuristic_1a4b858e0e212cea18f48976438a427ee9) | Reference to the brain that uses this [CoreBrainHeuristic](#classCoreBrainHeuristic)
`public `[`Decision`](#interfaceDecision)` `[`decision`](#classCoreBrainHeuristic_1a99d77d9dc4b9f80f014f532c3800672a) | Reference to the [Decision](#interfaceDecision) component used to decide the actions
`public inline void `[`SetBrain`](#classCoreBrainHeuristic_1a31c47e67f914756f0781eb58060315e2)`(`[`Brain`](#classBrain)` b)` | Create the reference to the brain.
`public inline void `[`InitializeCoreBrain`](#classCoreBrainHeuristic_1aa5486bda9e98ad6105f7bdc3566c92f0)`()` | Create the reference to decision.
`public inline void `[`DecideAction`](#classCoreBrainHeuristic_1a219ee30dceed5e480b7559643919892f)`()` | Uses the [Decision](#interfaceDecision) Component to decide that action to take.
`public inline void `[`SendState`](#classCoreBrainHeuristic_1a9889ae5b144e333020db71636f20eeb4)`()` | Nothing needs to be implemented, the states are collected in DecideAction.
`public inline void `[`OnInspector`](#classCoreBrainHeuristic_1a72e8439d02e4fac663b4f9edc40944a8)`()` | Displays an error if no decision component is attached to the brain.

## Members

#### `public `[`Brain`](#classBrain)` `[`brain`](#classCoreBrainHeuristic_1a4b858e0e212cea18f48976438a427ee9) 

Reference to the brain that uses this [CoreBrainHeuristic](#classCoreBrainHeuristic)

#### `public `[`Decision`](#interfaceDecision)` `[`decision`](#classCoreBrainHeuristic_1a99d77d9dc4b9f80f014f532c3800672a) 

Reference to the [Decision](#interfaceDecision) component used to decide the actions

#### `public inline void `[`SetBrain`](#classCoreBrainHeuristic_1a31c47e67f914756f0781eb58060315e2)`(`[`Brain`](#classBrain)` b)` 

Create the reference to the brain.

#### `public inline void `[`InitializeCoreBrain`](#classCoreBrainHeuristic_1aa5486bda9e98ad6105f7bdc3566c92f0)`()` 

Create the reference to decision.

#### `public inline void `[`DecideAction`](#classCoreBrainHeuristic_1a219ee30dceed5e480b7559643919892f)`()` 

Uses the [Decision](#interfaceDecision) Component to decide that action to take.

#### `public inline void `[`SendState`](#classCoreBrainHeuristic_1a9889ae5b144e333020db71636f20eeb4)`()` 

Nothing needs to be implemented, the states are collected in DecideAction.

#### `public inline void `[`OnInspector`](#classCoreBrainHeuristic_1a72e8439d02e4fac663b4f9edc40944a8)`()` 

Displays an error if no decision component is attached to the brain.

# class `CoreBrainInternal` 

```
class CoreBrainInternal
  : public ScriptableObject
  : public CoreBrain
```  

[CoreBrain](#interfaceCoreBrain) which decides actions using internally embedded TensorFlow model.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public TextAsset `[`graphModel`](#classCoreBrainInternal_1a53f57ce3a53a813063aa7e5adf69cb49) | Modify only in inspector : Reference to the Graph asset.
`public string `[`graphScope`](#classCoreBrainInternal_1a5761498f62f88f7495b27e5cca77c1db) | Modify only in inspector : If a scope was used when training the model, specify it here.
`public string `[`BatchSizePlaceholderName`](#classCoreBrainInternal_1a2cf7f1107295d08ba553d9ccc14f31b5) | Modify only in inspector : Name of the placholder of the batch size.
`public string `[`StatePlacholderName`](#classCoreBrainInternal_1a7a20e46e2329198578630f5bdaeabe0b) | Modify only in inspector : Name of the state placeholder.
`public string `[`RecurrentInPlaceholderName`](#classCoreBrainInternal_1ad0d404209bedd5a318421fa93ae95123) | Modify only in inspector : Name of the recurrent input.
`public string `[`RecurrentOutPlaceholderName`](#classCoreBrainInternal_1abdc7036baa9f29f796e079db898a2fef) | Modify only in inspector : Name of the recurrent output.
`public string [] `[`ObservationPlaceholderName`](#classCoreBrainInternal_1a582aa735e711257f6acd590a33bf3897) | Modify only in inspector : Names of the observations placeholders.
`public string `[`ActionPlaceholderName`](#classCoreBrainInternal_1ace39b6aa4184e39ace2704363800b730) | Modify only in inspector : Name of the action node.
`public `[`Brain`](#classBrain)` `[`brain`](#classCoreBrainInternal_1a4b858e0e212cea18f48976438a427ee9) | Reference to the brain that uses this [CoreBrainInternal](#classCoreBrainInternal).
`public inline void `[`SetBrain`](#classCoreBrainInternal_1a31c47e67f914756f0781eb58060315e2)`(`[`Brain`](#classBrain)` b)` | Create the reference to the brain.
`public inline void `[`InitializeCoreBrain`](#classCoreBrainInternal_1aa5486bda9e98ad6105f7bdc3566c92f0)`()` | Loads the tensorflow graph model to generate a TFGraph object.
`public inline void `[`SendState`](#classCoreBrainInternal_1a9889ae5b144e333020db71636f20eeb4)`()` | Collects information from the agents and store them.
`public inline void `[`DecideAction`](#classCoreBrainInternal_1a219ee30dceed5e480b7559643919892f)`()` | Uses the stored information to run the tensorflow graph and generate the actions.
`public inline void `[`OnInspector`](#classCoreBrainInternal_1a72e8439d02e4fac663b4f9edc40944a8)`()` | Displays the parameters of the [CoreBrainInternal](#classCoreBrainInternal) in the Inspector.

## Members

#### `public TextAsset `[`graphModel`](#classCoreBrainInternal_1a53f57ce3a53a813063aa7e5adf69cb49) 

Modify only in inspector : Reference to the Graph asset.

#### `public string `[`graphScope`](#classCoreBrainInternal_1a5761498f62f88f7495b27e5cca77c1db) 

Modify only in inspector : If a scope was used when training the model, specify it here.

#### `public string `[`BatchSizePlaceholderName`](#classCoreBrainInternal_1a2cf7f1107295d08ba553d9ccc14f31b5) 

Modify only in inspector : Name of the placholder of the batch size.

#### `public string `[`StatePlacholderName`](#classCoreBrainInternal_1a7a20e46e2329198578630f5bdaeabe0b) 

Modify only in inspector : Name of the state placeholder.

#### `public string `[`RecurrentInPlaceholderName`](#classCoreBrainInternal_1ad0d404209bedd5a318421fa93ae95123) 

Modify only in inspector : Name of the recurrent input.

#### `public string `[`RecurrentOutPlaceholderName`](#classCoreBrainInternal_1abdc7036baa9f29f796e079db898a2fef) 

Modify only in inspector : Name of the recurrent output.

#### `public string [] `[`ObservationPlaceholderName`](#classCoreBrainInternal_1a582aa735e711257f6acd590a33bf3897) 

Modify only in inspector : Names of the observations placeholders.

#### `public string `[`ActionPlaceholderName`](#classCoreBrainInternal_1ace39b6aa4184e39ace2704363800b730) 

Modify only in inspector : Name of the action node.

#### `public `[`Brain`](#classBrain)` `[`brain`](#classCoreBrainInternal_1a4b858e0e212cea18f48976438a427ee9) 

Reference to the brain that uses this [CoreBrainInternal](#classCoreBrainInternal).

#### `public inline void `[`SetBrain`](#classCoreBrainInternal_1a31c47e67f914756f0781eb58060315e2)`(`[`Brain`](#classBrain)` b)` 

Create the reference to the brain.

#### `public inline void `[`InitializeCoreBrain`](#classCoreBrainInternal_1aa5486bda9e98ad6105f7bdc3566c92f0)`()` 

Loads the tensorflow graph model to generate a TFGraph object.

#### `public inline void `[`SendState`](#classCoreBrainInternal_1a9889ae5b144e333020db71636f20eeb4)`()` 

Collects information from the agents and store them.

#### `public inline void `[`DecideAction`](#classCoreBrainInternal_1a219ee30dceed5e480b7559643919892f)`()` 

Uses the stored information to run the tensorflow graph and generate the actions.

#### `public inline void `[`OnInspector`](#classCoreBrainInternal_1a72e8439d02e4fac663b4f9edc40944a8)`()` 

Displays the parameters of the [CoreBrainInternal](#classCoreBrainInternal) in the Inspector.

# class `CoreBrainPlayer` 

```
class CoreBrainPlayer
  : public ScriptableObject
  : public CoreBrain
```  

[CoreBrain](#interfaceCoreBrain) which decides actions using Player input.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public `[`Brain`](#classBrain)` `[`brain`](#classCoreBrainPlayer_1a4b858e0e212cea18f48976438a427ee9) | Reference to the brain that uses this [CoreBrainPlayer](#classCoreBrainPlayer).
`public inline void `[`SetBrain`](#classCoreBrainPlayer_1a31c47e67f914756f0781eb58060315e2)`(`[`Brain`](#classBrain)` b)` | Create the reference to the brain.
`public inline void `[`InitializeCoreBrain`](#classCoreBrainPlayer_1aa5486bda9e98ad6105f7bdc3566c92f0)`()` | Nothing to implement.
`public inline void `[`DecideAction`](#classCoreBrainPlayer_1a219ee30dceed5e480b7559643919892f)`()` | Uses the continuous inputs or dicrete inputs of the player to decide action
`public inline void `[`SendState`](#classCoreBrainPlayer_1a9889ae5b144e333020db71636f20eeb4)`()` | Nothing to implement, the Player does not use the state to make decisions
`public inline void `[`OnInspector`](#classCoreBrainPlayer_1a72e8439d02e4fac663b4f9edc40944a8)`()` | Displays continuous or discrete input mapping in the inspector.

## Members

#### `public `[`Brain`](#classBrain)` `[`brain`](#classCoreBrainPlayer_1a4b858e0e212cea18f48976438a427ee9) 

Reference to the brain that uses this [CoreBrainPlayer](#classCoreBrainPlayer).

#### `public inline void `[`SetBrain`](#classCoreBrainPlayer_1a31c47e67f914756f0781eb58060315e2)`(`[`Brain`](#classBrain)` b)` 

Create the reference to the brain.

#### `public inline void `[`InitializeCoreBrain`](#classCoreBrainPlayer_1aa5486bda9e98ad6105f7bdc3566c92f0)`()` 

Nothing to implement.

#### `public inline void `[`DecideAction`](#classCoreBrainPlayer_1a219ee30dceed5e480b7559643919892f)`()` 

Uses the continuous inputs or dicrete inputs of the player to decide action

#### `public inline void `[`SendState`](#classCoreBrainPlayer_1a9889ae5b144e333020db71636f20eeb4)`()` 

Nothing to implement, the Player does not use the state to make decisions

#### `public inline void `[`OnInspector`](#classCoreBrainPlayer_1a72e8439d02e4fac663b4f9edc40944a8)`()` 

Displays continuous or discrete input mapping in the inspector.

# class `ExternalCommunicator` 

```
class ExternalCommunicator
  : public Communicator
```  

Responsible for communication with Python API.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline  `[`ExternalCommunicator`](#classExternalCommunicator_1a62c140342df7ac93e567d0b9a2589615)`(`[`Academy`](#classAcademy)` aca)` | Consrtuctor for the External [Communicator](#interfaceCommunicator).
`public inline void `[`SubscribeBrain`](#classExternalCommunicator_1af83f0ed2244661ef60044f36f2066ec6)`(`[`Brain`](#classBrain)` brain)` | Adds the brain to the list of brains which have already decided their actions.
`public inline bool `[`CommunicatorHandShake`](#classExternalCommunicator_1a7eb674f0acbe274e4933df5aea101f72)`()` | First contact between [Communicator](#interfaceCommunicator) and external process.
`public inline void `[`InitializeCommunicator`](#classExternalCommunicator_1a4b92ce2971224e0277e759e1d452c4ea)`()` | Contains the logic for the initializtation of the socket.
`public inline `[`ExternalCommand`](#Communicator_8cs_1a2056e4a11471702052b3ee067b38e355)` `[`GetCommand`](#classExternalCommunicator_1a1074576c637cf6309c12c16923398261)`()` | Listens to the socket for a command and returns the corresponding External Command.
`public inline Dictionary< string, float > `[`GetResetParameters`](#classExternalCommunicator_1a77da9de880e3bfc14125162d4c3adedc)`()` | Listens to the socket for the new resetParameters.
`public inline void `[`giveBrainInfo`](#classExternalCommunicator_1a0f500c2709fca846e0cee3069eb4a335)`(`[`Brain`](#classBrain)` brain)` | Collects the information from the brains and sends it accross the socket.
`public inline void `[`UpdateActions`](#classExternalCommunicator_1a45bd50d0aa1084d7a895bea7e2f6314c)`()` | Listens for actions, memories, and values and sends them to the corrensponding brains.
`public inline Dictionary< int, float[]> `[`GetDecidedAction`](#classExternalCommunicator_1acd5ab12458fc39be0ed706cf3508e10d)`(string brainName)` | Returns the actions corrensponding to the brain called brainName that were received throught the socket.
`public inline Dictionary< int, float[]> `[`GetMemories`](#classExternalCommunicator_1ab94d8e957e15431baa7681594e928098)`(string brainName)` | Returns the memories corrensponding to the brain called brainName that were received throught the socket.
`public inline Dictionary< int, float > `[`GetValues`](#classExternalCommunicator_1a7f9afb60d40a063caffe57010d25f073)`(string brainName)` | Returns the values corrensponding to the brain called brainName that were received throught the socket.

## Members

#### `public inline  `[`ExternalCommunicator`](#classExternalCommunicator_1a62c140342df7ac93e567d0b9a2589615)`(`[`Academy`](#classAcademy)` aca)` 

Consrtuctor for the External [Communicator](#interfaceCommunicator).

#### `public inline void `[`SubscribeBrain`](#classExternalCommunicator_1af83f0ed2244661ef60044f36f2066ec6)`(`[`Brain`](#classBrain)` brain)` 

Adds the brain to the list of brains which have already decided their actions.

#### `public inline bool `[`CommunicatorHandShake`](#classExternalCommunicator_1a7eb674f0acbe274e4933df5aea101f72)`()` 

First contact between [Communicator](#interfaceCommunicator) and external process.

#### `public inline void `[`InitializeCommunicator`](#classExternalCommunicator_1a4b92ce2971224e0277e759e1d452c4ea)`()` 

Contains the logic for the initializtation of the socket.

#### `public inline `[`ExternalCommand`](#Communicator_8cs_1a2056e4a11471702052b3ee067b38e355)` `[`GetCommand`](#classExternalCommunicator_1a1074576c637cf6309c12c16923398261)`()` 

Listens to the socket for a command and returns the corresponding External Command.

#### `public inline Dictionary< string, float > `[`GetResetParameters`](#classExternalCommunicator_1a77da9de880e3bfc14125162d4c3adedc)`()` 

Listens to the socket for the new resetParameters.

#### `public inline void `[`giveBrainInfo`](#classExternalCommunicator_1a0f500c2709fca846e0cee3069eb4a335)`(`[`Brain`](#classBrain)` brain)` 

Collects the information from the brains and sends it accross the socket.

#### `public inline void `[`UpdateActions`](#classExternalCommunicator_1a45bd50d0aa1084d7a895bea7e2f6314c)`()` 

Listens for actions, memories, and values and sends them to the corrensponding brains.

#### `public inline Dictionary< int, float[]> `[`GetDecidedAction`](#classExternalCommunicator_1acd5ab12458fc39be0ed706cf3508e10d)`(string brainName)` 

Returns the actions corrensponding to the brain called brainName that were received throught the socket.

#### `public inline Dictionary< int, float[]> `[`GetMemories`](#classExternalCommunicator_1ab94d8e957e15431baa7681594e928098)`(string brainName)` 

Returns the memories corrensponding to the brain called brainName that were received throught the socket.

#### `public inline Dictionary< int, float > `[`GetValues`](#classExternalCommunicator_1a7f9afb60d40a063caffe57010d25f073)`(string brainName)` 

Returns the values corrensponding to the brain called brainName that were received throught the socket.

# class `Monitor` 

```
class Monitor
  : public MonoBehaviour
```  

[Monitor](#classMonitor) is used to display information. Use the log function to add information to your monitor.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------

## Members

# class `ExternalCommunicator::ResetParametersMessage` 

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------

## Members

# class `ScreenConfiguration` 

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public int `[`width`](#classScreenConfiguration_1a2474a5474cbff19523a51eb1de01cda4) | 
`public int `[`height`](#classScreenConfiguration_1ad12fc34ce789bce6c8a05d8a17138534) | 
`public int `[`qualityLevel`](#classScreenConfiguration_1a6769bca3d7b4c1e3e32b774729afabd1) | 
`public float `[`timeScale`](#classScreenConfiguration_1ad921de4ff2f44d55940f0b2f3a86149d) | 
`public int `[`targetFrameRate`](#classScreenConfiguration_1ac63e74b2fef449402905cb50167aa622) | 
`public inline  `[`ScreenConfiguration`](#classScreenConfiguration_1afb0f03c15306997796f6a4adfea04389)`(int w,int h,int q,float ts,int tf)` | 

## Members

#### `public int `[`width`](#classScreenConfiguration_1a2474a5474cbff19523a51eb1de01cda4) 

#### `public int `[`height`](#classScreenConfiguration_1ad12fc34ce789bce6c8a05d8a17138534) 

#### `public int `[`qualityLevel`](#classScreenConfiguration_1a6769bca3d7b4c1e3e32b774729afabd1) 

#### `public float `[`timeScale`](#classScreenConfiguration_1ad921de4ff2f44d55940f0b2f3a86149d) 

#### `public int `[`targetFrameRate`](#classScreenConfiguration_1ac63e74b2fef449402905cb50167aa622) 

#### `public inline  `[`ScreenConfiguration`](#classScreenConfiguration_1afb0f03c15306997796f6a4adfea04389)`(int w,int h,int q,float ts,int tf)` 

# class `ExternalCommunicator::StepMessage` 

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------

## Members

# class `UnityAgentsException` 

```
class UnityAgentsException
  : public Exception
```  

Contains exceptions specific to ML-Agents.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline  `[`UnityAgentsException`](#classUnityAgentsException_1acce20398c37dbe35d46dd408e50e5cbc)`(string message)` | When a [UnityAgentsException](#classUnityAgentsException) is called, the timeScale is set to 0. The simulation will end since no steps will be taken.
`protected inline  `[`UnityAgentsException`](#classUnityAgentsException_1a0092d1a70ccdea86ed5a9208e2a3595a)`(System.Runtime.Serialization.SerializationInfo info,System.Runtime.Serialization.StreamingContext context)` | A constructor is needed for serialization when an exception propagates from a remoting server to the client.

## Members

#### `public inline  `[`UnityAgentsException`](#classUnityAgentsException_1acce20398c37dbe35d46dd408e50e5cbc)`(string message)` 

When a [UnityAgentsException](#classUnityAgentsException) is called, the timeScale is set to 0. The simulation will end since no steps will be taken.

#### `protected inline  `[`UnityAgentsException`](#classUnityAgentsException_1a0092d1a70ccdea86ed5a9208e2a3595a)`(System.Runtime.Serialization.SerializationInfo info,System.Runtime.Serialization.StreamingContext context)` 

A constructor is needed for serialization when an exception propagates from a remoting server to the client.

# struct `AcademyParameters` 

[AcademyParameters](#structAcademyParameters) is a structure containing basic information about the training environment.

The [AcademyParameters](#structAcademyParameters) will be sent via socket at the start of the Environment. This structure does not need to be modified.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public string `[`AcademyName`](#structAcademyParameters_1afda75cd0902be6bd752cfc3e171bc3a8) | The name of the [Academy](#classAcademy). If the communicator is External, it will be the name of the [Academy](#classAcademy) GameObject.
`public string `[`apiNumber`](#structAcademyParameters_1a67d3e9471a4ee14570b14b581dfa1e89) | The API number for the communicator.
`public string `[`logPath`](#structAcademyParameters_1a7f4934f0abd9c9e9d1da7b4b49fdfecc) | The location of the logfile.
`public Dictionary< string, float > `[`resetParameters`](#structAcademyParameters_1a1f320e419c5cdf7a00008b742edaba21) | The default reset parameters are sent via socket.
`public List< string > `[`brainNames`](#structAcademyParameters_1a75c46c02027d75ebe21ecbcfe4f68348) | A list of the all the brains names sent via socket.
`public List< `[`BrainParameters`](#classBrainParameters)` > `[`brainParameters`](#structAcademyParameters_1ace966bbe7680dd353ebff42ae8b83425) | A list of the External brains parameters sent via socket.
`public List< string > `[`externalBrainNames`](#structAcademyParameters_1a5bc089bc6f6b72bf37996804b3628e98) | A list of the External brains names sent via socket.

## Members

#### `public string `[`AcademyName`](#structAcademyParameters_1afda75cd0902be6bd752cfc3e171bc3a8) 

The name of the [Academy](#classAcademy). If the communicator is External, it will be the name of the [Academy](#classAcademy) GameObject.

#### `public string `[`apiNumber`](#structAcademyParameters_1a67d3e9471a4ee14570b14b581dfa1e89) 

The API number for the communicator.

#### `public string `[`logPath`](#structAcademyParameters_1a7f4934f0abd9c9e9d1da7b4b49fdfecc) 

The location of the logfile.

#### `public Dictionary< string, float > `[`resetParameters`](#structAcademyParameters_1a1f320e419c5cdf7a00008b742edaba21) 

The default reset parameters are sent via socket.

#### `public List< string > `[`brainNames`](#structAcademyParameters_1a75c46c02027d75ebe21ecbcfe4f68348) 

A list of the all the brains names sent via socket.

#### `public List< `[`BrainParameters`](#classBrainParameters)` > `[`brainParameters`](#structAcademyParameters_1ace966bbe7680dd353ebff42ae8b83425) 

A list of the External brains parameters sent via socket.

#### `public List< string > `[`externalBrainNames`](#structAcademyParameters_1a5bc089bc6f6b72bf37996804b3628e98) 

A list of the External brains names sent via socket.

# struct `CoreBrainPlayer::ContinuousPlayerAction` 

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public KeyCode `[`key`](#structCoreBrainPlayer_1_1ContinuousPlayerAction_1ab74d86e61479cd4349198a5190b3da48) | 
`public int `[`index`](#structCoreBrainPlayer_1_1ContinuousPlayerAction_1a750b5d744c39a06bfb13e6eb010e35d0) | 
`public float `[`value`](#structCoreBrainPlayer_1_1ContinuousPlayerAction_1a17956fe0129d3d4c94ebc06cfef2ad82) | 

## Members

#### `public KeyCode `[`key`](#structCoreBrainPlayer_1_1ContinuousPlayerAction_1ab74d86e61479cd4349198a5190b3da48) 

#### `public int `[`index`](#structCoreBrainPlayer_1_1ContinuousPlayerAction_1a750b5d744c39a06bfb13e6eb010e35d0) 

#### `public float `[`value`](#structCoreBrainPlayer_1_1ContinuousPlayerAction_1a17956fe0129d3d4c94ebc06cfef2ad82) 

# struct `CoreBrainPlayer::DiscretePlayerAction` 

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public KeyCode `[`key`](#structCoreBrainPlayer_1_1DiscretePlayerAction_1ab74d86e61479cd4349198a5190b3da48) | 
`public int `[`value`](#structCoreBrainPlayer_1_1DiscretePlayerAction_1ac4f474c82e82cbb89ca7c36dd52be0ed) | 

## Members

#### `public KeyCode `[`key`](#structCoreBrainPlayer_1_1DiscretePlayerAction_1ab74d86e61479cd4349198a5190b3da48) 

#### `public int `[`value`](#structCoreBrainPlayer_1_1DiscretePlayerAction_1ac4f474c82e82cbb89ca7c36dd52be0ed) 

# struct `Monitor::DisplayValue` 

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public float `[`time`](#structMonitor_1_1DisplayValue_1a8b8dfe2335a5bf90695960dc6a1c5d3b) | 
`public object `[`value`](#structMonitor_1_1DisplayValue_1a767de28b215ddaceb31188034a0838c0) | 
`public `[`MonitorType`](#Monitor_8cs_1ac69ce5a28782fdc01f7ba5c236cd5f72)` `[`monitorDisplayType`](#structMonitor_1_1DisplayValue_1a9b1b8611317e9d41f5ad11f786c993a6) | 

## Members

#### `public float `[`time`](#structMonitor_1_1DisplayValue_1a8b8dfe2335a5bf90695960dc6a1c5d3b) 

#### `public object `[`value`](#structMonitor_1_1DisplayValue_1a767de28b215ddaceb31188034a0838c0) 

#### `public `[`MonitorType`](#Monitor_8cs_1ac69ce5a28782fdc01f7ba5c236cd5f72)` `[`monitorDisplayType`](#structMonitor_1_1DisplayValue_1a9b1b8611317e9d41f5ad11f786c993a6) 

# struct `Academy::ResetParameter` 

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public string `[`key`](#structAcademy_1_1ResetParameter_1aa8fa75d45876bcbe59f33f49e7d3572d) | 
`public float `[`value`](#structAcademy_1_1ResetParameter_1a17956fe0129d3d4c94ebc06cfef2ad82) | 

## Members

#### `public string `[`key`](#structAcademy_1_1ResetParameter_1aa8fa75d45876bcbe59f33f49e7d3572d) 

#### `public float `[`value`](#structAcademy_1_1ResetParameter_1a17956fe0129d3d4c94ebc06cfef2ad82) 

# struct `resolution` 

Only need to be modified in the brain's inpector. Defines what is the resolution of the camera

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public int `[`width`](#structresolution_1a2474a5474cbff19523a51eb1de01cda4) | The width of the observation in pixels.
`public int `[`height`](#structresolution_1ad12fc34ce789bce6c8a05d8a17138534) | The height of the observation in pixels.
`public bool `[`blackAndWhite`](#structresolution_1a544e741c945ac9696397da980479be3b) | If true, the image will be in black and white. If false, it will be in colors RGB.

## Members

#### `public int `[`width`](#structresolution_1a2474a5474cbff19523a51eb1de01cda4) 

The width of the observation in pixels.

#### `public int `[`height`](#structresolution_1ad12fc34ce789bce6c8a05d8a17138534) 

The height of the observation in pixels.

#### `public bool `[`blackAndWhite`](#structresolution_1a544e741c945ac9696397da980479be3b) 

If true, the image will be in black and white. If false, it will be in colors RGB.

# struct `CoreBrainInternal::TensorFlowAgentPlaceholder` 

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public string `[`name`](#structCoreBrainInternal_1_1TensorFlowAgentPlaceholder_1a8ccf841cb59e451791bcb2e1ac4f1edc) | 
`public tensorType `[`valueType`](#structCoreBrainInternal_1_1TensorFlowAgentPlaceholder_1ae2c7b0361175e4d7af433ee4d498fcf1) | 
`public float `[`minValue`](#structCoreBrainInternal_1_1TensorFlowAgentPlaceholder_1a9368fa48348c19f6f31019f7705108db) | 
`public float `[`maxValue`](#structCoreBrainInternal_1_1TensorFlowAgentPlaceholder_1abdcc1d365e9355d5e10d50c4340e1cb5) | 

## Members

#### `public string `[`name`](#structCoreBrainInternal_1_1TensorFlowAgentPlaceholder_1a8ccf841cb59e451791bcb2e1ac4f1edc) 

#### `public tensorType `[`valueType`](#structCoreBrainInternal_1_1TensorFlowAgentPlaceholder_1ae2c7b0361175e4d7af433ee4d498fcf1) 

#### `public float `[`minValue`](#structCoreBrainInternal_1_1TensorFlowAgentPlaceholder_1a9368fa48348c19f6f31019f7705108db) 

#### `public float `[`maxValue`](#structCoreBrainInternal_1_1TensorFlowAgentPlaceholder_1abdcc1d365e9355d5e10d50c4340e1cb5) 

Generated by [Moxygen](https://sourcey.com/moxygen)
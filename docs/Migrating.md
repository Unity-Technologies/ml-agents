# Upgrading

# Migrating
## Migrating the package to version 2.0
- The official version of Unity ML-Agents supports is now 2020.3 LTS. If you run
  into issues, please consider deleting your project's Library folder and reponening your
  project.
- If you used any of the APIs that were deprecated before version 2.0, you need to use their replacement. These
deprecated APIs have been removed. See the migration steps bellow for specific API replacements.

### Deprecated methods removed
| **Deprecated API** | **Suggested Replacement** |
|:-------:|:------:|
| `IActuator ActuatorComponent.CreateActuator()` | `IActuator[] ActuatorComponent.CreateActuators()` |
| `IActionReceiver.PackActions(in float[] destination)` | none |
| `Agent.CollectDiscreteActionMasks(DiscreteActionMasker actionMasker)` | `Agent.WriteDiscreteActionMask(IDiscreteActionMask actionMask)` |
| `Agent.Heuristic(float[] actionsOut)` | `Agent.Heuristic(in ActionBuffers actionsOut)` |
| `Agent.OnActionReceived(float[] vectorAction)` | `Agent.OnActionReceived(ActionBuffers actions)` |
| `Agent.GetAction()` | `Agent.GetStoredActionBuffers()` |
| `BrainParameters.SpaceType`, `VectorActionSize`, `VectorActionSpaceType`, and `NumActions` | `BrainParameters.ActionSpec` |
| `ObservationWriter.AddRange(IEnumerable<float> data, int writeOffset = 0)` | `ObservationWriter. AddList(IList<float> data, int writeOffset = 0` |
| `SensorComponent.IsVisual()` and `IsVector()` | none |
| `VectorSensor.AddObservation(IEnumerable<float> observation)` | `VectorSensor.AddObservation(IList<float> observation)` |
| `SideChannelsManager` | `SideChannelManager` |

### IDiscreteActionMask changes
- The interface for disabling specific discrete actions has changed. `IDiscreteActionMask.WriteMask()` was removed,
and replaced with `SetActionEnabled()`. Instead of returning an IEnumerable with indices to disable, you can
now call `SetActionEnabled` for each index to disable (or enable). As an example, if you overrode
`Agent.WriteDiscreteActionMask()` with something that looked like:

```csharp
public override void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
{
    var branch = 2;
    var actionsToDisable = new[] {1, 3};
    actionMask.WriteMask(branch, actionsToDisable);
}
```

the equivalent code would now be

```csharp
public override void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
{
    var branch = 2;
    actionMask.SetActionEnabled(branch, 1, false);
    actionMask.SetActionEnabled(branch, 3, false);
}
```
### IActuator changes
- The `IActuator` interface now implements `IHeuristicProvider`.  Please add the corresponding `Heuristic(in ActionBuffers)`
method to your custom Actuator classes.

### ISensor and SensorComponent changes
- The `ISensor.GetObservationShape()` method and `ITypedSensor`
and `IDimensionPropertiesSensor` interfaces were removed, and `GetObservationSpec()` was added. You can use
`ObservationSpec.Vector()` or `ObservationSpec.Visual()` to generate `ObservationSpec`s that are equivalent to
the previous shape. For example, if your old ISensor looked like:

```csharp
public override int[] GetObservationShape()
{
    return new[] { m_Height, m_Width, m_NumChannels };
}
```

the equivalent code would now be

```csharp
public override ObservationSpec GetObservationSpec()
{
    return ObservationSpec.Visual(m_Height, m_Width, m_NumChannels);
}
```

- The `ISensor.GetCompressionType()` method and `ISparseChannelSensor` interface was removed,
and `GetCompressionSpec()` was added. You can use `CompressionSpec.Default()` or
`CompressionSpec.Compressed()` to generate `CompressionSpec`s that are  equivalent to
 the previous values. For example, if your old ISensor looked like:
 ```csharp
public virtual SensorCompressionType GetCompressionType()
{
    return SensorCompressionType.None;
}
```

the equivalent code would now be

```csharp
public CompressionSpec GetCompressionSpec()
{
    return CompressionSpec.Default();
}
```

- The abstract method `SensorComponent.GetObservationShape()` was removed.
- The abstract method `SensorComponent.CreateSensor()` was replaced with `CreateSensors()`, which returns an `ISensor[]`.

### Match3 integration changes
The Match-3 integration utilities were moved from `com.unity.ml-agents.extensions` to `com.unity.ml-agents`.

The `AbstractBoard` interface was changed:
* `AbstractBoard` no longer contains `Rows`, `Columns`, `NumCellTypes`, and `NumSpecialTypes` fields.
* `public abstract BoardSize GetMaxBoardSize()` was added as an abstract method. `BoardSize` is a new struct that
contains `Rows`, `Columns`, `NumCellTypes`, and `NumSpecialTypes` fields, with the same meanings as the old
`AbstractBoard` fields.
* `public virtual BoardSize GetCurrentBoardSize()` is an optional method; by default it returns `GetMaxBoardSize()`. If
you wish to use a single behavior to work with multiple board sizes, override `GetCurrentBoardSize()` to return the
current `BoardSize`. The values returned by `GetCurrentBoardSize()` must be less than or equal to the corresponding
values from `GetMaxBoardSize()`.

### GridSensor changes
The sensor configuration has changed:
* The sensor implementation has been refactored and exsisting GridSensor created from extension package
will not work in newer version. Some errors might show up when loading the old sensor in the scene.
You'll need to remove the old sensor and create a new GridSensor.
* These parameters names have changed but still refer to the same concept in the sensor: `GridNumSide` -> `GridSize`,
`RotateToAgent` -> `RotateWithAgent`, `ObserveMask` -> `ColliderMask`, `DetectableObjects` -> `DetectableTags`
* `DepthType` (`ChanelBase`/`ChannelHot`) option and `ChannelDepth` are removed. Now the default is
one-hot encoding for detected tag. If you were using original GridSensor without overriding any method,
switching to new GridSensor will produce similar effect for training although the actual observations
will be slightly different.

For creating your GridSensor implementation with custom data:
* To create custom GridSensor, derive from `GridSensorBase` instead of `GridSensor`. Besides overriding
`GetObjectData()`, you will also need to consider override `GetCellObservationSize()`, `IsDataNormalized()`
and `GetProcessCollidersMethod()` according to the data you collect. Also you'll need to override
`GridSensorComponent.GetGridSensors()` and return your custom GridSensor.
* The input argument `tagIndex` in `GetObjectData()` has changed from 1-indexed to 0-indexed and the
data type changed from `float` to `int`. The index of first detectable tag will be 0 instead of 1.
`normalizedDistance` was removed from input.
* The observation data should be written to the input `dataBuffer` instead of creating and returning a new array.
* Removed the constraint of all data required to be normalized. You should specify it in `IsDataNormalized()`.
Sensors with non-normalized data cannot use PNG compression type.
* The sensor will not further encode the data recieved from `GetObjectData()` anymore. The values
recieved from `GetObjectData()` will be the observation sent to the trainer.

### LSTM models from previous releases no longer supported
The way the Unity Inference Engine processes LSTM (recurrent neural networks) has changed. As a result, models
trained with previous versions of ML-Agents will not be usable at inference if they were trained with a `memory`
setting in the `.yaml` config file.
If you want to use a model that has a recurrent neural network in this release of ML-Agents, you need to train
the model using the python trainer from this release.


## Migrating to Release 13
### Implementing IHeuristic in your IActuator implementations
 - If you have any custom actuators, you can now implement the `IHeuristicProvider` interface to have your actuator
  handle the generation of actions when an Agent is running in heuristic mode.
- `VectorSensor.AddObservation(IEnumerable<float>)` is deprecated. Use `VectorSensor.AddObservation(IList<float>)`
  instead.
- `ObservationWriter.AddRange()` is deprecated. Use `ObservationWriter.AddList()` instead.
- `ActuatorComponent.CreateAcuator()` is deprecated.  Please use override `ActuatorComponent.CreateActuators`
  instead.  Since `ActuatorComponent.CreateActuator()` is abstract, you will still need to override it in your
  class until it is removed.  It is only ever called if you don't override `ActuatorComponent.CreateActuators`.
  You can suppress the warnings by surrounding the method with the following pragma:
    ```c#
    #pragma warning disable 672
    public IActuator CreateActuator() { ... }
    #pragma warning restore 672
    ```


# Migrating
## Migrating to Release 11
### Agent virtual method deprecation
 - `Agent.CollectDiscreteActionMasks()` was deprecated and should be replaced with `Agent.WriteDiscreteActionMask()`
 - `Agent.Heuristic(float[])` was deprecated and should be replaced with `Agent.Heuristic(ActionBuffers)`.
 - `Agent.OnActionReceived(float[])` was deprecated and should be replaced with `Agent.OnActionReceived(ActionBuffers)`.
 - `Agent.GetAction()` was deprecated and should be replaced with `Agent.GetStoredActionBuffers()`.

The default implementation of these will continue to call the deprecated versions where appropriate. However, the
deprecated versions may not be compatible with continuous and discrete actions on the same Agent.

### BrainParameters field and method deprecation
 - `BrainParameters.VectorActionSize` was deprecated; you can now set `BrainParameters.ActionSpec.NumContinuousActions`
 or `BrainParameters.ActionSpec.BranchSizes` instead.
 - `BrainParameters.VectorActionSpaceType` was deprecated, since both continuous and discrete actions can now be used.
 - `BrainParameters.NumActions()` was deprecated. Use  `BrainParameters.ActionSpec.NumContinuousActions` and
 `BrainParameters.ActionSpec.NumDiscreteActions` instead.

## Migrating from Release 7 to latest

### Important changes
- Some trainer files were moved. If you were using the `TrainerFactory` class, it was moved to
the `trainers/trainer` folder.
- The `components` folder containing `bc` and `reward_signals` code was moved to the `trainers/tf`
folder

### Steps to Migrate
- Replace calls to `from mlagents.trainers.trainer_util import TrainerFactory` to `from mlagents.trainers.trainer import TrainerFactory`
- Replace calls to `from mlagents.trainers.trainer_util import handle_existing_directories` to `from mlagents.trainers.directory_utils import validate_existing_directories`
- Replace `mlagents.trainers.components` with `mlagents.trainers.tf.components` in your import statements.


## Migrating from Release 3 to Release 7

### Important changes
- The Parameter Randomization feature has been merged with the Curriculum feature. It is now possible to specify a sampler
in the lesson of a Curriculum. Curriculum has been refactored and is now specified at the level of the parameter, not the
behavior. More information
[here](https://github.com/Unity-Technologies/ml-agents/blob/release_19_docs/docs/Training-ML-Agents.md).(#4160)

### Steps to Migrate
- The configuration format for curriculum and parameter randomization has changed. To upgrade your configuration files,
an upgrade script has been provided. Run `python -m mlagents.trainers.upgrade_config -h` to see the script usage. Note that you will have had to upgrade to/install the current version of ML-Agents before running the script. To update manually:
  - If your config file used a `parameter_randomization` section, rename that section to `environment_parameters`
  - If your config file used a `curriculum` section, you will need to rewrite your curriculum with this [format](Training-ML-Agents.md#curriculum).

## Migrating from Release 1 to Release 3

### Important changes
- Training artifacts (trained models, summaries) are now found under `results/`
  instead of `summaries/` and `models/`.
- Trainer configuration, curriculum configuration, and parameter randomization
  configuration have all been moved to a single YAML file. (#3791)
- Trainer configuration format has changed, and using a "default" behavior name has
  been deprecated. (#3936)
- `max_step` in the `TerminalStep` and `TerminalSteps` objects was renamed `interrupted`.
- On the UnityEnvironment API, `get_behavior_names()` and `get_behavior_specs()` methods were combined into the property `behavior_specs` that contains a mapping from behavior names to behavior spec.
- `use_visual` and `allow_multiple_visual_obs` in the `UnityToGymWrapper` constructor
were replaced by `allow_multiple_obs` which allows one or more visual observations and
vector observations to be used simultaneously.
- `--save-freq` has been removed from the CLI and is now configurable in the trainer configuration
  file.
- `--lesson` has been removed from the CLI. Lessons will resume when using `--resume`.
  To start at a different lesson, modify your Curriculum configuration.

### Steps to Migrate
- To upgrade your configuration files, an upgrade script has been provided. Run
  `python -m mlagents.trainers.upgrade_config -h` to see the script usage. Note that you will have
  had to upgrade to/install the current version of ML-Agents before running the script.

  To do it manually, copy your `<BehaviorName>` sections from `trainer_config.yaml` into a separate trainer configuration file, under a `behaviors` section.
  The `default` section is no longer needed. This new file should be specific to your environment, and not contain
  configurations for multiple environments (unless they have the same Behavior Names).
  - You will need to reformat your trainer settings as per the [example](Training-ML-Agents.md).
  - If your training uses [curriculum](Training-ML-Agents.md#curriculum-learning), move those configurations under a `curriculum` section.
  - If your training uses [parameter randomization](Training-ML-Agents.md#environment-parameter-randomization), move
  the contents of the sampler config to `parameter_randomization` in the main trainer configuration.
- If you are using `UnityEnvironment` directly, replace `max_step` with `interrupted`
 in the `TerminalStep` and `TerminalSteps` objects.
 - Replace usage of `get_behavior_names()` and `get_behavior_specs()` in UnityEnvironment with `behavior_specs`.
 - If you use the `UnityToGymWrapper`, remove `use_visual` and `allow_multiple_visual_obs`
 from the constructor and add `allow_multiple_obs = True` if the environment contains either
 both visual and vector observations or multiple visual observations.
 - If you were setting `--save-freq` in the CLI, add a `checkpoint_interval` value in your
  trainer configuration, and set it equal to `save-freq * n_agents_in_scene`.

## Migrating from 0.15 to Release 1

### Important changes

- The `MLAgents` C# namespace was renamed to `Unity.MLAgents`, and other nested
  namespaces were similarly renamed (#3843).
- The `--load` and `--train` command-line flags have been deprecated and
  replaced with `--resume` and `--inference`.
- Running with the same `--run-id` twice will now throw an error.
- The `play_against_current_self_ratio` self-play trainer hyperparameter has
  been renamed to `play_against_latest_model_ratio`
- Removed the multi-agent gym option from the gym wrapper. For multi-agent
  scenarios, use the [Low Level Python API](Python-API.md).
- The low level Python API has changed. You can look at the document
  [Low Level Python API documentation](Python-API.md) for more information. If
  you use `mlagents-learn` for training, this should be a transparent change.
- The obsolete `Agent` methods `GiveModel`, `Done`, `InitializeAgent`,
  `AgentAction` and `AgentReset` have been removed.
- The signature of `Agent.Heuristic()` was changed to take a `float[]` as a
  parameter, instead of returning the array. This was done to prevent a common
  source of error where users would return arrays of the wrong size.
- The SideChannel API has changed (#3833, #3660) :
  - Introduced the `SideChannelManager` to register, unregister and access side
    channels.
  - `EnvironmentParameters` replaces the default `FloatProperties`. You can
    access the `EnvironmentParameters` with
    `Academy.Instance.EnvironmentParameters` on C#. If you were previously
    creating a `UnityEnvironment` in python and passing it a
    `FloatPropertiesChannel`, create an `EnvironmentParametersChannel` instead.
  - `SideChannel.OnMessageReceived` is now a protected method (was public)
  - SideChannel IncomingMessages methods now take an optional default argument,
    which is used when trying to read more data than the message contains.
  - Added a feature to allow sending stats from C# environments to TensorBoard
    (and other python StatsWriters). To do this from your code, use
    `Academy.Instance.StatsRecorder.Add(key, value)`(#3660)
- `num_updates` and `train_interval` for SAC have been replaced with
  `steps_per_update`.
- The `UnityEnv` class from the `gym-unity` package was renamed
  `UnityToGymWrapper` and no longer creates the `UnityEnvironment`. Instead, the
  `UnityEnvironment` must be passed as input to the constructor of
  `UnityToGymWrapper`
- Public fields and properties on several classes were renamed to follow Unity's
  C# style conventions. All public fields and properties now use "PascalCase"
  instead of "camelCase"; for example, `Agent.maxStep` was renamed to
  `Agent.MaxStep`. For a full list of changes, see the pull request. (#3828)
- `WriteAdapter` was renamed to `ObservationWriter`. (#3834)

### Steps to Migrate

- In C# code, replace `using MLAgents` with `using Unity.MLAgents`. Replace
  other nested namespaces such as `using MLAgents.Sensors` with
  `using Unity.MLAgents.Sensors`
- Replace the `--load` flag with `--resume` when calling `mlagents-learn`, and
  don't use the `--train` flag as training will happen by default. To run
  without training, use `--inference`.
- To force-overwrite files from a pre-existing run, add the `--force`
  command-line flag.
- The Jupyter notebooks have been removed from the repository.
- If your Agent class overrides `Heuristic()`, change the signature to
  `public override void Heuristic(float[] actionsOut)` and assign values to
  `actionsOut` instead of returning an array.
- If you used `SideChannels` you must:
  - Replace `Academy.FloatProperties` with
    `Academy.Instance.EnvironmentParameters`.
  - `Academy.RegisterSideChannel` and `Academy.UnregisterSideChannel` were
    removed. Use `SideChannelManager.RegisterSideChannel` and
    `SideChannelManager.UnregisterSideChannel` instead.
- Set `steps_per_update` to be around equal to the number of agents in your
  environment, times `num_updates` and divided by `train_interval`.
- Replace `UnityEnv` with `UnityToGymWrapper` in your code. The constructor no
  longer takes a file name as input but a fully constructed `UnityEnvironment`
  instead.
- Update uses of "camelCase" fields and properties to "PascalCase".

## Migrating from 0.14 to 0.15

### Important changes

- The `Agent.CollectObservations()` virtual method now takes as input a
  `VectorSensor` sensor as argument. The `Agent.AddVectorObs()` methods were
  removed.
- The `SetMask` was renamed to `SetMask` method must now be called on the
  `DiscreteActionMasker` argument of the `CollectDiscreteActionMasks` virtual
  method.
- We consolidated our API for `DiscreteActionMasker`. `SetMask` takes two
  arguments : the branch index and the list of masked actions for that branch.
- The `Monitor` class has been moved to the Examples Project. (It was prone to
  errors during testing)
- The `MLAgents.Sensors` namespace has been introduced. All sensors classes are
  part of the `MLAgents.Sensors` namespace.
- The `MLAgents.SideChannels` namespace has been introduced. All side channel
  classes are part of the `MLAgents.SideChannels` namespace.
- The interface for `RayPerceptionSensor.PerceiveStatic()` was changed to take
  an input class and write to an output class, and the method was renamed to
  `Perceive()`.
- The `SetMask` method must now be called on the `DiscreteActionMasker` argument
  of the `CollectDiscreteActionMasks` method.
- The method `GetStepCount()` on the Agent class has been replaced with the
  property getter `StepCount`
- The `--multi-gpu` option has been removed temporarily.
- `AgentInfo.actionMasks` has been renamed to `AgentInfo.discreteActionMasks`.
- `BrainParameters` and `SpaceType` have been removed from the public API
- `BehaviorParameters` have been removed from the public API.
- `DecisionRequester` has been made internal (you can still use the
  DecisionRequesterComponent from the inspector). `RepeatAction` was renamed
  `TakeActionsBetweenDecisions` for clarity.
- The following methods in the `Agent` class have been renamed. The original
  method names will be removed in a later release:
  - `InitializeAgent()` was renamed to `Initialize()`
  - `AgentAction()` was renamed to `OnActionReceived()`
  - `AgentReset()` was renamed to `OnEpsiodeBegin()`
  - `Done()` was renamed to `EndEpisode()`
  - `GiveModel()` was renamed to `SetModel()`
- The `IFloatProperties` interface has been removed.
- The interface for SideChannels was changed:
  - In C#, `OnMessageReceived` now takes a `IncomingMessage` argument, and
    `QueueMessageToSend` takes an `OutgoingMessage` argument.
  - In python, `on_message_received` now takes a `IncomingMessage` argument, and
    `queue_message_to_send` takes an `OutgoingMessage` argument.
  - Automatic stepping for Academy is now controlled from the
    AutomaticSteppingEnabled property.

### Steps to Migrate

- Add the `using MLAgents.Sensors;` in addition to `using MLAgents;` on top of
  your Agent's script.
- Replace your Agent's implementation of `CollectObservations()` with
  `CollectObservations(VectorSensor sensor)`. In addition, replace all calls to
  `AddVectorObs()` with `sensor.AddObservation()` or
  `sensor.AddOneHotObservation()` on the `VectorSensor` passed as argument.
- Replace your calls to `SetActionMask` on your Agent to
  `DiscreteActionMasker.SetActionMask` in `CollectDiscreteActionMasks`.
- If you call `RayPerceptionSensor.PerceiveStatic()` manually, add your inputs
  to a `RayPerceptionInput`. To get the previous float array output, iterate
  through `RayPerceptionOutput.rayOutputs` and call
  `RayPerceptionOutput.RayOutput.ToFloatArray()`.
- Replace all calls to `Agent.GetStepCount()` with `Agent.StepCount`
- We strongly recommend replacing the following methods with their new
  equivalent as they will be removed in a later release:
  - `InitializeAgent()` to `Initialize()`
  - `AgentAction()` to `OnActionReceived()`
  - `AgentReset()` to `OnEpisodeBegin()`
  - `Done()` to `EndEpisode()`
  - `GiveModel()` to `SetModel()`
- Replace `IFloatProperties` variables with `FloatPropertiesChannel` variables.
- If you implemented custom `SideChannels`, update the signatures of your
  methods, and add your data to the `OutgoingMessage` or read it from the
  `IncomingMessage`.
- Replace calls to Academy.EnableAutomaticStepping()/DisableAutomaticStepping()
  with Academy.AutomaticSteppingEnabled = true/false.

## Migrating from 0.13 to 0.14

### Important changes

- The `UnitySDK` folder has been split into a Unity Package
  (`com.unity.ml-agents`) and an examples project (`Project`). Please follow the
  [Installation Guide](Installation.md) to get up and running with this new repo
  structure.
- Several changes were made to how agents are reset and marked as done:
  - Calling `Done()` on the Agent will now reset it immediately and call the
    `AgentReset` virtual method. (This is to simplify the previous logic in
    which the Agent had to wait for the next `EnvironmentStep` to reset)
  - The "Reset on Done" setting in AgentParameters was removed; this is now
    effectively always true. `AgentOnDone` virtual method on the Agent has been
    removed.
- The `Decision Period` and `On Demand decision` checkbox have been removed from
  the Agent. On demand decision is now the default (calling `RequestDecision` on
  the Agent manually.)
- The Academy class was changed to a singleton, and its virtual methods were
  removed.
- Trainer steps are now counted per-Agent, not per-environment as in previous
  versions. For instance, if you have 10 Agents in the scene, 20 environment
  steps now corresponds to 200 steps as printed in the terminal and in
  Tensorboard.
- Curriculum config files are now YAML formatted and all curricula for a
  training run are combined into a single file.
- The `--num-runs` command-line option has been removed from `mlagents-learn`.
- Several fields on the Agent were removed or made private in order to simplify
  the interface.
  - The `agentParameters` field of the Agent has been removed. (Contained only
    `maxStep` information)
  - `maxStep` is now a public field on the Agent. (Was moved from
    `agentParameters`)
  - The `Info` field of the Agent has been made private. (Was only used
    internally and not meant to be modified outside of the Agent)
  - The `GetReward()` method on the Agent has been removed. (It was being
    confused with `GetCumulativeReward()`)
  - The `AgentAction` struct no longer contains a `value` field. (Value
    estimates were not set during inference)
  - The `GetValueEstimate()` method on the Agent has been removed.
  - The `UpdateValueAction()` method on the Agent has been removed.
- The deprecated `RayPerception3D` and `RayPerception2D` classes were removed,
  and the `legacyHitFractionBehavior` argument was removed from
  `RayPerceptionSensor.PerceiveStatic()`.
- RayPerceptionSensor was inconsistent in how it handle scale on the Agent's
  transform. It now scales the ray length and sphere size for casting as the
  transform's scale changes.

### Steps to Migrate

- Follow the instructions on how to install the `com.unity.ml-agents` package
  into your project in the [Installation Guide](Installation.md).
- If your Agent implemented `AgentOnDone` and did not have the checkbox
  `Reset On Done` checked in the inspector, you must call the code that was in
  `AgentOnDone` manually.
- If you give your Agent a reward or penalty at the end of an episode (e.g. for
  reaching a goal or falling off of a platform), make sure you call
  `AddReward()` or `SetReward()` _before_ calling `Done()`. Previously, the
  order didn't matter.
- If you were not using `On Demand Decision` for your Agent, you **must** add a
  `DecisionRequester` component to your Agent GameObject and set its
  `Decision Period` field to the old `Decision Period` of the Agent.
- If you have a class that inherits from Academy:
  - If the class didn't override any of the virtual methods and didn't store any
    additional data, you can just remove the old script from the scene.
  - If the class had additional data, create a new MonoBehaviour and store the
    data in the new MonoBehaviour instead.
  - If the class overrode the virtual methods, create a new MonoBehaviour and
    move the logic to it:
    - Move the InitializeAcademy code to MonoBehaviour.Awake
    - Move the AcademyStep code to MonoBehaviour.FixedUpdate
    - Move the OnDestroy code to MonoBehaviour.OnDestroy.
    - Move the AcademyReset code to a new method and add it to the
      Academy.OnEnvironmentReset action.
- Multiply `max_steps` and `summary_freq` in your `trainer_config.yaml` by the
  number of Agents in the scene.
- Combine curriculum configs into a single file. See
  [the WallJump curricula](https://github.com/Unity-Technologies/ml-agents/blob/0.14.1/config/curricula/wall_jump.yaml) for an example of
  the new curriculum config format. A tool like https://www.json2yaml.com may be
  useful to help with the conversion.
- If you have a model trained which uses RayPerceptionSensor and has non-1.0
  scale in the Agent's transform, it must be retrained.

## Migrating from ML-Agents Toolkit v0.12.0 to v0.13.0

### Important changes

- The low level Python API has changed. You can look at the document
  [Low Level Python API documentation](Python-API.md) for more information. This
  should only affect you if you're writing a custom trainer; if you use
  `mlagents-learn` for training, this should be a transparent change.
  - `reset()` on the Low-Level Python API no longer takes a `train_mode`
    argument. To modify the performance/speed of the engine, you must use an
    `EngineConfigurationChannel`
  - `reset()` on the Low-Level Python API no longer takes a `config` argument.
    `UnityEnvironment` no longer has a `reset_parameters` field. To modify float
    properties in the environment, you must use a `FloatPropertiesChannel`. For
    more information, refer to the
    [Low Level Python API documentation](Python-API.md)
- `CustomResetParameters` are now removed.
- The Academy no longer has a `Training Configuration` nor
  `Inference Configuration` field in the inspector. To modify the configuration
  from the Low-Level Python API, use an `EngineConfigurationChannel`. To modify
  it during training, use the new command line arguments `--width`, `--height`,
  `--quality-level`, `--time-scale` and `--target-frame-rate` in
  `mlagents-learn`.
- The Academy no longer has a `Default Reset Parameters` field in the inspector.
  The Academy class no longer has a `ResetParameters`. To access shared float
  properties with Python, use the new `FloatProperties` field on the Academy.
- Offline Behavioral Cloning has been removed. To learn from demonstrations, use
  the GAIL and Behavioral Cloning features with either PPO or SAC.
- `mlagents.envs` was renamed to `mlagents_envs`. The previous repo layout
  depended on [PEP420](https://www.python.org/dev/peps/pep-0420/), which caused
  problems with some of our tooling such as mypy and pylint.
- The official version of Unity ML-Agents supports is now 2018.4 LTS. If you run
  into issues, please consider deleting your library folder and reponening your
  projects. You will need to install the Barracuda package into your project in
  order to ML-Agents to compile correctly.

### Steps to Migrate

- If you had a custom `Training Configuration` in the Academy inspector, you
  will need to pass your custom configuration at every training run using the
  new command line arguments `--width`, `--height`, `--quality-level`,
  `--time-scale` and `--target-frame-rate`.
- If you were using `--slow` in `mlagents-learn`, you will need to pass your old
  `Inference Configuration` of the Academy inspector with the new command line
  arguments `--width`, `--height`, `--quality-level`, `--time-scale` and
  `--target-frame-rate` instead.
- Any imports from `mlagents.envs` should be replaced with `mlagents_envs`.

## Migrating from ML-Agents Toolkit v0.11.0 to v0.12.0

### Important Changes

- Text actions and observations, and custom action and observation protos have
  been removed.
- RayPerception3D and RayPerception2D are marked deprecated, and will be removed
  in a future release. They can be replaced by RayPerceptionSensorComponent3D
  and RayPerceptionSensorComponent2D.
- The `Use Heuristic` checkbox in Behavior Parameters has been replaced with a
  `Behavior Type` dropdown menu. This has the following options:
  - `Default` corresponds to the previous unchecked behavior, meaning that
    Agents will train if they connect to a python trainer, otherwise they will
    perform inference.
  - `Heuristic Only` means the Agent will always use the `Heuristic()` method.
    This corresponds to having "Use Heuristic" selected in 0.11.0.
  - `Inference Only` means the Agent will always perform inference.
- Barracuda was upgraded to 0.3.2, and it is now installed via the Unity Package
  Manager.

### Steps to Migrate

- We [fixed a bug](https://github.com/Unity-Technologies/ml-agents/pull/2823) in
  `RayPerception3d.Perceive()` that was causing the `endOffset` to be used
  incorrectly. However this may produce different behavior from previous
  versions if you use a non-zero `startOffset`. To reproduce the old behavior,
  you should increase the value of `endOffset` by `startOffset`. You can
  verify your raycasts are performing as expected in scene view using the debug
  rays.
- If you use RayPerception3D, replace it with RayPerceptionSensorComponent3D
  (and similarly for 2D). The settings, such as ray angles and detectable tags,
  are configured on the component now. RayPerception3D would contribute
  `(# of rays) * (# of tags + 2)` to the State Size in Behavior Parameters, but
  this is no longer necessary, so you should reduce the State Size by this
  amount. Making this change will require retraining your model, since the
  observations that RayPerceptionSensorComponent3D produces are different from
  the old behavior.
- If you see messages such as
  `The type or namespace 'Barracuda' could not be found` or
  `The type or namespace 'Google' could not be found`, you will need to
  [install the Barracuda preview package](Installation.md#package-installation).

## Migrating from ML-Agents Toolkit v0.10 to v0.11.0

### Important Changes

- The definition of the gRPC service has changed.
- The online BC training feature has been removed.
- The BroadcastHub has been deprecated. If there is a training Python process,
  all LearningBrains in the scene will automatically be trained. If there is no
  Python process, inference will be used.
- The Brain ScriptableObjects have been deprecated. The Brain Parameters are now
  on the Agent and are referred to as Behavior Parameters. Make sure the
  Behavior Parameters is attached to the Agent GameObject.
- To use a heuristic behavior, implement the `Heuristic()` method in the Agent
  class and check the `use heuristic` checkbox in the Behavior Parameters.
- Several changes were made to the setup for visual observations (i.e. using
  Cameras or RenderTextures):
  - Camera resolutions are no longer stored in the Brain Parameters.
  - AgentParameters no longer stores lists of Cameras and RenderTextures
  - To add visual observations to an Agent, you must now attach a
    CameraSensorComponent or RenderTextureComponent to the agent. The
    corresponding Camera or RenderTexture can be added to these in the editor,
    and the resolution and color/grayscale is configured on the component
    itself.

#### Steps to Migrate

- In order to be able to train, make sure both your ML-Agents Python package and
  UnitySDK code come from the v0.11 release. Training will not work, for
  example, if you update the ML-Agents Python package, and only update the API
  Version in UnitySDK.
- If your Agents used visual observations, you must add a CameraSensorComponent
  corresponding to each old Camera in the Agent's camera list (and similarly for
  RenderTextures).
- Since Brain ScriptableObjects have been removed, you will need to delete all
  the Brain ScriptableObjects from your `Assets` folder. Then, add a
  `Behavior Parameters` component to each `Agent` GameObject. You will then need
  to complete the fields on the new `Behavior Parameters` component with the
  BrainParameters of the old Brain.

## Migrating from ML-Agents Toolkit v0.9 to v0.10

### Important Changes

- We have updated the C# code in our repository to be in line with Unity Coding
  Conventions. This has changed the name of some public facing classes and
  enums.
- The example environments have been updated. If you were using these
  environments to benchmark your training, please note that the resulting
  rewards may be slightly different in v0.10.

#### Steps to Migrate

- `UnitySDK/Assets/ML-Agents/Scripts/Communicator.cs` and its class
  `Communicator` have been renamed to
  `UnitySDK/Assets/ML-Agents/Scripts/ICommunicator.cs` and `ICommunicator`
  respectively.
- The `SpaceType` Enums `discrete`, and `continuous` have been renamed to
  `Discrete` and `Continuous`.
- We have removed the `Done` call as well as the capacity to set `Max Steps` on
  the Academy. Therefore an AcademyReset will never be triggered from C# (only
  from Python). If you want to reset the simulation after a fixed number of
  steps, or when an event in the simulation occurs, we recommend looking at our
  multi-agent example environments (such as FoodCollector). In our examples,
  groups of Agents can be reset through an "Area" that can reset groups of
  Agents.
- The import for `mlagents.envs.UnityEnvironment` was removed. If you are using
  the Python API, change `from mlagents_envs import UnityEnvironment` to
  `from mlagents_envs.environment import UnityEnvironment`.

## Migrating from ML-Agents Toolkit v0.8 to v0.9

### Important Changes

- We have changed the way reward signals (including Curiosity) are defined in
  the `trainer_config.yaml`.
- When using multiple environments, every "step" is recorded in TensorBoard.
- The steps in the command line console corresponds to a single step of a single
  environment. Previously, each step corresponded to one step for all
  environments (i.e., `num_envs` steps).

#### Steps to Migrate

- If you were overriding any of these following parameters in your config file,
  remove them from the top-level config and follow the steps below:
  - `gamma`: Define a new `extrinsic` reward signal and set it's `gamma` to your
    new gamma.
  - `use_curiosity`, `curiosity_strength`, `curiosity_enc_size`: Define a
    `curiosity` reward signal and set its `strength` to `curiosity_strength`,
    and `encoding_size` to `curiosity_enc_size`. Give it the same `gamma` as
    your `extrinsic` signal to mimic previous behavior.
- TensorBoards generated when running multiple environments in v0.8 are not
  comparable to those generated in v0.9 in terms of step count. Multiply your
  v0.8 step count by `num_envs` for an approximate comparison. You may need to
  change `max_steps` in your config as appropriate as well.

## Migrating from ML-Agents Toolkit v0.7 to v0.8

### Important Changes

- We have split the Python packages into two separate packages `ml-agents` and
  `ml-agents-envs`.
- `--worker-id` option of `learn.py` has been removed, use `--base-port` instead
  if you'd like to run multiple instances of `learn.py`.

#### Steps to Migrate

- If you are installing via PyPI, there is no change.
- If you intend to make modifications to `ml-agents` or `ml-agents-envs` please
  check the Installing for Development in the
  [Installation documentation](Installation.md).

## Migrating from ML-Agents Toolkit v0.6 to v0.7

### Important Changes

- We no longer support TFS and are now using the
  [Unity Inference Engine](Unity-Inference-Engine.md)

#### Steps to Migrate

- Make sure to remove the `ENABLE_TENSORFLOW` flag in your Unity Project
  settings

## Migrating from ML-Agents Toolkit v0.5 to v0.6

### Important Changes

- Brains are now Scriptable Objects instead of MonoBehaviors.
- You can no longer modify the type of a Brain. If you want to switch between
  `PlayerBrain` and `LearningBrain` for multiple agents, you will need to assign
  a new Brain to each agent separately. **Note:** You can pass the same Brain to
  multiple agents in a scene by leveraging Unity's prefab system or look for all
  the agents in a scene using the search bar of the `Hierarchy` window with the
  word `Agent`.

- We replaced the **Internal** and **External** Brain with **Learning Brain**.
  When you need to train a model, you need to drag it into the `Broadcast Hub`
  inside the `Academy` and check the `Control` checkbox.
- We removed the `Broadcast` checkbox of the Brain, to use the broadcast
  functionality, you need to drag the Brain into the `Broadcast Hub`.
- When training multiple Brains at the same time, each model is now stored into
  a separate model file rather than in the same file under different graph
  scopes.
- The **Learning Brain** graph scope, placeholder names, output names and custom
  placeholders can no longer be modified.

#### Steps to Migrate

- To update a scene from v0.5 to v0.6, you must:
  - Remove the `Brain` GameObjects in the scene. (Delete all of the Brain
    GameObjects under Academy in the scene.)
  - Create new `Brain` Scriptable Objects using `Assets -> Create -> ML-Agents`
    for each type of the Brain you plan to use, and put the created files under
    a folder called Brains within your project.
  - Edit their `Brain Parameters` to be the same as the parameters used in the
    `Brain` GameObjects.
  - Agents have a `Brain` field in the Inspector, you need to drag the
    appropriate Brain ScriptableObject in it.
  - The Academy has a `Broadcast Hub` field in the inspector, which is list of
    brains used in the scene. To train or control your Brain from the
    `mlagents-learn` Python script, you need to drag the relevant
    `LearningBrain` ScriptableObjects used in your scene into entries into this
    list.

## Migrating from ML-Agents Toolkit v0.4 to v0.5

### Important

- The Unity project `unity-environment` has been renamed `UnitySDK`.
- The `python` folder has been renamed to `ml-agents`. It now contains two
  packages, `mlagents.env` and `mlagents.trainers`. `mlagents.env` can be used
  to interact directly with a Unity environment, while `mlagents.trainers`
  contains the classes for training agents.
- The supported Unity version has changed from `2017.1 or later` to
  `2017.4 or later`. 2017.4 is an LTS (Long Term Support) version that helps us
  maintain good quality and support. Earlier versions of Unity might still work,
  but you may encounter an
  [error](FAQ.md#instance-of-corebraininternal-couldnt-be-created) listed here.

### Unity API

- Discrete Actions now use [branches](https://arxiv.org/abs/1711.08946). You can
  now specify concurrent discrete actions. You will need to update the Brain
  Parameters in the Brain Inspector in all your environments that use discrete
  actions. Refer to the
  [discrete action documentation](Learning-Environment-Design-Agents.md#discrete-action-space)
  for more information.

### Python API

- In order to run a training session, you can now use the command
  `mlagents-learn` instead of `python3 learn.py` after installing the `mlagents`
  packages. This change is documented
  [here](Training-ML-Agents.md#training-with-mlagents-learn). For example, if we
  previously ran

  ```sh
  python3 learn.py 3DBall --train
  ```

  from the `python` subdirectory (which is changed to `ml-agents` subdirectory
  in v0.5), we now run

  ```sh
  mlagents-learn config/trainer_config.yaml --env=3DBall --train
  ```

  from the root directory where we installed the ML-Agents Toolkit.

- It is now required to specify the path to the yaml trainer configuration file
  when running `mlagents-learn`. For an example trainer configuration file, see
  [trainer_config.yaml](https://github.com/Unity-Technologies/ml-agents/blob/0.5.0a/config/trainer_config.yaml). An example of passing a
  trainer configuration to `mlagents-learn` is shown above.
- The environment name is now passed through the `--env` option.
- Curriculum learning has been changed. In summary:
  - Curriculum files for the same environment must now be placed into a folder.
    Each curriculum file should be named after the Brain whose curriculum it
    specifies.
  - `min_lesson_length` now specifies the minimum number of episodes in a lesson
    and affects reward thresholding.
  - It is no longer necessary to specify the `Max Steps` of the Academy to use
    curriculum learning.

## Migrating from ML-Agents Toolkit v0.3 to v0.4

### Unity API

- `using MLAgents;` needs to be added in all of the C# scripts that use
  ML-Agents.

### Python API

- We've changed some of the Python packages dependencies in requirement.txt
  file. Make sure to run `pip3 install -e .` within your `ml-agents/python`
  folder to update your Python packages.

## Migrating from ML-Agents Toolkit v0.2 to v0.3

There are a large number of new features and improvements in the ML-Agents
toolkit v0.3 which change both the training process and Unity API in ways which
will cause incompatibilities with environments made using older versions. This
page is designed to highlight those changes for users familiar with v0.1 or v0.2
in order to ensure a smooth transition.

### Important

- The ML-Agents Toolkit is no longer compatible with Python 2.

### Python Training

- The training script `ppo.py` and `PPO.ipynb` Python notebook have been
  replaced with a single `learn.py` script as the launching point for training
  with ML-Agents. For more information on using `learn.py`, see
  [here](Training-ML-Agents.md#training-with-mlagents-learn).
- Hyperparameters for training Brains are now stored in the
  `trainer_config.yaml` file. For more information on using this file, see
  [here](Training-ML-Agents.md#training-configurations).

### Unity API

- Modifications to an Agent's rewards must now be done using either
  `AddReward()` or `SetReward()`.
- Setting an Agent to done now requires the use of the `Done()` method.
- `CollectStates()` has been replaced by `CollectObservations()`, which now no
  longer returns a list of floats.
- To collect observations, call `AddVectorObs()` within `CollectObservations()`.
  Note that you can call `AddVectorObs()` with floats, integers, lists and
  arrays of floats, Vector3 and Quaternions.
- `AgentStep()` has been replaced by `AgentAction()`.
- `WaitTime()` has been removed.
- The `Frame Skip` field of the Academy is replaced by the Agent's
  `Decision Frequency` field, enabling the Agent to make decisions at different
  frequencies.
- The names of the inputs in the Internal Brain have been changed. You must
  replace `state` with `vector_observation` and `observation` with
  `visual_observation`. In addition, you must remove the `epsilon` placeholder.

### Semantics

In order to more closely align with the terminology used in the Reinforcement
Learning field, and to be more descriptive, we have changed the names of some of
the concepts used in ML-Agents. The changes are highlighted in the table below.

| Old - v0.2 and earlier | New - v0.3 and later |
| ---------------------- | -------------------- |
| State                  | Vector Observation   |
| Observation            | Visual Observation   |
| Action                 | Vector Action        |
| N/A                    | Text Observation     |
| N/A                    | Text Action          |

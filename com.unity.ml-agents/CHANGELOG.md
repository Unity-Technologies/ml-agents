# Changelog
All notable changes to this package will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).


## [Unreleased]
### Major Changes

### Minor Changes
 - Format of console output has changed slightly and now matches the name of the model/summary directory. (#3630, #3616)
 - Raise the wall in CrawlerStatic scene to prevent Agent from falling off. (#3650)
 - Renamed 'Generalization' feature to 'Environment Parameter Randomization'.

## [0.15.0-preview] - 2020-03-18
### Major Changes
 - `Agent.CollectObservations` now takes a VectorSensor argument. (#3352, #3389)
 - Added `Agent.CollectDiscreteActionMasks` virtual method with a `DiscreteActionMasker` argument to specify which discrete actions are unavailable to the Agent. (#3525)
 - Beta support for ONNX export was added. If the `tf2onnx` python package is installed, models will be saved to `.onnx` as well as `.nn` format.
 Note that Barracuda 0.6.0 or later is required to import the `.onnx` files properly
 - Multi-GPU training and the `--multi-gpu` option has been removed temporarily. (#3345)
 - All Sensor related code has been moved to the namespace `MLAgents.Sensors`.
 - All SideChannel related code has been moved to the namespace `MLAgents.SideChannels`.
 - `BrainParameters` and `SpaceType` have been removed from the public API
 - `BehaviorParameters` have been removed from the public API.
 - The following methods in the `Agent` class have been deprecated and will be removed in a later release:
   - `InitializeAgent()` was renamed to `Initialize()`
   - `AgentAction()` was renamed to `OnActionReceived()`
   - `AgentReset()` was renamed to `OnEpisodeBegin()`
   - `Done()` was renamed to `EndEpisode()`
   - `GiveModel()` was renamed to `SetModel()`

### Minor Changes
 - Monitor.cs was moved to Examples. (#3372)
 - Automatic stepping for Academy is now controlled from the AutomaticSteppingEnabled property. (#3376)
 - The GetEpisodeCount, GetStepCount, GetTotalStepCount and methods of Academy were changed to EpisodeCount, StepCount, TotalStepCount properties respectively. (#3376)
 - Several classes were changed from public to internal visibility. (#3390)
 - Academy.RegisterSideChannel and UnregisterSideChannel methods were added. (#3391)
 - A tutorial on adding custom SideChannels was added (#3391)
 - The stepping logic for the Agent and the Academy has been simplified (#3448)
 - Update Barracuda to 0.6.1-preview
 * The interface for `RayPerceptionSensor.PerceiveStatic()` was changed to take an input class and write to an output class, and the method was renamed to `Perceive()`.
 - The checkpoint file suffix was changed from `.cptk` to `.ckpt` (#3470)
 - The command-line argument used to determine the port that an environment will listen on was changed from `--port` to `--mlagents-port`.
 - `DemonstrationRecorder` can now record observations outside of the editor.
 - `DemonstrationRecorder` now has an optional path for the demonstrations. This will default to `Application.dataPath` if not set.
 - `DemonstrationStore` was changed to accept a `Stream` for its constructor, and was renamed to `DemonstrationWriter`
 - The method `GetStepCount()` on the Agent class has been replaced with the property getter `StepCount`
 - `RayPerceptionSensorComponent` and related classes now display the debug gizmos whenever the Agent is selected (not just Play mode).
 - Most fields on `RayPerceptionSensorComponent` can now be changed while the editor is in Play mode. The exceptions to this are fields that affect the number of observations.
 - Most fields on `CameraSensorComponent` and `RenderTextureSensorComponent` were changed to private and replaced by properties with the same name.
 - Unused static methods from the `Utilities` class (ShiftLeft, ReplaceRange, AddRangeNoAlloc, and GetSensorFloatObservationSize) were removed.
 - The `Agent` class is no longer abstract.
 - SensorBase was moved out of the package and into the Examples directory.
 - `AgentInfo.actionMasks` has been renamed to `AgentInfo.discreteActionMasks`.
 - `DecisionRequester` has been made internal (you can still use the DecisionRequesterComponent from the inspector). `RepeatAction` was renamed `TakeActionsBetweenDecisions` for clarity. (#3555)
 - The `IFloatProperties` interface has been removed.
 - Fix #3579.
 - Improved inference performance for models with multiple action branches. (#3598)
 - Fixed an issue when using GAIL with less than `batch_size` number of demonstrations. (#3591)
 - The interfaces to the `SideChannel` classes (on C# and python) have changed to use new  `IncomingMessage` and `OutgoingMessage` classes. These should make reading and writing data to the channel easier. (#3596)
 - Updated the ExpertPyramid.demo example demonstration file (#3613)
 - Updated project version for example environments to 2018.4.18f1. (#3618)
 - Changed the Product Name in the example environments to remove spaces, so that the default build executable file doesn't contain spaces. (#3612)

## [0.14.1-preview] - 2020-02-25

### Bug Fixes
- Fixed an issue which caused self-play training sessions to consume a lot of memory. (#3451)
- Fixed an IndexError when using GAIL or behavioral cloning with demonstrations recorded with 0.14.0 or later (#3464)
- Updated the `gail_config.yaml` to work with per-Agent steps (#3475)
- Fixed demonstration recording of experiences when the Agent is done. (#3463)
- Fixed a bug with the rewards of multiple Agents in the gym interface (#3471, #3496)


## [0.14.0-preview] - 2020-02-13

### Major Changes
- A new self-play mechanism for training agents in adversarial scenarios was added (#3194)
- Tennis and Soccer environments were refactored to enable training with self-play (#3194, #3331)
- UnitySDK folder was split into a Unity Package (com.unity.ml-agents) and our examples were moved to the Project folder (#3267)
- Academy is now a singleton and is no longer abstract (#3210, #3184)
- In order to reduce the size of the API, several classes and methods were marked as internal or private. Some public fields on the Agent were trimmed (#3342, #3353, #3269)
- Decision Period and on-demand decision checkboxes were removed from the Agent. on-demand decision is now the default (#3243)
- Calling Done() on the Agent will reset it immediately and call the AgentReset virtual method (#3291, #3242)
- The "Reset on Done" setting in AgentParameters was removed; this is now always true. AgentOnDone virtual method on the Agent was removed (#3311, #3222)
- Trainer steps are now counted per-Agent, not per-environment as in previous versions. For instance, if you have 10 Agents in the scene, 20 environment steps now correspond to 200 steps as printed in the terminal and in Tensorboard (#3113)

### Minor Changes
- Barracuda was updated to 0.5.0-preview (#3329)
- --num-runs option was removed from mlagents-learn (#3155)
- Curriculum config files are now YAML formatted and all curricula for a training run are combined into a single file (#3186)
- ML-Agents components, such as BehaviorParameters and various Sensor implementations, now appear in the Components menu (#3231)
- Exceptions are now raised in Unity (in debug mode only) if NaN observations or rewards are passed (#3221)
- RayPerception MonoBehavior, which was previously deprecated, was removed (#3304)
- Uncompressed visual (i.e. 3d float arrays) observations are now supported. CameraSensorComponent and RenderTextureSensor now have an option to write uncompressed observations (#3148)
- Agent’s handling of observations during training was improved so that an extra copy of the observations is no longer maintained (#3229)
- Error message for missing trainer config files was improved to include the absolute path (#3230)
- Support for 2017.4 LTS was dropped (#3121, #3168)
- Some documentation improvements were made (#3296, #3292, #3295, #3281)

### Bug Fixes
- Numpy warning when stats don’t exist (#3251)
- A bug that caused RayPerceptionSensor to behave inconsistently with transforms that have non-1 scale was fixed (#3321)
- Some small bugfixes to tensorflow_to_barracuda.py were backported from the barracuda release (#3341)
- Base port in the jupyter notebook example was updated to use the same port that the editor uses (#3283)


## [0.13.0-preview] - 2020-01-24

### This is the first release of *Unity Package ML-Agents*.

*Short description of this release*

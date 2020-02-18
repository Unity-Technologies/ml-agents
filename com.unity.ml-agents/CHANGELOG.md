# Changelog
All notable changes to this package will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Major Changes
 - Agent.CollectObservations now takes a VectorSensor argument. It was also overloaded to optionally take an ActionMasker argument. (#3352, #3389)

### Minor Changes
 - Monitor.cs was moved to Examples. (#3372)
 - Automatic stepping for Academy is now controlled from the AutomaticSteppingEnabled property. (#3376)
 - The GetEpisodeCount, GetStepCount, GetTotalStepCount and methods of Academy were changed to EpisodeCount, StepCount, TotalStepCount properties respectively. (#3376)
 - Several classes were changed from public to internal visibility. (#3390)
 - Academy.RegisterSideChannel and UnregisterSideChannel methods were added. (#3391)
 - A tutorial on adding custom SideChannels was added (#3391)
 - Update Barracuda to 0.6.0-preview

### Bugfixes
- Fixed an issue which caused self-play training sessions to consume a lot of memory. (#3451)

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

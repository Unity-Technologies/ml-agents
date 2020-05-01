# About ML-Agents package (`com.unity.ml-agents`)

The Unity ML-Agents package contains the C# SDK for the [Unity ML-Agents
Toolkit].

The package allows you to convert any Unity scene to into a learning environment
and train character behaviors using a variety of machine learning algorithms.
Additionally, it allows you to embed these trained behaviors back into Unity
scenes to control your characters. More specifically, the package provides the
following core functionalities:

- Define Agents: entities, or characters, whose behavior will be learned. Agents
  are entities that generate observations (through sensors), take actions, and
  receive rewards from the environment.
- Define Behaviors: entities that specifiy how an agent should act. Multiple
  agents can share the same Behavior and a scene may have multiple Behaviors.
- Record demonstrations of an agent within the Editor. You can use
  demonstrations to help train a behavior for that agent.
- Embedding a trained behavior into the scene via the [Unity Inference Engine].
  Embedded behaviors allow you to switch an Agent between learning and
  inference.

Note that the _ML-Agents_ package does not contain the machine learning
algorithms for training behaviors. The _ML-Agents_ package only supports
instrumenting a Unity scene, setting it up for training, and then embedding the
trained model back into your Unity scene. The machine learning algorithms that
orchestrate training are part of the companion [Python package].

## Package contents

The following table describes the package folder structure:

| **Location**     | **Description**                                                        |
| ---------------- | ---------------------------------------------------------------------- |
| _Documentation~_ | Contains the documentation for the Unity package.                      |
| _Editor_         | Contains utilities for Editor windows and drawers.                     |
| _Plugins_        | Contains third-party DLLs.                                             |
| _Runtime_        | Contains core C# APIs for integrating ML-Agents into your Unity scene. |
| _Tests_          | Contains the unit tests for the package.                               |

<a name="Installation"></a>

## Installation

To install this _ML-Agents_ package, follow the instructions in the [Package
Manager documentation].

To install the companion Python package to enable training behaviors, follow the
[installation instructions] on our [GitHub repository].

## Requirements

This version of the Unity ML-Agents package is compatible with the following
versions of the Unity Editor:

- 2018.4 and later

## Known Limitations

### Training

Training is limited to the Unity Editor and Standalone builds on Windows, MacOS,
and Linux with the Mono scripting backend. Currently, training does not work
with the IL2CPP scripting backend. Your environment will default to inference
mode if training is not supported or is not currently running.

### Inference

Inference is executed via the
[Unity Inference Engine](https://docs.unity3d.com/Packages/com.unity.barracuda@latest/index.html).

**CPU**

All platforms supported.

**GPU**

All platforms supported except:

- WebGL and GLES 3/2 on Android / iPhone

**NOTE:** Mobile platform support includes:

- Vulkan for Android
- Metal for iOS.

### Headless Mode

If you enable Headless mode, you will not be able to collect visual observations
from your agents.

### Rendering Speed and Synchronization

Currently the speed of the game physics can only be increased to 100x real-time.
The Academy also moves in time with FixedUpdate() rather than Update(), so game
behavior implemented in Update() may be out of sync with the agent decision
making. See [Execution Order of Event Functions] for more information.

You can control the frequency of Academy stepping by calling
`Academy.Instance.DisableAutomaticStepping()`, and then calling
`Academy.Instance.EnvironmentStep()`

### Unity Inference Engine Models

Currently, only models created with our trainers are supported for running
ML-Agents with a neural network behavior.

## Helpful links

If you are new to the Unity ML-Agents package, or have a question after reading
the documentation, you can checkout our [GitHUb Repository], which also includes
a number of ways to [connect with us] including our [ML-Agents Forum].

[unity ML-Agents Toolkit]: https://github.com/Unity-Technologies/ml-agents
[unity inference engine]: https://docs.unity3d.com/Packages/com.unity.barracuda@latest/index.html
[package manager documentation]: https://docs.unity3d.com/Manual/upm-ui-install.html
[installation instructions]: https://github.com/Unity-Technologies/ml-agents/blob/release_1_docs/docs/Installation.md
[github repository]: https://github.com/Unity-Technologies/ml-agents
[python package]: https://github.com/Unity-Technologies/ml-agents
[execution order of event functions]: https://docs.unity3d.com/Manual/ExecutionOrder.html
[connect with us]: https://github.com/Unity-Technologies/ml-agents#community-and-feedback
[ml-agents forum]: https://forum.unity.com/forums/ml-agents.453/

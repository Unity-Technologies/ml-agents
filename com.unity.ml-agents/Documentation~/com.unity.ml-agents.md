# About ML-Agents package (`com.unity.ml-agents`)

The Unity ML-Agents package contains the C# SDK for the
[Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents).

The package provides the ability for any Unity scene to be converted into a learning
environment where character behaviors can be trained using a variety of machine learning
algorithms. Additionally, it enables any trained behavior to be embedded back into the Unity
scene. More specifically, the package provides the following core functionalities:
* Define Agents: entities whose behavior will be learned. Agents are entities
that generate observations (through sensors), take actions and receive rewards from
the environment.
* Define Behaviors: entities that specifiy how an agent should act. Multiple agents can
share the same Behavior and a scene may have multiple Behaviors.
* Record demonstrations of an agent within the Editor. These demonstrations can be
valuable to train a behavior for that agent.
* Embedding a trained behavior into the scene via the
[Unity Inference Engine](https://docs.unity3d.com/Packages/com.unity.barracuda@latest/index.html).
Thus an Agent can switch from a learning behavior to an inference behavior.

Note that this package does not contain the machine learning algorithms for training
behaviors. It relies on a Python package to orchestrate the training. This package
only enables instrumenting a Unity scene and setting it up for training, and then
embedding the trained model back into your Unity scene.

## Preview package
This package is available as a preview, so it is not ready for production use.
The features and documentation in this package might change before it is verified for release.


## Package contents

The following table describes the package folder structure:

|**Location**|**Description**|
|---|---|
|*Documentation~*|Contains the documentation for the Unity package.|
|*Editor*|Contains utilities for Editor windows and drawers.|
|*Plugins*|Contains third-party DLLs.|
|*Runtime*|Contains core C# APIs for integrating ML-Agents into your Unity scene. |
|*Tests*|Contains the unit tests for the package.|

<a name="Installation"></a>

## Installation

To install this package, follow the instructions in the
[Package Manager documentation](https://docs.unity3d.com/Manual/upm-ui-install.html).

To install the Python package to enable training behaviors, follow the instructions on our
[GitHub repository](https://github.com/Unity-Technologies/ml-agents/blob/latest_release/docs/Installation.md).

## Requirements

This version of the Unity ML-Agents package is compatible with the following versions of the Unity Editor:

* 2018.4 and later (recommended)

## Known limitations

### Headless Mode

If you enable Headless mode, you will not be able to collect visual observations
from your agents.

### Rendering Speed and Synchronization

Currently the speed of the game physics can only be increased to 100x real-time.
The Academy also moves in time with FixedUpdate() rather than Update(), so game
behavior implemented in Update() may be out of sync with the agent decision
making. See
[Execution Order of Event Functions](https://docs.unity3d.com/Manual/ExecutionOrder.html)
for more information.

You can control the frequency of Academy stepping by calling
`Academy.Instance.DisableAutomaticStepping()`, and then calling
`Academy.Instance.EnvironmentStep()`

### Unity Inference Engine Models
Currently, only models created with our trainers are supported for running
ML-Agents with a neural network behavior.


## Helpful links

If you are new to the Unity ML-Agents package, or have a question after reading
the documentation, you can checkout our
[GitHUb Repository](https://github.com/Unity-Technologies/ml-agents), which
also includes a number of ways to
[connect with us](https://github.com/Unity-Technologies/ml-agents#community-and-feedback)
including our [ML-Agents Forum](https://forum.unity.com/forums/ml-agents.453/).


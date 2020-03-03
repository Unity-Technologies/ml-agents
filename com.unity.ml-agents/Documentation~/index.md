# About Unity MLAgents package

The MLAgents package contains the C# SDK for the 
[Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents). 

The package provides the ability for any Unity scene to be converted into a learning environment where
character behaviors can be trained using a variety of machine learning algorithms. Additionally, it enables
any trained behavior to embedded back into the Unity scene. More specifically, the package provides the
following core functionalities:
* Define Agents: entities whose behavior will be learned. Agents are entities
that generate observations (through sensors), take actions and receive rewards from the environment.
* Define Brains: entities that specifiy a behavior. Multiple agents can share the same Brain and a scene may
have multiple Brains.
* Record demonstrations of an agent within the Editor. These demonstrations can be valuable to train a behavior for that agent.
* Embedding a trained behavior into the scene. This an Agent can switch from a learning brain to an inference brain. 

Note that this package does not contain the machine learning algorithms for training behaviors. It relies on a Python
package to orchestrate the training. This package only enables instrumenting a Unity scene and setting it up for training.

## Preview package
This package is available as a preview, so it is not ready for production use. The features and documentation in this package might change before it is verified for release.


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

To install this package, follow the instructions in the [Package Manager documentation](https://docs.unity3d.com/Manual/upm-ui-install.html).


## Requirements

This version of MLAgents is compatible with the following versions of the Unity Editor:

* 2018.4 and later (recommended)


To use this package, you must have the following 3rd party products:

* &lt;TBC&gt;&trade;


## Known limitations

There are currently no known limitations.

---

*The Unity Recorder version 1.0 has the following limitations:*

* *The Unity Recorder does not support sound.*
* *The Recorder window and Recorder properties are not available in standalone Players.*
* *MP4 encoding is only available on Windows.*

---

## Helpful links

If you are new to MLAgents, or have a question after reading the documentation, you can
checkout our [GitHUb Repository](https://github.com/Unity-Technologies/ml-agents), which also
includes a number of ways to (connect with us)[https://github.com/Unity-Technologies/ml-agents#community-and-feedback].


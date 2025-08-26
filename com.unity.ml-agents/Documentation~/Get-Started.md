# Get started
The ML-Agents Toolkit contains several main components:

- Unity package `com.unity.ml-agents` contains the Unity C# SDK that will be integrated into your Unity project.
- Two Python packages:
  - `mlagents` contains the machine learning algorithms that enables you to train behaviors in your Unity scene. Most users of ML-Agents will only need to directly install `mlagents`.
  - `mlagents_envs` contains a set of Python APIs to interact with a Unity scene. It is a foundational layer that facilitates data messaging between Unity scene and the Python machine learning algorithms. Consequently, `mlagents` depends on `mlagents_envs`.
- Unity [Project](https://github.com/Unity-Technologies/ml-agents/tree/main/Project/Assets/ML-Agents/Examples) that contains several [example environments](Learning-Environment-Examples.md) that highlight the various features of the toolkit to help you get started.

Use the following topics to get started with ML-Agents.

| **Section**                                                   | **Description**                    |
|---------------------------------------------------------------|------------------------------------|
| [Install ML-Agents](Installation.md)                          | Install ML-Agents.                 |
| [Sample: Running an Example Environment](Sample.md)           | Learn how to run a sample project. |
| [More Example Environments](Learning-Environment-Examples.md) | Explore 17+ sample environments.   |

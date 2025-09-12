# ML-Agents Overview

<img src="images/image-banner.png" align="middle" width="3000"/>

ML-Agents enable games and simulations to serve as environments for training intelligent agents in Unity. Training can be done with reinforcement learning, imitation learning, neuroevolution, or any other methods. Trained agents can be used for many use cases, including controlling NPC behavior (in a variety of settings such as multi-agent and adversarial), automated testing of game builds and evaluating different game design decisions pre-release.

This documentation contains comprehensive instructions about [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents) including the C# package, along with detailed training guides and Python API references. Note that the C# package does not contain the machine learning algorithms for training behaviors. The C# package only supports instrumenting a Unity scene, setting it up for training, and then embedding the trained model back into your Unity scene. The machine learning algorithms that orchestrate training are part of the companion Python package.

## Documentation structure

| **Section**                                                         | **Description**                                                                                                                                                      |
|---------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [ML-Agents Theory](ML-Agents-Overview.md)                           | Learn about core concepts of ML-Agents.                                                                                                                              |
| [Get started](Get-Started.md)                                       | Learn how to install ML-Agents and explore samples.                                                                                                                  |
| [Learning Environments and Agents](Learning-Environments-Agents.md) | Learn about Environments, Agents, creating environments, and using executable builds.                                                                                |
| [Training](Training.md)                                             | Training workflow, config file, monitoring tools, custom plugins, and profiling.                                                                                     |
| [Python APIs](Python-APIs.md)                                       | Gym, PettingZoo, low-level interfaces, and trainer documentation.                                                                                                    |
| [Python Tutorial with Google Colab](Tutorial-Colab.md)              | Interactive tutorials for using ML-Agents with Google Colab.                                                                                                         |
| [Advanced Features](Advanced-Features.md)                           | Custom sensors, side channels, package settings, environment registry, input system integrations, and game integrations (e.g., [Match-3](Integrations-Match3.md)).   |
| [Cloud & Deployment](Cloud-Deployment.md)                           | Legacy cloud deployment guides (deprecated).                                                                                                                         |
| [Reference & Support](Reference-Support.md)                         | FAQ, troubleshooting, and migration guides.                                                                                                                          |
| [Background](Background.md)                                         | Machine Learning, Unity, PyTorch fundamentals, virtual environments, and ELO rating systems.                                                                         |

## Capabilities
The package allows you to convert any Unity scene into a learning environment and train character behaviors using a variety of machine-learning algorithms. Additionally, it allows you to embed these trained behaviors back into Unity scenes to control your characters. More specifically, the package provides the following core functionalities:

* Define Agents: entities, or characters, whose behavior will be learned. Agents are entities that generate observations (through sensors), take actions, and receive rewards from the environment.
* Define Behaviors: entities that specify how an agent should act. Multiple agents can share the same Behavior and a scene may have multiple Behaviors.
* Record demonstrations: To show the behaviors of an agent within the Editor. You can use demonstrations to help train a behavior for that agent.
* Embed a trained behavior (aka: run your ML model) in the scene via the [Inference Engine](https://docs.unity3d.com/Packages/com.unity.ai.inference@latest). Embedded behaviors allow you to switch an Agent between learning and inference.

## Community and Feedback

The ML-Agents Toolkit is an open-source project, and we encourage and welcome contributions. If you wish to contribute, be sure to review our [contribution guidelines](CONTRIBUTING.md) and [code of conduct](https://github.com/Unity-Technologies/ml-agents/blob/release/4.0.0/CODE_OF_CONDUCT.md).

For problems with the installation and setup of the ML-Agents Toolkit, or discussions about how to best setup or train your agents, please create a new thread on the [Unity ML-Agents discussion forum](https://discussions.unity.com/tag/ml-agents). Be sure to include as many details as possible to help others assist you effectively. If you run into any other problems using the ML-Agents Toolkit or have a specific feature request, please [submit a GitHub issue](https://github.com/Unity-Technologies/ml-agents/issues).

Please tell us which samples you would like to see shipped with the ML-Agents Unity package by replying to [this discussion thread](https://discussions.unity.com/t/help-shape-the-future-of-ml-agents/1661019).

## Privacy
In order to improve the developer experience for Unity ML-Agents Toolkit, we have added in-editor analytics. Please refer to "Information that is passively collected by Unity" in the [Unity Privacy Policy](https://unity3d.com/legal/privacy-policy).

## Additional Resources

* [GitHub repository](https://github.com/Unity-Technologies/ml-agents)
* [Unity Discussions](https://discussions.unity.com/tag/ml-agents)
* [ML-Agents tutorials by CodeMonkeyUnity](https://www.youtube.com/playlist?list=PLzDRvYVwl53vehwiN_odYJkPBzcqFw110)
* [Introduction to ML-Agents by Huggingface](https://huggingface.co/learn/deep-rl-course/en/unit5/introduction)
* [Community created ML-Agents projects](https://discussions.unity.com/t/post-your-ml-agents-project/816756)
* [ML-Agents models on Huggingface](https://huggingface.co/models?library=ml-agents)
* [Blog posts](Blog-posts.md)
* [Discord](https://discord.com/channels/489222168727519232/1202574086115557446)

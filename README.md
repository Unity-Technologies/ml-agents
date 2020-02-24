<img src="docs/images/unity-wide.png" align="middle" width="3000"/>

<img src="docs/images/image-banner.png" align="middle" width="3000"/>

# Unity ML-Agents Toolkit (Beta)
[![docs badge](https://img.shields.io/badge/docs-reference-blue.svg)](https://github.com/Unity-Technologies/ml-agents/tree/latest_release/docs/)
[![license badge](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

([latest release](https://github.com/Unity-Technologies/ml-agents/releases/tag/latest_release))
([all releases](https://github.com/Unity-Technologies/ml-agents/releases))

**The Unity Machine Learning Agents Toolkit** (ML-Agents) is an open-source
Unity plugin that enables games and simulations to serve as environments for
training intelligent agents. Agents can be trained using reinforcement learning,
imitation learning, neuroevolution, or other machine learning methods through a
simple-to-use Python API. We also provide implementations (based on TensorFlow)
of state-of-the-art algorithms to enable game developers and hobbyists to easily
train intelligent agents for 2D, 3D and VR/AR games. These trained agents can be
used for multiple purposes, including controlling NPC behavior (in a variety of
settings such as multi-agent and adversarial), automated testing of game builds
and evaluating different game design decisions pre-release. The ML-Agents
toolkit is mutually beneficial for both game developers and AI researchers as it
provides a central platform where advances in AI can be evaluated on Unity’s
rich environments and then made accessible to the wider research and game
developer communities.

## Features

* Unity environment control from Python
* 10+ sample Unity environments
* Two deep reinforcement learning algorithms,
[Proximal Policy Optimization](https://github.com/Unity-Technologies/ml-agents/tree/latest_release/docs/Training-PPO.md)
 (PPO) and [Soft Actor-Critic](https://github.com/Unity-Technologies/ml-agents/tree/latest_release/docs/Training-SAC.md)
 (SAC)
* Support for multiple environment configurations and training scenarios
* Self-play mechanism for training agents in adversarial scenarios
* Train memory-enhanced agents using deep reinforcement learning
* Easily definable Curriculum Learning and Generalization scenarios
* Built-in support for Imitation Learning
* Flexible agent control with On Demand Decision Making
* Visualizing network outputs within the environment
* Simplified set-up with Docker
* Wrap learning environments as a gym
* Utilizes the Unity Inference Engine
* Train using concurrent Unity environment instances

## Documentation

* For more information, in addition to installation and usage instructions, see
  the [documentation for the latest release](https://github.com/Unity-Technologies/ml-agents/tree/latest_release/docs/Readme.md).
* If you are a researcher interested in a discussion of Unity as an AI platform, see a pre-print of our [reference paper on Unity and the ML-Agents Toolkit](https://arxiv.org/abs/1809.02627). Also, see below for instructions on citing this paper.
* If you have used an earlier version of the ML-Agents toolkit, we strongly
  recommend our [guide on migrating from earlier versions](docs/Migrating.md).

## Additional Resources

We have published a series of blog posts that are relevant for ML-Agents:

* Overviewing reinforcement learning concepts
  ([multi-armed bandit](https://blogs.unity3d.com/2017/06/26/unity-ai-themed-blog-entries/)
  and
  [Q-learning](https://blogs.unity3d.com/2017/08/22/unity-ai-reinforcement-learning-with-q-learning/))
* [Using Machine Learning Agents in a real game: a beginner’s guide](https://blogs.unity3d.com/2017/12/11/using-machine-learning-agents-in-a-real-game-a-beginners-guide/)
* [Post](https://blogs.unity3d.com/2018/02/28/introducing-the-winners-of-the-first-ml-agents-challenge/)
  announcing the winners of our
  [first ML-Agents Challenge](https://connect.unity.com/challenges/ml-agents-1)
* [Post](https://blogs.unity3d.com/2018/01/23/designing-safer-cities-through-simulations/)
  overviewing how Unity can be leveraged as a simulator to design safer cities.

In addition to our own documentation, here are some additional, relevant articles:

* [A Game Developer Learns Machine Learning](https://mikecann.co.uk/machine-learning/a-game-developer-learns-machine-learning-intent/)
* [Explore Unity Technologies ML-Agents Exclusively on Intel Architecture](https://software.intel.com/en-us/articles/explore-unity-technologies-ml-agents-exclusively-on-intel-architecture)
* [ML-Agents Penguins tutorial](https://learn.unity.com/project/ml-agents-penguins)

## Community and Feedback

The ML-Agents toolkit is an open-source project and we encourage and welcome
contributions. If you wish to contribute, be sure to review our
[contribution guidelines](com.unity.ml-agents/CONTRIBUTING.md) and
[code of conduct](CODE_OF_CONDUCT.md).

For problems with the installation and setup of the the ML-Agents toolkit, or
discussions about how to best setup or train your agents, please create a new
thread on the [Unity ML-Agents forum](https://forum.unity.com/forums/ml-agents.453/)
and make sure to include as much detail as possible.
If you run into any other problems using the ML-Agents toolkit, or have a specific
feature requests, please [submit a GitHub issue](https://github.com/Unity-Technologies/ml-agents/issues).

Your opinion matters a great deal to us. Only by hearing your thoughts on the Unity ML-Agents Toolkit can we continue
to improve and grow. Please take a few minutes to [let us know about it](https://github.com/Unity-Technologies/ml-agents/issues/1454).

For any other questions or feedback, connect directly with the ML-Agents
team at ml-agents@unity3d.com.

## Releases
The latest release is 0.14.0. Previous releases can be found below:

| **Version** | **Source** | **Documentation** | **Download** |
|:-------:|:------:|:-------------:|:-------:|
| **0.13.1**  | [source](https://github.com/Unity-Technologies/ml-agents/tree/0.13.1) |  [docs](https://github.com/Unity-Technologies/ml-agents/tree/0.13.1/docs) | [download](https://github.com/Unity-Technologies/ml-agents/archive/0.13.1.zip) |
| **0.13.0**  | [source](https://github.com/Unity-Technologies/ml-agents/tree/0.13.0) |  [docs](https://github.com/Unity-Technologies/ml-agents/tree/0.13.0/docs) | [download](https://github.com/Unity-Technologies/ml-agents/archive/0.13.0.zip) |
| **0.12.1**  | [source](https://github.com/Unity-Technologies/ml-agents/tree/0.12.1) |  [docs](https://github.com/Unity-Technologies/ml-agents/tree/0.12.1/docs) | [download](https://github.com/Unity-Technologies/ml-agents/archive/0.12.1.zip) |
| **0.12.0**  | [source](https://github.com/Unity-Technologies/ml-agents/tree/0.12.0) |  [docs](https://github.com/Unity-Technologies/ml-agents/tree/0.12.0/docs) | [download](https://github.com/Unity-Technologies/ml-agents/archive/0.12.0.zip) |
| **0.11.0**  | [source](https://github.com/Unity-Technologies/ml-agents/tree/0.11.0) |  [docs](https://github.com/Unity-Technologies/ml-agents/tree/0.11.0/docs) | [download](https://github.com/Unity-Technologies/ml-agents/archive/0.11.0.zip) |
| **0.10.1**  | [source](https://github.com/Unity-Technologies/ml-agents/tree/0.10.1) |  [docs](https://github.com/Unity-Technologies/ml-agents/tree/0.10.1/docs) | [download](https://github.com/Unity-Technologies/ml-agents/archive/0.10.1.zip) |
| **0.10.0**  | [source](https://github.com/Unity-Technologies/ml-agents/tree/0.10.0) |  [docs](https://github.com/Unity-Technologies/ml-agents/tree/0.10.0/docs) | [download](https://github.com/Unity-Technologies/ml-agents/archive/0.10.0.zip) |

See the [GitHub releases](https://github.com/Unity-Technologies/ml-agents/releases) for more details of the changes
between versions.

Please note that the `master` branch is under active development, so the documentation there may differ from the code
of a previous release. Always use the documentation that corresponds to the release version you're using.

## License

[Apache License 2.0](LICENSE)

## Citation

If you use Unity or the ML-Agents Toolkit to conduct research, we ask that you cite the following paper as a reference:

Juliani, A., Berges, V., Vckay, E., Gao, Y., Henry, H., Mattar, M., Lange, D. (2018). Unity: A General Platform for Intelligent Agents. *arXiv preprint arXiv:1809.02627.* https://github.com/Unity-Technologies/ml-agents.

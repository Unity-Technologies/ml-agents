<img src="docs/images/image-banner.png" align="middle" width="3000"/>

# Unity ML-Agents Toolkit

[![docs badge](https://img.shields.io/badge/docs-reference-blue.svg)](https://github.com/Unity-Technologies/ml-agents/tree/release_12_docs/docs/)

[![license badge](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

([latest release](https://github.com/Unity-Technologies/ml-agents/releases/tag/latest_release))
([all releases](https://github.com/Unity-Technologies/ml-agents/releases))

**The Unity Machine Learning Agents Toolkit** (ML-Agents) is an open-source
project that enables games and simulations to serve as environments for
training intelligent agents. We provide implementations (based on PyTorch)
of state-of-the-art algorithms to enable game developers and hobbyists to easily
train intelligent agents for 2D, 3D and VR/AR games. Researchers can also use the
provided simple-to-use Python API to train Agents using reinforcement learning,
imitation learning, neuroevolution, or any other methods. These trained agents can be
used for multiple purposes, including controlling NPC behavior (in a variety of
settings such as multi-agent and adversarial), automated testing of game builds
and evaluating different game design decisions pre-release. The ML-Agents
Toolkit is mutually beneficial for both game developers and AI researchers as it
provides a central platform where advances in AI can be evaluated on Unity’s
rich environments and then made accessible to the wider research and game
developer communities.

## Features

- 15+ [example Unity environments](docs/Learning-Environment-Examples.md)
- Support for multiple environment configurations and training scenarios
- Flexible Unity SDK that can be integrated into your game or custom Unity scene
- Training using two deep reinforcement learning algorithms, Proximal Policy
  Optimization (PPO) and Soft Actor-Critic (SAC)
- Built-in support for Imitation Learning through Behavioral Cloning or
  Generative Adversarial Imitation Learning
- Self-play mechanism for training agents in adversarial scenarios
- Easily definable Curriculum Learning scenarios for complex tasks
- Train robust agents using environment randomization
- Flexible agent control with On Demand Decision Making
- Train using multiple concurrent Unity environment instances
- Utilizes the [Unity Inference Engine](docs/Unity-Inference-Engine.md) to
  provide native cross-platform support
- Unity environment [control from Python](docs/Python-API.md)
- Wrap Unity learning environments as a [gym](gym-unity/README.md)

See our [ML-Agents Overview](docs/ML-Agents-Overview.md) page for detailed
descriptions of all these features.

## Releases & Documentation

**Our latest, stable release is `Release 12`. Click
[here](https://github.com/Unity-Technologies/ml-agents/tree/release_12_docs/docs/Readme.md)
to get started with the latest release of ML-Agents.**

The table below lists all our releases, including our `master` branch which is
under active development and may be unstable. A few helpful guidelines:
- The [Versioning page](docs/Versioning.md) overviews how we manage our GitHub
  releases and the versioning process for each of the ML-Agents components.
- The [Releases page](https://github.com/Unity-Technologies/ml-agents/releases)
  contains details of the changes between releases.
- The [Migration page](docs/Migrating.md) contains details on how to upgrade
  from earlier releases of the ML-Agents Toolkit.
- The **Documentation** links in the table below include installation and usage
  instructions specific to each release. Remember to always use the
  documentation that corresponds to the release version you're using.

| **Version** | **Release Date** | **Source** | **Documentation** | **Download** |
|:-------:|:------:|:-------------:|:-------:|:------------:|
| **master (unstable)** | -- | [source](https://github.com/Unity-Technologies/ml-agents/tree/master) | [docs](https://github.com/Unity-Technologies/ml-agents/tree/master/docs/Readme.md) | [download](https://github.com/Unity-Technologies/ml-agents/archive/master.zip) |
| **Release 12** | **December 22, 2020** | **[source](https://github.com/Unity-Technologies/ml-agents/tree/release_12)** | **[docs](https://github.com/Unity-Technologies/ml-agents/tree/release_12_docs/docs/Readme.md)** | **[download](https://github.com/Unity-Technologies/ml-agents/archive/release_12.zip)** |
| **Release 11** | December 21, 2020 | [source](https://github.com/Unity-Technologies/ml-agents/tree/release_11) | [docs](https://github.com/Unity-Technologies/ml-agents/tree/release_11_docs/docs/Readme.md) | [download](https://github.com/Unity-Technologies/ml-agents/archive/release_11.zip) |
| **Release 10** | November 18, 2020 | [source](https://github.com/Unity-Technologies/ml-agents/tree/release_10) | [docs](https://github.com/Unity-Technologies/ml-agents/tree/release_10_docs/docs/Readme.md) | [download](https://github.com/Unity-Technologies/ml-agents/archive/release_10.zip) |
| **Release 9** | November 4, 2020 | [source](https://github.com/Unity-Technologies/ml-agents/tree/release_9) | [docs](https://github.com/Unity-Technologies/ml-agents/tree/release_9_docs/docs/Readme.md) | [download](https://github.com/Unity-Technologies/ml-agents/archive/release_9.zip) |
| **Release 8** | October 14, 2020 | [source](https://github.com/Unity-Technologies/ml-agents/tree/release_8) | [docs](https://github.com/Unity-Technologies/ml-agents/tree/release_8_docs/docs/Readme.md) | [download](https://github.com/Unity-Technologies/ml-agents/archive/release_8.zip) |
| **Release 7** | September 16, 2020 | [source](https://github.com/Unity-Technologies/ml-agents/tree/release_7) | [docs](https://github.com/Unity-Technologies/ml-agents/tree/release_7_docs/docs/Readme.md) | [download](https://github.com/Unity-Technologies/ml-agents/archive/release_7.zip) |
| **Release 6** | August 12, 2020 | [source](https://github.com/Unity-Technologies/ml-agents/tree/release_6) | [docs](https://github.com/Unity-Technologies/ml-agents/tree/release_6_docs/docs/Readme.md) | [download](https://github.com/Unity-Technologies/ml-agents/archive/release_6.zip) |
| **Release 5** | July 31, 2020 | [source](https://github.com/Unity-Technologies/ml-agents/tree/release_5) | [docs](https://github.com/Unity-Technologies/ml-agents/tree/release_5_docs/docs/Readme.md) | [download](https://github.com/Unity-Technologies/ml-agents/archive/release_5.zip) |

## Citation

If you are a researcher interested in a discussion of Unity as an AI platform,
see a pre-print of our
[reference paper on Unity and the ML-Agents Toolkit](https://arxiv.org/abs/1809.02627).

If you use Unity or the ML-Agents Toolkit to conduct research, we ask that you
cite the following paper as a reference:

Juliani, A., Berges, V., Teng, E., Cohen, A., Harper, J., Elion, C., Goy, C.,
Gao, Y., Henry, H., Mattar, M., Lange, D. (2020). Unity: A General Platform for
Intelligent Agents. _arXiv preprint
[arXiv:1809.02627](https://arxiv.org/abs/1809.02627)._
https://github.com/Unity-Technologies/ml-agents.

## Additional Resources

We have published a series of blog posts that are relevant for ML-Agents:

- (May 12, 2020)
  [Announcing ML-Agents Unity Package v1.0!](https://blogs.unity3d.com/2020/05/12/announcing-ml-agents-unity-package-v1-0/)
- (February 28, 2020)
  [Training intelligent adversaries using self-play with ML-Agents](https://blogs.unity3d.com/2020/02/28/training-intelligent-adversaries-using-self-play-with-ml-agents/)
- (November 11, 2019)
  [Training your agents 7 times faster with ML-Agents](https://blogs.unity3d.com/2019/11/11/training-your-agents-7-times-faster-with-ml-agents/)
- (October 21, 2019)
  [The AI@Unity interns help shape the world](https://blogs.unity3d.com/2019/10/21/the-aiunity-interns-help-shape-the-world/)
- (April 15, 2019)
  [Unity ML-Agents Toolkit v0.8: Faster training on real games](https://blogs.unity3d.com/2019/04/15/unity-ml-agents-toolkit-v0-8-faster-training-on-real-games/)
- (March 1, 2019)
  [Unity ML-Agents Toolkit v0.7: A leap towards cross-platform inference](https://blogs.unity3d.com/2019/03/01/unity-ml-agents-toolkit-v0-7-a-leap-towards-cross-platform-inference/)
- (December 17, 2018)
  [ML-Agents Toolkit v0.6: Improved usability of Brains and Imitation Learning](https://blogs.unity3d.com/2018/12/17/ml-agents-toolkit-v0-6-improved-usability-of-brains-and-imitation-learning/)
- (October 2, 2018)
  [Puppo, The Corgi: Cuteness Overload with the Unity ML-Agents Toolkit](https://blogs.unity3d.com/2018/10/02/puppo-the-corgi-cuteness-overload-with-the-unity-ml-agents-toolkit/)
- (September 11, 2018)
  [ML-Agents Toolkit v0.5, new resources for AI researchers available now](https://blogs.unity3d.com/2018/09/11/ml-agents-toolkit-v0-5-new-resources-for-ai-researchers-available-now/)
- (June 26, 2018)
  [Solving sparse-reward tasks with Curiosity](https://blogs.unity3d.com/2018/06/26/solving-sparse-reward-tasks-with-curiosity/)
- (June 19, 2018)
  [Unity ML-Agents Toolkit v0.4 and Udacity Deep Reinforcement Learning Nanodegree](https://blogs.unity3d.com/2018/06/19/unity-ml-agents-toolkit-v0-4-and-udacity-deep-reinforcement-learning-nanodegree/)
- (May 24, 2018)
  [Imitation Learning in Unity: The Workflow](https://blogs.unity3d.com/2018/05/24/imitation-learning-in-unity-the-workflow/)
- (March 15, 2018)
  [ML-Agents Toolkit v0.3 Beta released: Imitation Learning, feedback-driven features, and more](https://blogs.unity3d.com/2018/03/15/ml-agents-v0-3-beta-released-imitation-learning-feedback-driven-features-and-more/)
- (December 11, 2017)
  [Using Machine Learning Agents in a real game: a beginner’s guide](https://blogs.unity3d.com/2017/12/11/using-machine-learning-agents-in-a-real-game-a-beginners-guide/)
- (December 8, 2017)
  [Introducing ML-Agents Toolkit v0.2: Curriculum Learning, new environments, and more](https://blogs.unity3d.com/2017/12/08/introducing-ml-agents-v0-2-curriculum-learning-new-environments-and-more/)
- (September 19, 2017)
  [Introducing: Unity Machine Learning Agents Toolkit](https://blogs.unity3d.com/2017/09/19/introducing-unity-machine-learning-agents/)
- Overviewing reinforcement learning concepts
  ([multi-armed bandit](https://blogs.unity3d.com/2017/06/26/unity-ai-themed-blog-entries/)
  and
  [Q-learning](https://blogs.unity3d.com/2017/08/22/unity-ai-reinforcement-learning-with-q-learning/))

In addition to our own documentation, here are some additional, relevant
articles:

- [A Game Developer Learns Machine Learning](https://mikecann.co.uk/machine-learning/a-game-developer-learns-machine-learning-intent/)
- [Explore Unity Technologies ML-Agents Exclusively on Intel Architecture](https://software.intel.com/en-us/articles/explore-unity-technologies-ml-agents-exclusively-on-intel-architecture)
- [ML-Agents Penguins tutorial](https://learn.unity.com/project/ml-agents-penguins)

## Community and Feedback

The ML-Agents Toolkit is an open-source project and we encourage and welcome
contributions. If you wish to contribute, be sure to review our
[contribution guidelines](com.unity.ml-agents/CONTRIBUTING.md) and
[code of conduct](CODE_OF_CONDUCT.md).

For problems with the installation and setup of the ML-Agents Toolkit, or
discussions about how to best setup or train your agents, please create a new
thread on the
[Unity ML-Agents forum](https://forum.unity.com/forums/ml-agents.453/) and make
sure to include as much detail as possible. If you run into any other problems
using the ML-Agents Toolkit or have a specific feature request, please
[submit a GitHub issue](https://github.com/Unity-Technologies/ml-agents/issues).

Your opinion matters a great deal to us. Only by hearing your thoughts on the
Unity ML-Agents Toolkit can we continue to improve and grow. Please take a few
minutes to
[let us know about it](https://github.com/Unity-Technologies/ml-agents/issues/1454).

For any other questions or feedback, connect directly with the ML-Agents team at
ml-agents@unity3d.com.

## Privacy

In order to improve the developer experience for Unity ML-Agents Toolkit, we have added in-editor analytics.
Please refer to "Information that is passively collected by Unity" in the
[Unity Privacy Policy](https://unity3d.com/legal/privacy-policy).

## License

[Apache License 2.0](LICENSE)

<img src="docs/images/unity-wide.png" align="middle" width="3000"/>

<img src="docs/images/image-banner.png" align="middle" width="3000"/>

# Unity ML-Agents Toolkit (Beta)
[![docs badge](https://img.shields.io/badge/docs-reference-blue.svg)](docs/Readme.md)
[![license badge](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

**Unity Machine Learning Agents Toolkit** (ML-Agents) 은 지능형 에이전트를 학습시키기 위한
환경을 제공하여 게임 또는 시뮬레이션을 만들 수 있게 해주는 오픈소스 유니티 플러그인 입니다. 사용하기 쉬운 
파이썬 API를 통해 강화학습, 모방학습, 신경진화 또는 다른 기계학습 방법론을 사용하여 에이전트들을 학습시킬 수 있습니다.
우리는 또한 게임 개발자와 취미를 가지신 분들이 2D, 3D 그리고 VR/AR 게임들의 지능형 에이전트를
쉽게 훈련시킬수 있도록하는 최신 알고리즘 구현(TensorFlow에 기반하여)을 제공합니다. 학습된 에이전트들은
NPC의 행동 제어(다중-에이전트 또는 적대 관계와 같은 다양한 설정 속에서), 게임 빌드 테스트 자동화
그리고 출시 전 게임 설계 검증 등을 포함한 다양한 목적을 위해 사용될 수 있습니다. 
ML-Agents toolkit은 유니티의 풍부한 환경에서 AI의 발전을 위한 중심 플랫폼을 제공함으로써 
더욱 광범위한 연구와 게임 개발자 커뮤니티가 만들어지도록 하기 때문에 게임 개발자들과 AI 연구원들 모두에게 상호적으로 이익이 됩니다.

## Features

* Unity environment control from Python
* 10+ sample Unity environments
* Support for multiple environment configurations and training scenarios
* Train memory-enhanced agents using deep reinforcement learning
* Easily definable Curriculum Learning scenarios
* Broadcasting of agent behavior for supervised learning
* Built-in support for Imitation Learning
* Flexible agent control with On Demand Decision Making
* Visualizing network outputs within the environment
* Simplified set-up with Docker
* Wrap learning environments as a gym
* Utilizes the Unity Inference Engine
* Train using concurrent Unity environment instances

## Documentation

* For more information, in addition to installation and usage instructions, see
  our [documentation home](docs/Readme.md).
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

* [Unity AI - Unity 3D Artificial Intelligence](https://www.youtube.com/watch?v=bqsfkGbBU6k)
* [A Game Developer Learns Machine Learning](https://mikecann.co.uk/machine-learning/a-game-developer-learns-machine-learning-intent/)
* [Explore Unity Technologies ML-Agents Exclusively on Intel Architecture](https://software.intel.com/en-us/articles/explore-unity-technologies-ml-agents-exclusively-on-intel-architecture)

## Community and Feedback

The ML-Agents toolkit is an open-source project and we encourage and welcome
contributions. If you wish to contribute, be sure to review our
[contribution guidelines](CONTRIBUTING.md) and
[code of conduct](CODE_OF_CONDUCT.md).

If you run into any problems using the ML-Agents toolkit,
[submit an issue](https://github.com/Unity-Technologies/ml-agents/issues) and
make sure to include as much detail as possible.

Your opinion matters a great deal to us. Only by hearing your thoughts on the Unity ML-Agents Toolkit can we continue to improve and grow. Please take a few minutes to [let us know about it](https://github.com/Unity-Technologies/ml-agents/issues/1454). 


For any other questions or feedback, connect directly with the ML-Agents
team at ml-agents@unity3d.com. 

## Translations

To make the Unity ML-Agents toolkit accessible to the global research and
Unity developer communities, we're attempting to create and maintain
translations of our documentation. We've started with translating a subset
of the documentation to one language (Chinese), but we hope to continue
translating more pages and to other languages. Consequently,
we welcome any enhancements and improvements from the community.

* [Chinese](docs/localized/zh-CN/)

## License

[Apache License 2.0](LICENSE)

## Citation

If you use Unity or the ML-Agents Toolkit to conduct research, we ask that you cite the following paper as a reference:

Juliani, A., Berges, V., Vckay, E., Gao, Y., Henry, H., Mattar, M., Lange, D. (2018). Unity: A General Platform for Intelligent Agents. *arXiv preprint arXiv:1809.02627.* https://github.com/Unity-Technologies/ml-agents.

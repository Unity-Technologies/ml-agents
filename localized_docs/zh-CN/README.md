<img src="docs/images/unity-wide.png" align="middle" width="3000"/>

# Unity ML-Agents 工具包(Beta) v0.3.1

**注意:** 本文档为v0.3版本文档的部分翻译版，目前并不会随着英文版文档更新而更新。若要查看更新更全的英文版文档，请查看[这里](https://github.com/Unity-Technologies/ml-agents)。

**Unity Machine Learning Agents** (ML-Agents) 是一款开源的 Unity 插件，
使得我们得以在游戏环境和模拟环境中训练智能 agent。您可以使用 reinforcement learning（强化学习）、imitation learning（模仿学习）、neuroevolution（神经进化）或其他机器学习方法， 通过简单易用的 Python API进行控制，对 Agent 进行训练。我们还提供最先进算法的实现方式（基于
TensorFlow），让游戏开发者和业余爱好者能够轻松地
训练用于 2D、3D 和 VR/AR 游戏的智能 agent。
这些经过训练的 agent 可用于多种目的，
包括控制 NPC 行为（采用各种设置，
例如多个 agent 和对抗）、对游戏内部版本进行自动化测试、以及评估不同游戏设计决策的预发布版本。ML-Agents 对于游戏开发者和 AI 研究人员双方
都有利，因为它提供了一个集中的平台，
使得我们得以在 Unity 的丰富环境中测试 AI 的最新进展，
并使结果为更多的研究者和游戏开发者所用。

## 功能
* 用于控制 Unity 环境的 Python API
* 10 多个示例 Unity 环境
* 支持多种环境配置方案和训练方案
* 使用 deep reinforcement learning（深度强化学习）技术训练带记忆的Agent
* 可轻松定义的 Curriculum Learning（课程学习）方案
* 通过广播 Agent 行为实现监督学习
* 内置 Imitation Learning（模仿学习）支持
* 通过按需决策功能实现灵活的 Agent 控制
* 在环境中可查看神经网络的输出
* 通过 Docker 实现简化设置（测试功能）

## 文档和参考

**除了安装和使用说明外，如需更多信息，
请参阅我们的[文档主页](docs/Readme.md)。**如果您使用的
是 v0.3 之前的 ML-Agents 版本，强烈建议您参考
我们的[关于迁移到 v0.3 的指南](/docs/Migrating.md)。

我们还发布了一系列与 ML-Agents 相关的博客文章：
- reinforcement learning（强化学习）概念概述
([多臂强盗](https://blogs.unity3d.com/2017/06/26/unity-ai-themed-blog-entries/)
和 [Q-learning](https://blogs.unity3d.com/2017/08/22/unity-ai-reinforcement-learning-with-q-learning/))
- [在实际游戏中使用机器学习 Agent：初学者指南](https://blogs.unity3d.com/2017/12/11/using-machine-learning-agents-in-a-real-game-a-beginners-guide/)
- [文章](https://blogs.unity3d.com/2018/02/28/introducing-the-winners-of-the-first-ml-agents-challenge/)公布我们
[第一个 ML-Agents 挑战](https://connect.unity.com/challenges/ml-agents-1)的获胜者
- [文章](https://blogs.unity3d.com/2018/01/23/designing-safer-cities-through-simulations/)
概述如何利用 Unity 作为模拟器来设计更安全的城市。

除了我们自己的文档外，这里还有一些额外的相关文章：
- [Unity AI - Unity 3D 人工智能](https://www.youtube.com/watch?v=bqsfkGbBU6k)
- [游戏开发者学习机器学习](https://mikecann.co.uk/machine-learning/a-game-developer-learns-machine-learning-intent/)
- [在 Intel 体系结构上单独研究 Unity Technologies ML-Agents](https://software.intel.com/en-us/articles/explore-unity-technologies-ml-agents-exclusively-on-intel-architecture)

## 社区和反馈

ML-Agents 是一个开源项目，我们鼓励并欢迎大家贡献自己的力量。
如果您想做出贡献，请务必查看我们的
[贡献准则](/com.unity.ml-agents/CONTRIBUTING.md)和
[行为准则](/CODE_OF_CONDUCT.md)。

您可以通过 Unity Connect 和 GitHub 与我们以及更广泛的社区进行交流：
* 加入我们的
[Unity 机器学习频道](https://connect.unity.com/messages/c/035fba4f88400000)
与使用 ML-Agents 的其他人以及对机器学习充满热情的 Unity 开发者
交流。我们使用该频道来展示关于 ML-Agents
（在更广泛的范围内，还包括游戏中的机器学习）的最新动态。
* 如果您在使用 ML-Agents 时遇到任何问题，请
[提交问题](https://github.com/Unity-Technologies/ml-agents/issues)并
确保提供尽可能多的详细信息。

对于任何其他问题或反馈，请直接与 ML-Agents 团队联系，
电子邮件地址为 ml-agents@unity3d.com。

## 许可证

[Apache 许可证 2.0](LICENSE)

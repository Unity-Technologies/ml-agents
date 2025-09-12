# Unity ML-Agents Toolkit

[![docs badge](https://img.shields.io/badge/docs-reference-blue.svg)](https://docs.unity3d.com/Packages/com.unity.ml-agents@latest)

[![license badge](https://img.shields.io/badge/license-Apache--2.0-green.svg)](https://github.com/Unity-Technologies/ml-agents/blob/release/4.0.0/LICENSE.md)

([latest release](https://github.com/Unity-Technologies/ml-agents/releases/tag/latest_release)) ([all releases](https://github.com/Unity-Technologies/ml-agents/releases))

**The Unity Machine Learning Agents Toolkit** (ML-Agents) is an open-source project that enables games and simulations to serve as environments for training intelligent agents. We provide implementations (based on PyTorch) of state-of-the-art algorithms to enable game developers and hobbyists to easily train intelligent agents for 2D, 3D and VR/AR games. Researchers can also use the provided simple-to-use Python API to train Agents using reinforcement learning, imitation learning, neuroevolution, or any other methods. These trained agents can be used for multiple purposes, including controlling NPC behavior (in a variety of settings such as multi-agent and adversarial), automated testing of game builds and evaluating different game design decisions pre-release. The ML-Agents Toolkit is mutually beneficial for both game developers and AI researchers as it provides a central platform where advances in AI can be evaluated on Unity’s rich environments and then made accessible to the wider research and game developer communities.

## Features
- 17+ [example Unity environments](https://docs.unity3d.com/Packages/com.unity.ml-agents@latest/index.html?subfolder=/manual/Learning-Environment-Examples.html)
- Support for multiple environment configurations and training scenarios
- Flexible Unity SDK that can be integrated into your game or custom Unity scene
- Support for training single-agent, multi-agent cooperative, and multi-agent competitive scenarios via several Deep Reinforcement Learning algorithms (PPO, SAC, MA-POCA, self-play).
- Support for learning from demonstrations through two Imitation Learning algorithms (BC and GAIL).
- Quickly and easily add your own [custom training algorithm](https://docs.unity3d.com/Packages/com.unity.ml-agents@latest/index.html?subfolder=/manual/Python-Custom-Trainer-Plugin.html) and/or components.
- Easily definable Curriculum Learning scenarios for complex tasks
- Train robust agents using environment randomization
- Flexible agent control with On Demand Decision Making
- Train using multiple concurrent Unity environment instances
- Utilizes the [Inference Engine](https://docs.unity3d.com/Packages/com.unity.ml-agents@latest/index.html?subfolder=/manual/Inference-Engine.html) to provide native cross-platform support
- Unity environment [control from Python](https://docs.unity3d.com/Packages/com.unity.ml-agents@latest/index.html?subfolder=/manual/Python-LLAPI.html)
- Wrap Unity learning environments as a [gym](https://docs.unity3d.com/Packages/com.unity.ml-agents@latest/index.html?subfolder=/manual/Python-Gym-API.html) environment
- Wrap Unity learning environments as a [PettingZoo](https://docs.unity3d.com/Packages/com.unity.ml-agents@latest/index.html?subfolder=/manual/Python-PettingZoo-API.html) environment

## Releases & Documentation

> **⚠️ Documentation Migration Notice**
> We have moved to [Unity Package documentation](https://docs.unity3d.com/Packages/com.unity.ml-agents@latest) as the **primary developer documentation** and have **deprecated** the maintenance of [web docs](https://unity-technologies.github.io/ml-agents/). Please use the Unity Package documentation for the most up-to-date information.

The table below shows our latest release, including our `develop` branch which is under active development and may be unstable. A few helpful guidelines:

- The [Versioning page](https://docs.unity3d.com/Packages/com.unity.ml-agents@latest/index.html?subfolder=/manual/Versioning.html) overviews how we manage our GitHub releases and the versioning process for each of the ML-Agents components.
- The [Releases page](https://github.com/Unity-Technologies/ml-agents/releases) contains details of the changes between releases.
- The [Migration page](https://docs.unity3d.com/Packages/com.unity.ml-agents@latest/index.html?subfolder=/manual/Migrating.html) contains details on how to upgrade from earlier releases of the ML-Agents Toolkit.
- The `com.unity.ml-agents` package is [verified](https://docs.unity3d.com/2020.1/Documentation/Manual/pack-safe.html) for Unity 2020.1 and later. Verified packages releases are numbered 1.0.x.

|      **Version**       |  **Release Date**   |                                  **Source**                                   |                                                 **Documentation**                                                  |                                      **Download**                                      |                  **Python Package**                   |                                   **Unity Package**                                   |
|:----------------------:|:-------------------:|:-----------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------:|:-----------------------------------------------------:|:-------------------------------------------------------------------------------------:|
|     **Release 23**     | **August 15, 2025** | **[source](https://github.com/Unity-Technologies/ml-agents/tree/release_23)** |              **[docs](https://docs.unity3d.com/Packages/com.unity.ml-agents@4.0/manual/index.html)**               | **[download](https://github.com/Unity-Technologies/ml-agents/archive/release_23.zip)** | **[1.1.0](https://pypi.org/project/mlagents/1.1.0/)** |                                       **4.0.0**                                       |
| **develop (unstable)** |         --          |    [source](https://github.com/Unity-Technologies/ml-agents/tree/develop)     | [docs](https://github.com/Unity-Technologies/ml-agents/tree/develop/com.unity.ml-agents/Documentation~/index.md)   |    [download](https://github.com/Unity-Technologies/ml-agents/archive/develop.zip)     |                         --                            |                                          --                                           |



If you are a researcher interested in a discussion of Unity as an AI platform, see a pre-print of our [reference paper on Unity and the ML-Agents Toolkit](https://arxiv.org/abs/1809.02627).

If you use Unity or the ML-Agents Toolkit to conduct research, we ask that you cite the following paper as a reference:

```
@article{juliani2020,
  title={Unity: A general platform for intelligent agents},
  author={Juliani, Arthur and Berges, Vincent-Pierre and Teng, Ervin and Cohen, Andrew and Harper, Jonathan and Elion, Chris and Goy, Chris and Gao, Yuan and Henry, Hunter and Mattar, Marwan and Lange, Danny},
  journal={arXiv preprint arXiv:1809.02627},
  url={https://arxiv.org/pdf/1809.02627.pdf},
  year={2020}
}
```

Additionally, if you use the MA-POCA trainer in your research, we ask that you cite the following paper as a reference:

```
@article{cohen2022,
  title={On the Use and Misuse of Absorbing States in Multi-agent Reinforcement Learning},
  author={Cohen, Andrew and Teng, Ervin and Berges, Vincent-Pierre and Dong, Ruo-Ping and Henry, Hunter and Mattar, Marwan and Zook, Alexander and Ganguly, Sujoy},
  journal={RL in Games Workshop AAAI 2022},
  url={http://aaai-rlg.mlanctot.info/papers/AAAI22-RLG_paper_32.pdf},
  year={2022}
}
```


## Additional Resources

* [Unity Discussions](https://discussions.unity.com/tag/ml-agents)
* [ML-Agents tutorials by CodeMonkeyUnity](https://www.youtube.com/playlist?list=PLzDRvYVwl53vehwiN_odYJkPBzcqFw110)
* [Introduction to ML-Agents by Huggingface](https://huggingface.co/learn/deep-rl-course/en/unit5/introduction)
* [Community created ML-Agents projects](https://discussions.unity.com/t/post-your-ml-agents-project/816756)
* [ML-Agents models on Huggingface](https://huggingface.co/models?library=ml-agents)
* [Blog posts](https://docs.unity3d.com/Packages/com.unity.ml-agents@latest/index.html?subfolder=/manual/Blog-posts.html)
* [Discord](https://discord.com/channels/489222168727519232/1202574086115557446)

## Community and Feedback

The ML-Agents Toolkit is an open-source project and we encourage and welcome contributions. If you wish to contribute, be sure to review our [contribution guidelines](https://docs.unity3d.com/Packages/com.unity.ml-agents@latest/index.html?subfolder=/manual/CONTRIBUTING.html) and [code of conduct](https://github.com/Unity-Technologies/ml-agents/blob/release/4.0.0/CODE_OF_CONDUCT.md).

For problems with the installation and setup of the ML-Agents Toolkit, or discussions about how to best setup or train your agents, please create a new thread on the [Unity ML-Agents discussion forum](https://discussions.unity.com/tag/ml-agents). Be sure to include as many details as possible to help others assist you effectively. If you run into any other problems using the ML-Agents Toolkit or have a specific feature request, please [submit a GitHub issue](https://github.com/Unity-Technologies/ml-agents/issues).

Please tell us which samples you would like to see shipped with the ML-Agents Unity package by replying to [this discussion thread](https://discussions.unity.com/t/help-shape-the-future-of-ml-agents/1661019).

## Privacy

In order to improve the developer experience for Unity ML-Agents Toolkit, we have added in-editor analytics. Please refer to "Information that is passively collected by Unity" in the [Unity Privacy Policy](https://unity3d.com/legal/privacy-policy).

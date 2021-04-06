<img src="docs/images/image-banner.png" align="middle" width="3000"/>

# 유니티 ML-Agents 툴킷

[![docs badge](https://img.shields.io/badge/docs-reference-blue.svg)](https://github.com/Unity-Technologies/ml-agents/tree/release_12_docs/docs/)

[![license badge](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

([latest release](https://github.com/Unity-Technologies/ml-agents/releases/tag/latest_release))
([all releases](https://github.com/Unity-Technologies/ml-agents/releases))

**유니티 기계학습 에이전트 툴킷** (ML-Agents) 은 게임 컨텐츠 및 게임을 포함한 다양한 시뮬레이션에서 활용하기 위한 지능형 에이전트를 훈련시키는 환경을 제공하는 오픈 소스 프로젝트입니다.
ML-Agents는 게임 개발자 들이 2D, 3D 및 가상현실/증강현실 게임에서 지능형 에이전트를 쉽게 교육할 수 있도록 최신 알고리즘의 구현(PyTorch 기반)을 제공합니다.
간단한 파이썬 API를 사용하여 강화 학습, 모방 학습, 신경 진화 등 다른 방법을 활용하여 에이전트를 교육할 수 있습니다.
학습된 에이전트는 NPC 행동 제어(다중 에이전트 및 적대적 에이전트와 같은 다양한 설정), 게임 빌드 테스트 자동화, 그리고 출시 전 게임 설계(밸런스) 검증 등을 포함한 다양한 용도로 활용할 수 있습니다.
ML-Agents 툴킷은 유니티의 자유로운 환경에서 인공지능 에이전트를 개발하기 위한 기반을 제공하며, 이틀 통해 연구자 및 게임 개발자 등 광범위한 커뮤니티에 접근할 수 있기 때문에 게임 개발자와 인공지능 연구원 모두에게 상호 이익이 됩니다.

## 특징

- 15+ [유니티 환경 예제](docs/Learning-Environment-Examples.md)
- 다양한 환경 구성 및 교육 시나리오 지원
- 게임이나 커스텀 유니티 씬에 통합될 수 있는 유연한 유니티 SDK
- Proximal Policy Optimization (PPO) 와 Soft Actor-Critic (SAC) 의 두 가지 심층 강화 학습 알고리즘을 이용한 훈련
- Behavioral Cloning 이나 Generative Adversarial Imitation Learning 을 통한 모방 학습에 대한 내장 지원
- 적대적(Adversarial) 시나리오에서 에이전트를 교육하기 위한 Self-play 메커니즘
- 복잡한 작업에 대해 쉽게 정의할 수 있는 커리큘럼 학습 시나리오
- 환경 랜덤화를 사용하여 강력한 에이전트 학습
- 온 디맨드 의사 결정을 통한 유연한 에이전트 제어
- 여러 개의 유니티 환경 인스턴스를 동시에 사용하는 학습
- 네이티브 크로스 플랫폼을 지원하기 위해 [유니티 추론(Inference) 엔진](docs/Unity-Inference-Engine.md) 이용
- 유니티 환경 [파이썬에서 제어](docs/Python-API.md)
- [gym](gym-unity/README.md) 과 같은 유니티 학습 환경 제공

이 모든 기능에 대한 자세한 설명은 [ML-Agents 개요](docs/ML-Agents-Overview.md) 페이지를 참조하십시오.

## 릴리즈 & 설명서

**최신의 안정적 릴리즈는 `Release 12` 입니다. 클릭해서 ML-Agents의 최신 릴리스를 시작하세요.** [여기](https://github.com/Unity-Technologies/ml-agents/tree/release_12_docs/docs/Readme.md)

아래 표에는 현재 개발이 진행 중이며 불안정할 수 있는 `master` 브랜치를 포함한 모든 릴리스가 나와 있습니다. 몇 가지 유용한 지침:
- [버전 관리 페이지](docs/Versioning.md) 는 GitHub 릴리즈를 관리하는 방법과 각 ML-Agents 구성 요소에 대한 버전 관리 프로세스를 간략히 설명합니다.
- [릴리즈 페이지](https://github.com/Unity-Technologies/ml-agents/releases) 는 릴리스 간의 변경 사항에 대한 세부 정보가 포함되어 있습니다.
- [마이그레이션(Migration) 페이지](docs/Migrating.md) 는 이전 릴리스의 ML-Agents 툴킷에서 업그레이드하는 방법에 대한 세부 정보가 포함되어 있습니다.
- 아래 표의 **설명서** 링크에는 각 릴리스에 대한 설치 및 사용 지침이 포함되어 있습니다. 사용 중인 릴리스 버전에 해당하는 설명서를 항상 사용해야 합니다.

| **버전** | **릴리즈 날짜** | **소스** | **설명서** | **다운로드** |
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

## 인용

인공지능 플랫폼으로서의 유니티에 대한 논의에 관심이 있는 연구자라면, 프리프린트를 참조하시오.
[ 및 ML-Agents 툴킷에 대한 참조 문서](https://arxiv.org/abs/1809.02627).

유니티 또는 ML-Agents 툴킷을 사용하여 연구를 수행하는 경우, 다음 논문을 참조 자료로 인용할 것을 요청합니다.
Juliani, A., Berges, V., Teng, E., Cohen, A., Harper, J., Elion, C., Goy, C., Gao, Y., Henry, H., Mattar, M., Lange, D. (2020). Unity: A General Platform for Intelligent Agents. _arXiv preprint [arXiv:1809.02627](https://arxiv.org/abs/1809.02627)._ https://github.com/Unity-Technologies/ml-agents.

## 추가 리소스

유니티 및 ML-Agents 툴킷에 대해 자세히 소개하는 유니티 학습 과정이 있습니다. [ML-Agents: 벌새](https://learn.unity.com/course/ml-agents-hummingbirds)

또한 [CodeMonkeyUnity](https://www.youtube.com/c/CodeMonkeyUnity)와 제휴하여 ML-Agents 툴킷의 구현 및 사용 방법에 대한 [튜토리얼 비디오](https://www.youtube.com/playlist?list=PLzDRvYVwl53vehwiN_odYJkPBzcqFw110)도 제작했습니다.

또한 ML-Agents 관련 블로그 게시물도 게시했습니다.

- (December 28, 2020)
  [Happy holidays from the Unity ML-Agents team!](https://blogs.unity3d.com/2020/12/28/happy-holidays-from-the-unity-ml-agents-team/)
- (November 20, 2020)
  [How Eidos-Montréal created Grid Sensors to improve observations for training agents](https://blogs.unity3d.com/2020/11/20/how-eidos-montreal-created-grid-sensors-to-improve-observations-for-training-agents/)
- (November 11, 2020)
  [2020 AI@Unity interns shoutout](https://blogs.unity3d.com/2020/11/11/2020-aiunity-interns-shoutout/)
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


## 커뮤니티 그리고 피드백

ML-Agents 툴킷은 오픈소스 프로젝트이며 컨트리뷰션을 환영합니다. 만약 컨트리뷰션을 원하시는 경우
[컨트리뷰션 가이드라인](com.unity.ml-agents/CONTRIBUTING.md) 과 [행동 규칙](CODE_OF_CONDUCT.md) 을 검토해주십시오.

ML-Agents 툴킷 설치 및 설정과 관련된 문제 또는 에이전트를 가장 잘 설정하거나 교육하는 방법에 대한 논의는 [유니티 ML-Agents 포럼](https://forum.unity.com/forums/ml-agents.453/) 에 새 스레드를 작성하십시오. 가능한 많은 세부 정보를 포함해야 합니다.
ML-Agents 툴킷을 사용하여 다른 문제가 발생하거나 특정 기능 요청이 있는 경우 [이슈 제출](https://github.com/Unity-Technologies/ml-agents/issues) 부탁합니다.

여러분의 의견은 저희에게 매우 중요합니다. 유니티 ML-Agents 툴킷에 관련된 여러분의 의견을 통해서 저희는 계속해서 발전하고 성장할 수 있습니다. 단 몇 분만 사용하여 [저희에게 알려주세요](https://github.com/Unity-Technologies/ml-agents/issues/1454).

다른 의견과 피드백은 ML-Agents 팀과 직접 연락부탁드립니다. (ml-agents@unity3d.com)


## 개인정보

Unity ML-Agents 툴킷에 대한 개발자 경험을 개선하기 위해, 우리는 에디터 내부 분석을 추가했습니다.
[유니티 개인 정보 보호 정책](https://unity3d.com/legal/privacy-policy) 의 "Unity가 기본적으로 수집하는 정보"를 참조하시기 바랍니다.


## 라이센스

[Apache License 2.0](LICENSE)


## 한글 번역

유니티 ML-Agents 관련 문서의 한글 번역은 [장현준(Hyeonjun Jang)][https://github.com/JangHyeonJun],  [민규식 (Kyushik Min)]([https://github.com/Kyushik](https://github.com/Kyushik))에 의해 진행되었습니다. 내용상 오류나 오탈자가 있는 경우 각 문서의 번역을 진행한 사람의 이메일을 통해 연락주시면 감사드리겠습니다.

장현준: totok682@naver.com

민규식: kyushikmin@gmail.com

최태혁: chlxogur_@naver.com

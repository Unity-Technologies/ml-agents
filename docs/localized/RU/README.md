<img src="https://github.com/Unity-Technologies/ml-agents/blob/master/docs/images/image-banner.png" align="middle" width="3000"/>

# Unity ML-Agents Toolkit Version Release 7

[![docs badge](https://img.shields.io/badge/docs-reference-blue.svg)](https://github.com/Unity-Technologies/ml-agents/tree/release_7_docs/docs/)
[![license badge](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

([latest release](https://github.com/Unity-Technologies/ml-agents/releases/tag/latest_release))
([all releases](https://github.com/Unity-Technologies/ml-agents/releases))

**The Unity Machine Learning Agents Toolkit (ML-Agents)** - open-source проект,
предназначенный для обучения искусственного интеллекта (агента) через взаимодействие со средой, -
игрой или симуляцией, - используя различные методы машинного обучения:
обучение с подкреплением (reinforcement learning), имитационное обучение (imitation learning),
нейроэволюция (neuroevolution) и др. средствами Python API. В проекте реализованы также и современные
алгоритмы (на основе TensorFlow), чтобы дать возможность как разработчикам игр так и любым другим,
кто увлечен темой AI, обучать искусственный интеллект для 2D, 3D и VR/AR игр. Применение таких агентов
бесчисленно: например, вы можете использовать их для управления NPC (опций также много - будь то
обучение действиям в кооперативе или друг против друга), для тестирования различных версий сборок
игры, а также для оценки гейм дизайнерских решений. ML-Agents объединяет разработчиков игр и
исследователей AI, так как предоставляет единую платформу, в рамках которой новые разработки
в сфере искусственного интеллекта могут быть протестированы через движок Unity и, как следствие,
стать доступнее большему количеству и тех, и других.

## Особенности:

- Более [15 примеров на Unity](docs/Learning-Environment-Examples.md).
- Большие возможности по конфигурации среды и тренировочных сценариев.
- Unity SDK, который легко встроить в код вашей игры или в кастомную сцену в Unity
- Два алгоритма глубинного обучения с подкреплением (deep reinforcement learning):
Proximal Policy Optimization (PPO) и Soft Actor-Critic (SAC). Первый алгоритм старается узнать,
какой будет наилучший шаг в конкретной ситуации, тогда как второй - узнать в целом правила
игры/системы/симуляции, их закон и действовать согласно этому усвоенному закону изменения среды.
- Встроенная поддержка для имитационного обучения (Imitation Learning), которое можно сделать
либо через клонирование поведения (Behavioral Cloning), либо через генеративно-состязательное
имитационное обучение (Generative Adversarial Imitation Learning - GAIL), когда одна часть алгоритма
генерирует поведение, а другая определяет, похоже данное поведение на то, которое было дано как исходное,
например, самим пользователем в виде записи его действий. Генерация происходит до тех пор, пока
сгенерированное поведение не будет определено как неотличимое или очень близкое к исходному.
- Возможность для агента игры с самим собой, если агент обучается в контексте сценария “состязание”:
например, игра в футбол, где есть две команды.
- ML-Agents позволяет настроить череду сцен, где каждая новая сцена - это усложнение сцены предыдущей,
например, добавление новой преграды. Не всегда поставленную задачу агент сможет научиться
выполнять, если среда слишком сложная изначально. Дайте ему сначала сценку попроще, когда
он научиться ее проходить, его перенесет на уровень посложнее.
- Обучение агента, устойчивого к изменениям, с помощью возможности случайного генерации элементов сцены
- Гибкий контроль агента: демонстрация выученного поведения только при определенных условиях.
Например, NPC входит в контекст “атака” - атакует так, как научился ранее в рамках обучающего сценария.
- Обучение агента сразу на множестве сцен одновременно. Представьте, как он играет в футбол сразу
на десяти стадионах, набираясь опыта одновременно на них всех. Выглядит это в Unity также,
как и представляется.
- Использование [Unity Inference Engine](docs/Unity-Inference-Engine.md) для поддержки кроссплатформенности.
- Контроль через [Python API](docs/Python-API.md) сцен.
- Возможность обернуть Unity среду для обучения как [gym](gym-unity/README.md).

Для более детального ознакомления с данными особенностями см. [Обзор ML-Agents] (docs/ML-Agents-Overview.md).

## Релизы и Документация

**Наш последний стабильный релиз - это `7-ой Релиз` (Release 7).
См. [здесь](https://github.com/Unity-Technologies/ml-agents/tree/release_7_docs/docs/Readme.md),
чтобы начать работать с самой последней версий ML-Agents.**

Таблица внизу - список всех наших релизов, включая master ветку, над которой мы ведем активную работу
и которая может быть нестабильной. Полезная информация:

[Управление версиями](docs/Versioning.md) - описание того, как мы работам с GitHub.
[Релизы](https://github.com/Unity-Technologies/ml-agents/releases) - об изменениях между версиями
[Миграция](docs/Migrating.md) - как перейти с более ранней версии ML-Agents на новую.
Ссылки на **документацию** - как установить и начать пользоваться ML-Agents в зависимости от версии.
Всегда используйте только ту документацию, которая относится к той версии, которую вы установили:

| **Version** | **Дата релиза** | **Source** | **Документация** | **Загрузка** |
|:-------:|:------:|:-------------:|:-------:|:------------:|
| **master (unstable)** | -- | [source](https://github.com/Unity-Technologies/ml-agents/tree/master) | [docs](https://github.com/Unity-Technologies/ml-agents/tree/master/docs/Readme.md) | [download](https://github.com/Unity-Technologies/ml-agents/archive/master.zip) |
| **Release 7** | **16 Сентября, 2020** | **[source](https://github.com/Unity-Technologies/ml-agents/tree/release_7)** | **[docs](https://github.com/Unity-Technologies/ml-agents/tree/release_7_docs/docs/Readme.md)** | **[download](https://github.com/Unity-Technologies/ml-agents/archive/release_7.zip)** |
| **Release 6** | 12 Августа, 2020 | [source](https://github.com/Unity-Technologies/ml-agents/tree/release_6) | [docs](https://github.com/Unity-Technologies/ml-agents/tree/release_6_docs/docs/Readme.md) | [download](https://github.com/Unity-Technologies/ml-agents/archive/release_6.zip) |
| **Release 5** | 31 Июля, 2020 | [source](https://github.com/Unity-Technologies/ml-agents/tree/release_5) | [docs](https://github.com/Unity-Technologies/ml-agents/tree/release_5_docs/docs/Readme.md) | [download](https://github.com/Unity-Technologies/ml-agents/archive/release_5.zip) |
| **Release 4** | 15 Июля, 2020 | [source](https://github.com/Unity-Technologies/ml-agents/tree/release_4) | [docs](https://github.com/Unity-Technologies/ml-agents/tree/release_4_docs/docs/Readme.md) | [download](https://github.com/Unity-Technologies/ml-agents/archive/release_4.zip) |
| **Release 3** | 10 Июня, 2020 | [source](https://github.com/Unity-Technologies/ml-agents/tree/release_3) | [docs](https://github.com/Unity-Technologies/ml-agents/tree/release_3_docs/docs/Readme.md) | [download](https://github.com/Unity-Technologies/ml-agents/archive/release_3.zip) |
| **Release 2** | 20 Мая, 2020 | [source](https://github.com/Unity-Technologies/ml-agents/tree/release_2) | [docs](https://github.com/Unity-Technologies/ml-agents/tree/release_2_docs/docs/Readme.md) | [download](https://github.com/Unity-Technologies/ml-agents/archive/release_2.zip) |
| **Release 1** | 30 Апреля, 2020 | [source](https://github.com/Unity-Technologies/ml-agents/tree/release_1) | [docs](https://github.com/Unity-Technologies/ml-agents/tree/release_1_docs/docs/Readme.md) | [download](https://github.com/Unity-Technologies/ml-agents/archive/release_1.zip) |

## Цитирование

Если вас интересует Unity как платформа для изучения AI, см. [нашу работу Unity и ML-Agents](https://arxiv.org/abs/1809.02627).
Если вы используете Unity или ML-Agents для исследовательской работы, пожалуйста, указывайте
в списке используемой литературы следующую работу:
Juliani, A., Berges, V., Teng, E., Cohen, A., Harper, J., Elion, C., Goy,
C., Gao, Y., Henry, H., Mattar, M., Lange, D. (2020). Unity: A General Platform for
Intelligent Agents. _arXiv preprint
[arXiv:1809.02627].(https://arxiv.org/abs/1809.02627)._
https://github.com/Unity-Technologies/ml-agents.

## Дополнительные источники:

Мы опубликовали серию статей на нашем блоге про ML-Agents (**пока без перевода на русский**):

- (12 Мая, 2020)
[Announcing ML-Agents Unity Package v1.0!](https://blogs.unity3d.com/2020/05/12/announcing-ml-agents-unity-package-v1-0/)
- (28 Февраля, 2020)
[Training intelligent adversaries using self-play with ML-Agents](https://blogs.unity3d.com/2020/02/28/training-intelligent-adversaries-using-self-play-with-ml-agents/)
- (11 Ноября, 2019)
[Training your agents 7 times faster with ML-Agents](https://blogs.unity3d.com/2019/11/11/training-your-agents-7-times-faster-with-ml-agents/)
- (21 Октября, 2019)
[The AI@Unity interns help shape the world](https://blogs.unity3d.com/2019/10/21/the-aiunity-interns-help-shape-the-world/)
- (15 Апреля, 2019)
[Unity ML-Agents Toolkit v0.8: Faster training on real games](https://blogs.unity3d.com/2019/04/15/unity-ml-agents-toolkit-v0-8-faster-training-on-real-games/)
- (1 Марта, 2019)
[Unity ML-Agents Toolkit v0.7: A leap towards cross-platform inference](https://blogs.unity3d.com/2019/03/01/unity-ml-agents-toolkit-v0-7-a-leap-towards-cross-platform-inference/)
- (17 Декабря, 2018)
[ML-Agents Toolkit v0.6: Improved usability of Brains and Imitation Learning](https://blogs.unity3d.com/2018/12/17/ml-agents-toolkit-v0-6-improved-usability-of-brains-and-imitation-learning/)
- (2 Октября, 2018)
[Puppo, The Corgi: Cuteness Overload with the Unity ML-Agents Toolkit](https://blogs.unity3d.com/2018/10/02/puppo-the-corgi-cuteness-overload-with-the-unity-ml-agents-toolkit/)
- (11 Сентября, 2018)
[ML-Agents Toolkit v0.5, new resources for AI researchers available now](https://blogs.unity3d.com/2018/09/11/ml-agents-toolkit-v0-5-new-resources-for-ai-researchers-available-now/)
- (26 Июня, 2018)
[Solving sparse-reward tasks with Curiosity](https://blogs.unity3d.com/2018/06/26/solving-sparse-reward-tasks-with-curiosity/)
- (19 Июня, 2018)
[Unity ML-Agents Toolkit v0.4 and Udacity Deep Reinforcement Learning Nanodegree](https://blogs.unity3d.com/2018/06/19/unity-ml-agents-toolkit-v0-4-and-udacity-deep-reinforcement-learning-nanodegree/)
- (24 Мая, 2018)
[Imitation Learning in Unity: The Workflow](https://blogs.unity3d.com/2018/05/24/imitation-learning-in-unity-the-workflow/)
- (15 Марта, 2018)
[ML-Agents Toolkit v0.3 Beta released: Imitation Learning, feedback-driven features, and more](https://blogs.unity3d.com/2018/03/15/ml-agents-v0-3-beta-released-imitation-learning-feedback-driven-features-and-more/)
- (11 Декабря, 2017)
[Using Machine Learning Agents in a real game: a beginner’s guide](https://blogs.unity3d.com/2017/12/11/using-machine-learning-agents-in-a-real-game-a-beginners-guide/)
- (8 Декабря, 2017)
[Introducing ML-Agents Toolkit v0.2: Curriculum Learning, new environments, and more](https://blogs.unity3d.com/2017/12/08/introducing-ml-agents-v0-2-curriculum-learning-new-environments-and-more/)
- (19 Сентября, 2017)
[Introducing: Unity Machine Learning Agents Toolkit](https://blogs.unity3d.com/2017/09/19/introducing-unity-machine-learning-agents/)
- Обзор обучения с подкреплением (
[multi-armed bandit](https://blogs.unity3d.com/2017/06/26/unity-ai-themed-blog-entries/)
и
[Q-learning](https://blogs.unity3d.com/2017/08/22/unity-ai-reinforcement-learning-with-q-learning/))

Дополнительные материалы от других авторов:
- [A Game Developer Learns Machine Learning] (https://mikecann.co.uk/machine-learning/a-game-developer-learns-machine-learning-intent/)
- [Explore Unity Technologies ML-Agents Exclusively on Intel Architecture](https://software.intel.com/en-us/articles/explore-unity-technologies-ml-agents-exclusively-on-intel-architecture)
- [ML-Agents Penguins tutorial](https://learn.unity.com/project/ml-agents-penguins)

## Community and Feedback

ML-Agents Toolkit - open-source проект, поэтому мы рады любой помощи. Если вы хотите нам помочь,
ознакомьтесь, для начала, пожалуйста, для с [гайдом, как сделать это правильно](com.unity.ml-agents/CONTRIBUTING.md),
и [кодексом поведения](CODE_OF_CONDUCT.md).

Если возникли проблемы с установкой и настройкой ML-Agents, если вы хотите обсудить как лучше всего
обучать агентов и пр., пожалуйста, посмотрите возможные решения на [форуме Unity ML-Agents](https://forum.unity.com/forums/ml-agents.453/).
Если вы не найдете нужной вам информации, начните новую тему, дав подробное описания вашей проблемы. Если вы обнаружили
какие-то баги или ошибки во время работы с ML-Agents, пожалуйста, сообщите об этом [здесь](https://github.com/Unity-Technologies/ml-agents/issues).

Нам важно знать ваше мнение. Только на его основе проект Unity ML-Agents и продолжает развиваться.
Пожалуйста, уделите несколько минут и [поделитесь](https://github.com/Unity-Technologies/ml-agents/issues/1454)
с нами тем, что могло бы улучшить наш проект.

По всем остальным вопросам или отзыву, пишите сразу на адрес команды разработчиков ML-Agents - ml-agents@unity3d.com.

## Лицензия

Apache License 2.0

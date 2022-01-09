# Unity 中的 Reinforcement Learning（强化学习）

Reinforcement learning（强化学习）是一种人工智能技术，通过奖励期望的行为来训练 _agent_ 执行任务。在 reinforcement learning（强化学习）过程中，agent 会探索自己所处的环境，观测事物的状态，并根据这些观测结果采取相应动作。如果该动作带来了更好的状态，agent 会得到正奖励。如果该动作带来的状态不太理想，则 agent 不会得到奖励或会得到负奖励（惩罚）。随着 agent 在训练期间不断学习，它会优化自己的决策能力，以便随着时间的推移获得最高奖励。

ML-Agents 使用一种称为 [Proximal Policy Optimization (PPO)](https://blog.openai.com/openai-baselines-ppo/) 的 reinforcement learning（强化学习）技术。PPO 使用神经网络来逼近理想函数；这种理想函数将 agent 的观测结果映射为 agent 在给定状态下可以采取的最佳动作。ML-Agents PPO 算法在 TensorFlow 中实现，并在单独的 Python 过程中运行（通过一个socket与正在运行的 Unity 应用程序进行通信）。

**注意：**如果您并非要专门研究机器学习和 reinforcement learning（强化学习）主题，只想训练 agent 完成任务，则可以将 PPO 训练视为一个_黑盒_。在 Unity 内部以及在 Python 训练方面有一些与训练相关的参数可进行调整，但您不需要深入了解算法本身就可以成功创建和训练 agent。[训练 ML-Agents](/docs/Training-ML-Agents.md)提供了执行训练过程的逐步操作程序。

## 模拟和训练过程

训练和模拟过程以 ML-Agents Academy 类编排的步骤进行。Academy 与场景中的 Agent 和 Brain 对象一起协作逐步完成模拟。当 Academy 已达到其最大步数或场景中的所有 agent 均_完成_时，一个训练场景即完成。

在训练期间，处于外部的 Python 进程会在训练过程中与 Academy 不断进行通信以便运行一系列场景，同时会收集数据并优化其神经网络模型。分配给 agent 的 Brain 类型决定了我们是否进行训练。**External** Brain 会与外部过程进行通信以训练 TensorFlow 模型。成功完成训练后，您可以将经过训练的模型文件添加到您的 Unity 项目中，以便提供给 **Internal** Brain 来控制agent的行为。

ML-Agents Academy 类按如下方式编排 agent 模拟循环：

1. 调用您的 Academy 子类的 `AcademyReset()` 函数。
2. 对场景中的每个 agent 调用 `AgentReset()` 函数。
3. 对场景中的每个 agent 调用 `CollectObservations()` 函数。
4. 使用每个 agent 的 Brain 类来决定 agent 的下一动作。
5. 调用您的子类的 `AcademyAct()` 函数。
6. 对场景中的每个 agent 调用 `AgentAction()` 函数，传入由 agent 的 Brain 选择的动作。（如果 agent 已完成，则不调用此函数。）
7. 如果 agent 已达到其 `Max Step` 计数或者已将其自身标记为 `done`，则调用 agent 的 `AgentOnDone()` 函数。或者，如果某个 agent 在场景结束之前已完成，您可以将其设置为重新开始。在这种情况下，Academy 会调用 `AgentReset()` 函数。
8. 当 Academy 达到其自身的 `Max Step` 计数时，它会通过调用您的 Academy 子类的 `AcademyReset()` 函数来再次开始下一场景。

要创建训练环境，请扩展 Academy 和 Agent 类以实现上述方法。`Agent.CollectObservations()` 和 `Agent.AgentAction()` 函数必须实现；而其他方法是可选的，即是否需要实现它们取决于您的具体情况。

**注意：**在这里用到的 Python API 也可用于其他目的。例如，借助于该 API，您可以将 Unity 用作您自己的机器学习算法的模拟引擎。请参阅 [Python API](/docs/Python-API.md) 以了解更多信息。

## 组织 Unity 场景

为了在 Unity 场景中训练和使用 ML-Agents，该场景必须包含一个 Academy 子类实例，若干个 Brain 游戏对象和 Agent 子类实例。场景中的任何 Brain 游戏对象都必须附加到 Hierarchy 视图中的 Academy 游戏对象的子级下。Agent 子类实例应该附到代表该 agent 的游戏对象下。

[Screenshot of scene hierarchy]

您必须为每个 agent 分配一个 Brain，但可以在多个 agent 之间共享 Brain。每个 agent 都将进行自己的观测并独立行动，但会使用相同的决策逻辑，而对于 **Internal** Brain，则会使用相同的经过训练的 TensorFlow 模型。

### Academy

Academy 对象会指挥多个 agent 的决策过程。一个场景中有且仅能有一个Academy 对象。

您必须创建 Academy 类的子类（因为Academy类是抽象类）。创建 Academy 的子类时，你可以实现以下方法（全部都是可选方法）：

* `InitializeAcademy()` — 第一次启动时准备环境。
* `AcademyReset()` — 为下一轮的模拟准备环境和 agent。你可以使用这个函数在场景中放入并初始化实体。
* `AcademyStep()` — 为下一模拟步骤准备环境。Academy 基类首先调用此函数，然后才调用当前步骤的任何 `AgentAction()` 方法。您可以使用此函数在 agent 采取动作之前更新场景中的其他对象。请注意，在 Academy 调用此方法之前，agent 已收集了自己的观测结果并选择了动作。

Academy 基类还定义了若干可以在 Unity Editor Inspector 中设置的重要属性。对于训练而言，这些属性中最重要的是 `Max Steps`，它决定了每个训练场景的持续时间。Academy 的步骤计数器达到此值后，它将调用 `AcademyReset()` 函数来开始下一轮模拟。

  请参阅 [Academy](/docs/Learning-Environment-Design-Academy.md) 以查看 Academy 属性及其用途的完整列表。

### Brain

Brain 内部封装了决策过程。Brain 对象必须放在 Hierarchy 视图中的 Academy 的子级。我们必须为每个 Agent 分配一个 Brain，但可以在多个 Agent 之间共享同一个 Brain。

当我们使用 Brain 类的时候不需要使用其子类，而应该直接使用 Brain 这个类。Brain 的行为取决于 Brain 的类型。在训练期间，应将 agent 上连接的 Brain 的 Brain Type 设置为 **External**。要使用经过训练的模型，请将模型文件导入 Unity 项目，并将对应 Brain 的 Brain  Type 更改为 **Internal**。请参阅 [Brain](/docs/Learning-Environment-Design-Brains.md) 以了解有关使用不同类型的 Brain 的详细信息。如果四种内置的类型不能满足您的需求，您可以扩展 CoreBrain 类以创建其它的 Brain 类型。

Brain 类有若干可以使用 Inspector 窗口进行设置的重要属性。对于使用 Brain 的 agent，这些属性必须恰当。例如，`Vector Observation Space Size` 属性必须与 agent 创建的特征向量的长度完全匹配。请参阅 [Agent](/docs/Learning-Environment-Design-Agents.md) 以获取有关创建 agent 和正确设置 Brain 实例的信息。

请参阅 [Brain](/docs/Learning-Environment-Design-Brains.md) 以查看 Brain 属性的完整列表。

### Agent

Agent 类代表场景中负责收集观测结果并采取动作的一个参与者 (actor)。我们在配置的时候通常会把Agent 类的脚本附在这个参与者对应的游戏对象上。例如，附加到足球比赛中的球员对象，或车辆模拟中的汽车对象上。此外，必须为每个 Agent 类的脚本分配一个 Brain。

要创建 agent，请扩展 Agent 类并实现基本的 `CollectObservations()` 和 `AgentAction()` 方法：

* `CollectObservations()` — 收集 agent 对其环境的观测结果。
* `AgentAction()` — 执行由 agent 的 Brain 选择的动作，并为当前状态分配奖励。

这些函数的实现决定了分配给此 agent 的 Brain 的属性要如何设置。

您还必须确定 Agent 如何完成任务，以及当它超时后如何处理。agent 完成其任务（或彻底失败）后，您可以在 `AgentAction()` 函数中手动将 agent 设置为完成。您还可以将 agent 的 `Max Steps` 属性设置为正值，这样 agent 在执行了此数量的步骤后会认为自己已完成。Academy 达到自己的 `Max Steps` 计数后，会开始下一场景。如果将 agent 的 `ResetOnDone` 属性设置为 true，则 agent 可以在一个场景中多次尝试自己的任务。（在 `Agent.AgentReset()` 函数中可以设置 agent 的初始化逻辑，为下一次的任务做好准备。）

请参阅 [Agent](/docs/Learning-Environment-Design-Agents.md) 以详细了解如何编写一个你自己的 agent。

## 环境

ML-Agents 中的_环境_可以是 Unity 中构建的任何场景。Unity 场景为 agent 提供了观察、行动和学习的环境。如何设置 Unity 场景实际上取决于您的目标。您可能想要试图解决某个特定的reinforcement learning（强化学习）问题，这种情况下您可以在某一个场景内又进行训练又进行测试。或者，您可能想要训练 agent 在复杂的游戏或模拟条件下的做出某些行为，这种情况下创建更有针对性的训练场景可能会更加高效和实用。

训练和测试（或正常游戏）场景都必须包含一个 Academy 对象来控制 agent 的决策过程。Academy 定义了若干可以针对训练场景与测试场景进行不同设置的属性。Academy 的 **Configuration** 属性用于控制渲染和时间刻度。您可以设置 **Training Configuration** 来最大限度缩短 Unity 用于渲染图形的时间，从而加快训练速度。您可能还需要调整其他 Academy 功能设置。例如，`Max Steps` 的大小应尽可能小，从而尽量缩短训练时间，但也不能太小，必须大到足以让 agent 完成任务并在学习过程中有一些额外的“徘徊思考”(wandering) 时间。在测试场景中，您通常根本不希望 Academy 重置场景；如果是这样，应将 `Max Steps` 设置为零。

在 Unity 中创建训练环境时，必须设置场景以便可以通过外部训练过程来控制场景。注意以下几点：

* 在训练程序启动后，Unity 可执行文件会被自动打开，然后训练场景会自动开始训练。
* 场景中至少须包括一个 **External** Brain。
* Academy 必须在每一轮训练后将场景重置为有效的初始状态。
* 训练场景必须有明确的结束状态，为此需要使用 `Max Steps`，或让每个 agent 将自身设置为 `done`。


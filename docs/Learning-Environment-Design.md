# Reinforcement Learning in Unity

Reinforcement learning is an artificial intelligence technique that trains
_agents_ to perform tasks by rewarding desirable behavior. During reinforcement
learning, an agent explores its environment, observes the state of things, and,
based on those observations, takes an action. If the action leads to a better
state, the agent receives a positive reward. If it leads to a less desirable
state, then the agent receives no reward or a negative reward (punishment). As
the agent learns during training, it optimizes its decision making so that it
receives the maximum reward over time.

The ML-Agents toolkit uses a reinforcement learning technique called
[Proximal Policy Optimization (PPO)](https://blog.openai.com/openai-baselines-ppo/).
PPO uses a neural network to approximate the ideal function that maps an agent's
observations to the best action an agent can take in a given state. The
ML-Agents PPO algorithm is implemented in TensorFlow and runs in a separate
Python process (communicating with the running Unity application over a socket).

**Note:** if you aren't studying machine and reinforcement learning as a subject
and just want to train agents to accomplish tasks, you can treat PPO training as
a _black box_. There are a few training-related parameters to adjust inside
Unity as well as on the Python training side, but you do not need in-depth
knowledge of the algorithm itself to successfully create and train agents.
Step-by-step procedures for running the training process are provided in the
[Training section](Training-ML-Agents.md).

## The Simulation and Training Process

Training and simulation proceed in steps orchestrated by the ML-Agents Academy
class. The Academy works with Agent objects in the scene to step
through the simulation. When either the Academy has reached its maximum number
of steps or all Agents in the scene are _done_, one training episode is
finished.

During training, the external Python training process communicates with the
Academy to run a series of episodes while it collects data and optimizes its
neural network model. The kind of Brain assigned to an Agent determines whether
it participates in training or not. The **Learning Brain** can be used to train 
or execute a TensorFlow model. When training is completed
successfully, you can add the trained model file to your Unity project for later
use.

The ML-Agents Academy class orchestrates the agent simulation loop as follows:

1. Calls your Academy subclass's `AcademyReset()` function.
2. Calls the `AgentReset()` function for each Agent in the scene.
3. Calls the  `CollectObservations()` function for each Agent in the scene.
4. Uses each Agent's Brain to decide on the Agent's next action.
5. Calls your subclass's `AcademyStep()` function.
6. Calls the `AgentAction()` function for each Agent in the scene, passing in
   the action chosen by the Agent's Brain. (This function is not called if the
   Agent is done.)
7. Calls the Agent's `AgentOnDone()` function if the Agent has reached its `Max
   Step` count or has otherwise marked itself as `done`. Optionally, you can set
   an Agent to restart if it finishes before the end of an episode. In this
   case, the Academy calls the `AgentReset()` function.
8. When the Academy reaches its own `Max Step` count, it starts the next episode
   again by calling your Academy subclass's `AcademyReset()` function.

To create a training environment, extend the Academy and Agent classes to
implement the above methods. The `Agent.CollectObservations()` and
`Agent.AgentAction()` functions are required; the other methods are optional —
whether you need to implement them or not depends on your specific scenario.

**Note:** The API used by the Python PPO training process to communicate with
and control the Academy during training can be used for other purposes as well.
For example, you could use the API to use Unity as the simulation engine for
your own machine learning algorithms. See [Python API](Python-API.md) for more
information.

## Organizing the Unity Scene

To train and use the ML-Agents toolkit in a Unity scene, the scene must contain
a single Academy subclass and as many Agent subclasses
as you need. The Brain assets are present in the project and should be grouped 
together and named according to the type of agents they are compatible with.
Agent instances should be attached to the GameObject representing that Agent.

You must assign a Brain to every Agent, but you can share Brains between
multiple Agents. Each Agent will make its own observations and act
independently, but will use the same decision-making logic and, for **Learning
Brains**, the same trained TensorFlow model.

### Academy

The Academy object orchestrates Agents and their decision making processes. Only
place a single Academy object in a scene.

You must create a subclass of the Academy class (since the base class is
abstract). When you create your Academy subclass, you can implement the
following methods (all are optional):

* `InitializeAcademy()` — Prepare the environment the first time it launches.
* `AcademyReset()` — Prepare the environment and Agents for the next training
  episode. Use this function to place and initialize entities in the scene as
  necessary.
* `AcademyStep()` — Prepare the environment for the next simulation step. The
  base Academy class calls this function before calling any `AgentAction()`
  methods for the current step. You can use this function to update other
  objects in the scene before the Agents take their actions. Note that the
  Agents have already collected their observations and chosen an action before
  the Academy invokes this method.

The base Academy classes also defines several important properties that you can
set in the Unity Editor Inspector. For training, the most important of these
properties is `Max Steps`, which determines how long each training episode
lasts. Once the Academy's step counter reaches this value, it calls the
`AcademyReset()` function to start the next episode.

See [Academy](Learning-Environment-Design-Academy.md) for a complete list of
the Academy properties and their uses.

### Brain

The Brain encapsulates the decision making process. Every Agent must be
assigned a Brain, but you can use the same Brain with more than one Agent.
__Note__:You can assign the same Brain to multiple agents by using prefabs
or by selecting all the agents you want to attach the Brain to using the 
search bar on top of the Scene Hierarchy window.

To Create a Brain, go to `Assets -> Create -> Ml-Agents` and select the 
type of Brain you want to use. During training, use a **Learning Brain** 
and drag it into the Academy's `Broadcast Hub` with the `Control` checkbox checked.
When you want to use the trained model, import the model file into the Unity
project, add it to the **Model** property of the **Learning Brain** and uncheck
the `Control` checkbox of the `Broadcast Hub`. See
[Brains](Learning-Environment-Design-Brains.md) for details on using the
different types of Brains. You can create new kinds of Brains if the three
built-in don't do what you need.

The Brain class has several important properties that you can set using the
Inspector window. These properties must be appropriate for the Agents using the
Brain. For example, the `Vector Observation Space Size` property must match the
length of the feature vector created by an Agent exactly. See
[Agents](Learning-Environment-Design-Agents.md) for information about creating
agents and setting up a Brain instance correctly.

See [Brains](Learning-Environment-Design-Brains.md) for a complete list of the
Brain properties.

### Agent

The Agent class represents an actor in the scene that collects observations and
carries out actions. The Agent class is typically attached to the GameObject in
the scene that otherwise represents the actor — for example, to a player object
in a football game or a car object in a vehicle simulation. Every Agent must be
assigned a Brain.

To create an Agent, extend the Agent class and implement the essential
`CollectObservations()` and `AgentAction()` methods:

* `CollectObservations()` — Collects the Agent's observation of its environment.
* `AgentAction()` — Carries out the action chosen by the Agent's Brain and
  assigns a reward to the current state.

Your implementations of these functions determine how the properties of the
Brain assigned to this Agent must be set.

You must also determine how an Agent finishes its task or times out. You can
manually set an Agent to done in your `AgentAction()` function when the Agent
has finished (or irrevocably failed) its task. You can also set the Agent's `Max
Steps` property to a positive value and the Agent will consider itself done
after it has taken that many steps. When the Academy reaches its own `Max Steps`
count, it starts the next episode. If you set an Agent's `ResetOnDone` property
to true, then the Agent can attempt its task several times in one episode. (Use
the `Agent.AgentReset()` function to prepare the Agent to start again.)

See [Agents](Learning-Environment-Design-Agents.md) for detailed information
about programing your own Agents.

## Environments

An _environment_ in the ML-Agents toolkit can be any scene built in Unity. The
Unity scene provides the environment in which agents observe, act, and learn.
How you set up the Unity scene to serve as a learning environment really depends
on your goal. You may be trying to solve a specific reinforcement learning
problem of limited scope, in which case you can use the same scene for both
training and for testing trained agents. Or, you may be training agents to
operate in a complex game or simulation. In this case, it might be more
efficient and practical to create a purpose-built training scene.

Both training and testing (or normal game) scenes must contain an Academy object
to control the agent decision making process. The Academy defines several
properties that can be set differently for a training scene versus a regular
scene. The Academy's **Configuration** properties control rendering and time
scale. You can set the **Training Configuration** to minimize the time Unity
spends rendering graphics in order to speed up training. You may need to adjust
the other functional, Academy settings as well. For example, `Max Steps` should
be as short as possible for training — just long enough for the agent to
accomplish its task, with some extra time for "wandering" while it learns. In
regular scenes, you often do not want the Academy to reset the scene at all; if
so, `Max Steps` should be set to zero.

When you create a training environment in Unity, you must set up the scene so
that it can be controlled by the external training process. Considerations
include:

* The training scene must start automatically when your Unity application is
  launched by the training process.
* The scene must include an Academy with at least one Brain in the `Broadcast Hub`
  with the `Control` checkbox checked.
* The Academy must reset the scene to a valid starting point for each episode of
  training.
* A training episode must have a definite end — either using `Max Steps` or by
  each Agent setting itself to `done`.

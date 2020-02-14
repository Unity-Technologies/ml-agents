# Getting Started with the 3D Balance Ball Environment

This tutorial walks through the end-to-end process of opening a ML-Agents
toolkit example environment in Unity, building the Unity executable, training an
Agent in it, and finally embedding the trained model into the Unity environment.

The ML-Agents toolkit includes a number of [example
environments](Learning-Environment-Examples.md) which you can examine to help
understand the different ways in which the ML-Agents toolkit can be used. These
environments can also serve as templates for new environments or as ways to test
new ML algorithms. After reading this tutorial, you should be able to explore
and build the example environments.

![3D Balance Ball](images/balance.png)

This walk-through uses the **3D Balance Ball** environment. 3D Balance Ball
contains a number of agent cubes and balls (which are all copies of each other).
Each agent cube tries to keep its ball from falling by rotating either
horizontally or vertically. In this environment, an agent cube is an **Agent** that
receives a reward for every step that it balances the ball. An agent is also
penalized with a negative reward for dropping the ball. The goal of the training
process is to have the agents learn to balance the ball on their head.

Let's get started!

## Installation

In order to install and set up the ML-Agents toolkit, the Python dependencies
and Unity, see the [installation instructions](Installation.md).

## Understanding the Unity Environment (3D Balance Ball)

An agent is an autonomous actor that observes and interacts with an
_environment_. In the context of Unity, an environment is a scene containing an
Academy and one or more Agent objects, and, of course, the other
entities that an agent interacts with.

![Unity Editor](images/mlagents-3DBallHierarchy.png)

**Note:** In Unity, the base object of everything in a scene is the
_GameObject_. The GameObject is essentially a container for everything else,
including behaviors, graphics, physics, etc. To see the components that make up
a GameObject, select the GameObject in the Scene window, and open the Inspector
window. The Inspector shows every component on a GameObject.

The first thing you may notice after opening the 3D Balance Ball scene is that
it contains not one, but several agent cubes.  Each agent cube in the scene is an
independent agent, but they all share the same Behavior. 3D Balance Ball does this
to speed up training since all twelve agents contribute to training in parallel.


### Agent

The Agent is the actor that observes and takes actions in the environment. In
the 3D Balance Ball environment, the Agent components are placed on the twelve
"Agent" GameObjects. The base Agent object has a few properties that affect its
behavior:

* **Behavior Parameters** — Every Agent must have a Behavior. The Behavior
  determines how an Agent makes decisions. More on Behavior Parameters in
  the next section.
* **Max Step** — Defines how many simulation steps can occur before the Agent
  decides it is done. In 3D Balance Ball, an Agent restarts after 5000 steps.

Perhaps the more interesting aspect of an agents is the Agent subclass
implementation. When you create an Agent, you must extend the base Agent class.
The Ball3DAgent subclass defines the following methods:

* agent.AgentReset() — Called when the Agent resets, including at the beginning
  of a session. The Ball3DAgent class uses the reset function to reset the
  agent cube and ball. The function randomizes the reset values so that the
  training generalizes to more than a specific starting position and agent cube
  attitude.
* agent.CollectObservations(VectorSensor sensor) — Called every simulation step. Responsible for
  collecting the Agent's observations of the environment. Since the Behavior
  Parameters of the Agent are set with vector observation
  space with a state size of 8, the `CollectObservations(VectorSensor sensor)` must call
  `VectorSensor.AddObservation()` such that vector size adds up to 8.
* agent.AgentAction() — Called every simulation step. Receives the action chosen
  by the Agent. The vector action spaces result in a
  small change in the agent cube's rotation at each step. The `AgentAction()` function
  assigns a reward to the Agent; in this example, an Agent receives a small
  positive reward for each step it keeps the ball on the agent cube's head and a larger,
  negative reward for dropping the ball. An Agent is also marked as done when it
  drops the ball so that it will reset with a new ball for the next simulation
  step.
* agent.Heuristic() - When the `Use Heuristic` checkbox is checked in the Behavior
  Parameters of the Agent, the Agent will use the `Heuristic()` method to generate
  the actions of the Agent. As such, the `Heuristic()` method returns an array of
  floats. In the case of the Ball 3D Agent, the `Heuristic()` method converts the
  keyboard inputs into actions.


#### Behavior Parameters : Vector Observation Space

Before making a decision, an agent collects its observation about its state in
the world. The vector observation is a vector of floating point numbers which
contain relevant information for the agent to make decisions.

The Behavior Parameters of the 3D Balance Ball example uses a **Space Size** of 8.
This means that the feature
vector containing the Agent's observations contains eight elements: the `x` and
`z` components of the agent cube's rotation and the `x`, `y`, and `z` components
of the ball's relative position and velocity. (The observation values are
defined in the Agent's `CollectObservations(VectorSensor sensor)` method.)

#### Behavior Parameters : Vector Action Space

An Agent is given instructions in the form of a float array of *actions*.
ML-Agents toolkit classifies actions into two types: the **Continuous** vector
action space is a vector of numbers that can vary continuously. What each
element of the vector means is defined by the Agent logic (the training
process just learns what values are better given particular state observations
based on the rewards received when it tries different values). For example, an
element might represent a force or torque applied to a `Rigidbody` in the Agent.
The **Discrete** action vector space defines its actions as tables. An action
given to the Agent is an array of indices into tables.

The 3D Balance Ball example is programmed to use continuous action
space with `Space Size` of 2.

## Training with Reinforcement Learning

Now that we have an environment, we can perform the training.

### Training with Deep Reinforcement Learning

In order to train an agent to correctly balance the ball, we provide two
deep reinforcement learning algorithms.

The default algorithm is Proximal Policy Optimization (PPO). This
is a method that has been shown to be more general purpose and stable
than many other RL algorithms. For more information on PPO, OpenAI
has a [blog post](https://blog.openai.com/openai-baselines-ppo/)
explaining it, and [our page](Training-PPO.md) for how to use it in training.

We also provide Soft-Actor Critic, an off-policy algorithm that
has been shown to be both stable and sample-efficient.
For more information on SAC, see UC Berkeley's
[blog post](https://bair.berkeley.edu/blog/2018/12/14/sac/) and
[our page](Training-SAC.md) for more guidance on when to use SAC vs. PPO. To
use SAC to train Balance Ball, replace all references to `config/trainer_config.yaml`
with `config/sac_trainer_config.yaml` below.

To train the agents within the Balance Ball environment, we will be using the
ML-Agents Python package. We have provided a convenient command called `mlagents-learn`
which accepts arguments used to configure both training and inference phases.

We can use `run_id` to identify the experiment and create a folder where the
model and summary statistics are stored. When using TensorBoard to observe the
training statistics, it helps to set this to a sequential value for each
training run. In other words, "BalanceBall1" for the first run, "BalanceBall2"
or the second, and so on. If you don't, the summaries for every training run are
saved to the same directory and will all be included on the same graph.

To summarize, go to your command line, enter the `ml-agents` directory and type:

```sh
mlagents-learn config/trainer_config.yaml --run-id=<run-identifier> --train --time-scale=100
```

When the message _"Start training by pressing the Play button in the Unity
Editor"_ is displayed on the screen, you can press the :arrow_forward: button in
Unity to start training in the Editor.

**Note**: If you're using Anaconda, don't forget to activate the ml-agents
environment first.

The `--train` flag tells the ML-Agents toolkit to run in training mode.
The `--time-scale=100` sets the `Time.TimeScale` value in Unity.

**Note**: You can train using an executable rather than the Editor. To do so,
follow the instructions in
[Using an Executable](Learning-Environment-Executable.md).

**Note**: Re-running this command will start training from scratch again. To resume
a previous training run, append the `--load` flag and give the same `--run-id` as the
run you want to resume.

### Observing Training Progress

Once you start training using `mlagents-learn` in the way described in the
previous section, the `ml-agents` directory will contain a `summaries`
directory. In order to observe the training process in more detail, you can use
TensorBoard. From the command line run:

```sh
tensorboard --logdir=summaries
```

Then navigate to `localhost:6006` in your browser.

From TensorBoard, you will see the summary statistics:

* Lesson - only interesting when performing [curriculum
  training](Training-Curriculum-Learning.md). This is not used in the 3D Balance
  Ball environment.
* Cumulative Reward - The mean cumulative episode reward over all agents. Should
  increase during a successful training session.
* Entropy - How random the decisions of the model are. Should slowly decrease
  during a successful training process. If it decreases too quickly, the `beta`
  hyperparameter should be increased.
* Episode Length - The mean length of each episode in the environment for all
  agents.
* Learning Rate - How large a step the training algorithm takes as it searches
  for the optimal policy. Should decrease over time.
* Policy Loss - The mean loss of the policy function update. Correlates to how
  much the policy (process for deciding actions) is changing. The magnitude of
  this should decrease during a successful training session.
* Value Estimate - The mean value estimate for all states visited by the agent.
  Should increase during a successful training session.
* Value Loss - The mean loss of the value function update. Correlates to how
  well the model is able to predict the value of each state. This should
  decrease during a successful training session.

![Example TensorBoard Run](images/mlagents-TensorBoard.png)

## Embedding the Model into the Unity Environment

Once the training process completes, and the training process saves the model
(denoted by the `Saved Model` message) you can add it to the Unity project and
use it with compatible Agents (the Agents that generated the model).
__Note:__ Do not just close the Unity Window once the `Saved Model` message appears.
Either wait for the training process to close the window or press Ctrl+C at the
command-line prompt. If you close the window manually, the `.nn` file
containing the trained model is not exported into the ml-agents folder.

### Embedding the trained model into Unity

To embed the trained model into Unity, follow the later part of [Training the
Model with Reinforcement
Learning](Basic-Guide.md#training-the-model-with-reinforcement-learning) section
of the Basic Guide page.

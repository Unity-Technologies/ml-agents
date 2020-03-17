# Getting Started Guide

This guide walks through the end-to-end process of opening an ML-Agents
toolkit example environment in Unity, building the Unity executable, training an
Agent in it, and finally embedding the trained model into the Unity environment.

The ML-Agents toolkit includes a number of [example
environments](Learning-Environment-Examples.md) which you can examine to help
understand the different ways in which the ML-Agents toolkit can be used. These
environments can also serve as templates for new environments or as ways to test
new ML algorithms. After reading this tutorial, you should be able to explore
train the example environments.

If you are not familiar with the [Unity Engine](https://unity3d.com/unity), we
highly recommend the [Roll-a-ball
tutorial](https://unity3d.com/learn/tutorials/s/roll-ball-tutorial) to learn all
the basic concepts first.

![3D Balance Ball](images/balance.png)

This guide uses the **3D Balance Ball** environment to teach the basic concepts and
usage patterns of ML-Agents. 3D Balance Ball
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

Depending on your version of Unity, it may be necessary to change the **Scripting Runtime Version** of your project. This can be done as follows:

1. Launch Unity
2. On the Projects dialog, choose the **Open** option at the top of the window.
3. Using the file dialog that opens, locate the `Project` folder
   within the ML-Agents toolkit project and click **Open**.
4. Go to **Edit** > **Project Settings** > **Player**
5. For **each** of the platforms you target (**PC, Mac and Linux Standalone**,
   **iOS** or **Android**):
    1. Expand the **Other Settings** section.
    2. Select **Scripting Runtime Version** to **Experimental (.NET 4.6
       Equivalent or .NET 4.x Equivalent)**
6. Go to **File** > **Save Project**


## Understanding a Unity Environment

An agent is an autonomous actor that observes and interacts with an
_environment_. In the context of Unity, an environment is a scene containing
one or more Agent objects, and, of course, the other
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
* **Max Step** — Defines how many simulation steps can occur before the Agent's
  episode ends. In 3D Balance Ball, an Agent restarts after 5000 steps.

When you create an Agent, you must extend the base Agent class.
The Ball3DAgent subclass defines the following methods:

* `Agent.OnEpisodeBegin()` — Called at the beginning of an Agent's episode, including at the beginning
  of the simulation. The Ball3DAgent class uses this function to reset the
  agent cube and ball to their starting positions. The function randomizes the reset values so that the
  training generalizes to more than a specific starting position and agent cube
  attitude.
* `Agent.CollectObservations(VectorSensor sensor)` — Called every simulation step. Responsible for
  collecting the Agent's observations of the environment. Since the Behavior
  Parameters of the Agent are set with vector observation
  space with a state size of 8, the `CollectObservations(VectorSensor sensor)` must call
  `VectorSensor.AddObservation()` such that vector size adds up to 8.
* `Agent.OnActionReceived()` — Called every time the Agent receives an action to take. Receives the action chosen
  by the Agent. The vector action spaces result in a
  small change in the agent cube's rotation at each step. The `OnActionReceived()` method
  assigns a reward to the Agent; in this example, an Agent receives a small
  positive reward for each step it keeps the ball on the agent cube's head and a larger,
  negative reward for dropping the ball. An Agent's episode is also ended when it
  drops the ball so that it will reset with a new ball for the next simulation
  step.
* `Agent.Heuristic()` - When the `Behavior Type` is set to `Heuristic Only` in the Behavior
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

## Running a pre-trained model

We include pre-trained models for our agents (`.nn` files) and we use the
[Unity Inference Engine](Unity-Inference-Engine.md) to run these models
inside Unity. In this section, we will use the pre-trained model for the
3D Ball example.

1. In the **Project** window, go to the `Assets/ML-Agents/Examples/3DBall/Scenes` folder
   and open the `3DBall` scene file.
2. In the **Project** window, go to the `Assets/ML-Agents/Examples/3DBall/Prefabs` folder.
   Expand `3DBall` and click on the `Agent` prefab.  You should see the `Agent` prefab in the **Inspector** window.

   **Note**: The platforms in the `3DBall` scene were created using the `3DBall` prefab.  Instead of updating all 12 platforms individually, you can update the `3DBall` prefab instead.

   ![Platform Prefab](images/platform_prefab.png)

3. In the **Project** window, drag the **3DBall** Model located in
   `Assets/ML-Agents/Examples/3DBall/TFModels` into the `Model` property under `Behavior Parameters (Script)` component in the Agent GameObject **Inspector** window.

   ![3dball learning brain](images/3dball_learning_brain.png)

4. You should notice that each `Agent` under each `3DBall` in the **Hierarchy** windows now contains **3DBall** as `Model` on the `Behavior Parameters`. __Note__ : You can modify multiple game objects in a scene by selecting them all at
   once using the search bar in the Scene Hierarchy.
8. Select the **InferenceDevice** to use for this model (CPU or GPU) on the Agent.
   _Note: CPU is faster for the majority of ML-Agents toolkit generated models_
9. Click the **Play** button and you will see the platforms balance the balls
   using the pre-trained model.

## Training a new model with Reinforcement Learning

While we provide pre-trained `.nn` files for the agents in this environment, any environment you make yourself will require training agents from scratch to generate a new model file. We can do this using reinforcement learning.

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

### Training the environment

1. Open a command or terminal window.
2. Navigate to the folder where you cloned the ML-Agents toolkit repository.
   **Note**: If you followed the default [installation](Installation.md), then
   you should be able to run `mlagents-learn` from any directory.
3. Run `mlagents-learn <trainer-config-path> --run-id=<run-identifier> --train`
   where:
    - `<trainer-config-path>` is the relative or absolute filepath of the
      trainer configuration. The defaults used by example environments included
      in `MLAgentsSDK` can be found in `config/trainer_config.yaml`.
    - `<run-identifier>` is a string used to separate the results of different
      training runs
    - `--train` tells `mlagents-learn` to run a training session (rather
      than inference)
4. If you cloned the ML-Agents repo, then you can simply run

      ```sh
      mlagents-learn config/trainer_config.yaml --run-id=firstRun --train
      ```

5. When the message _"Start training by pressing the Play button in the Unity
   Editor"_ is displayed on the screen, you can press the :arrow_forward: button
   in Unity to start training in the Editor.

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

If `mlagents-learn` runs correctly and starts training, you should see something
like this:

```console
INFO:mlagents_envs:
'Ball3DAcademy' started successfully!
Unity Academy name: Ball3DAcademy

INFO:mlagents_envs:Connected new brain:
Unity brain name: 3DBallLearning
        Number of Visual Observations (per agent): 0
        Vector Observation space size (per agent): 8
        Number of stacked Vector Observation: 1
        Vector Action space type: continuous
        Vector Action space size (per agent): [2]
        Vector Action descriptions: ,
INFO:mlagents_envs:Hyperparameters for the PPO Trainer of brain 3DBallLearning:
        batch_size:          64
        beta:                0.001
        buffer_size:         12000
        epsilon:             0.2
        gamma:               0.995
        hidden_units:        128
        lambd:               0.99
        learning_rate:       0.0003
        max_steps:           5.0e4
        normalize:           True
        num_epoch:           3
        num_layers:          2
        time_horizon:        1000
        sequence_length:     64
        summary_freq:        1000
        use_recurrent:       False
        summary_path:        ./summaries/first-run-0
        memory_size:         256
        use_curiosity:       False
        curiosity_strength:  0.01
        curiosity_enc_size:  128
        model_path:	./models/first-run-0/3DBallLearning
INFO:mlagents.trainers: first-run-0: 3DBallLearning: Step: 1000. Mean Reward: 1.242. Std of Reward: 0.746. Training.
INFO:mlagents.trainers: first-run-0: 3DBallLearning: Step: 2000. Mean Reward: 1.319. Std of Reward: 0.693. Training.
INFO:mlagents.trainers: first-run-0: 3DBallLearning: Step: 3000. Mean Reward: 1.804. Std of Reward: 1.056. Training.
INFO:mlagents.trainers: first-run-0: 3DBallLearning: Step: 4000. Mean Reward: 2.151. Std of Reward: 1.432. Training.
INFO:mlagents.trainers: first-run-0: 3DBallLearning: Step: 5000. Mean Reward: 3.175. Std of Reward: 2.250. Training.
INFO:mlagents.trainers: first-run-0: 3DBallLearning: Step: 6000. Mean Reward: 4.898. Std of Reward: 4.019. Training.
INFO:mlagents.trainers: first-run-0: 3DBallLearning: Step: 7000. Mean Reward: 6.716. Std of Reward: 5.125. Training.
INFO:mlagents.trainers: first-run-0: 3DBallLearning: Step: 8000. Mean Reward: 12.124. Std of Reward: 11.929. Training.
INFO:mlagents.trainers: first-run-0: 3DBallLearning: Step: 9000. Mean Reward: 18.151. Std of Reward: 16.871. Training.
INFO:mlagents.trainers: first-run-0: 3DBallLearning: Step: 10000. Mean Reward: 27.284. Std of Reward: 28.667. Training.
```


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

* **Lesson** - only interesting when performing [curriculum
  training](Training-Curriculum-Learning.md). This is not used in the 3D Balance
  Ball environment.
* **Cumulative Reward** - The mean cumulative episode reward over all agents. Should
  increase during a successful training session.
* **Entropy** - How random the decisions of the model are. Should slowly decrease
  during a successful training process. If it decreases too quickly, the `beta`
  hyperparameter should be increased.
* **Episode Length** - The mean length of each episode in the environment for all
  agents.
* **Learning Rate** - How large a step the training algorithm takes as it searches
  for the optimal policy. Should decrease over time.
* **Policy Loss** - The mean loss of the policy function update. Correlates to how
  much the policy (process for deciding actions) is changing. The magnitude of
  this should decrease during a successful training session.
* **Value Estimate** - The mean value estimate for all states visited by the agent.
  Should increase during a successful training session.
* **Value Loss** - The mean loss of the value function update. Correlates to how
  well the model is able to predict the value of each state. This should
  decrease during a successful training session.

![Example TensorBoard Run](images/mlagents-TensorBoard.png)

## Embedding the model into the Unity Environment

Once the training process completes, and the training process saves the model
(denoted by the `Saved Model` message) you can add it to the Unity project and
use it with compatible Agents (the Agents that generated the model).
__Note:__ Do not just close the Unity Window once the `Saved Model` message appears.
Either wait for the training process to close the window or press Ctrl+C at the
command-line prompt. If you close the window manually, the `.nn` file
containing the trained model is not exported into the ml-agents folder.

You can press Ctrl+C to stop the training, and your trained model will be at
`models/<run-identifier>/<behavior_name>.nn` where
`<behavior_name>` is the name of the `Behavior Name` of the agents corresponding to the model.
(**Note:** There is a known bug on Windows that causes the saving of the model to
fail when you early terminate the training, it's recommended to wait until Step
has reached the max_steps parameter you set in trainer_config.yaml.) This file
corresponds to your model's latest checkpoint. You can now embed this trained
model into your Agents by following the steps below, which is similar to
the steps described
[above](#running-a-pre-trained-model).

1. Move your model file into
   `Project/Assets/ML-Agents/Examples/3DBall/TFModels/`.
2. Open the Unity Editor, and select the **3DBall** scene as described above.
3. Select the  **3DBall** prefab Agent object.
4. Drag the `<behavior_name>.nn` file from the Project window of
   the Editor to the **Model** placeholder in the **Ball3DAgent**
   inspector window.
5. Press the :arrow_forward: button at the top of the Editor.

## Next Steps

- For more information on the ML-Agents toolkit, in addition to helpful
  background, check out the [ML-Agents Toolkit Overview](ML-Agents-Overview.md)
  page.
- For a "Hello World" introduction to creating your own Learning Environment,
  check out the [Making a New Learning
  Environment](Learning-Environment-Create-New.md) page.
- For a series of YouTube video tutorials, checkout the
  [Machine Learning Agents PlayList](https://www.youtube.com/playlist?list=PLX2vGYjWbI0R08eWQkO7nQkGiicHAX7IX)
  page.

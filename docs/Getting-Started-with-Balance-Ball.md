# Getting Started with the 3D Balance Ball Example

This tutorial walks through the end-to-end process of opening an ML Agents example environment in Unity, building the Unity executable, training an agent in it, and finally embedding the trained model into the Unity environment. 

The Unity ML Agents SDK contains a number of example environments which you can examine to help understand the different ways in which ML Agents can be used. These environments can also serve as templates for new environments or as ways to test new ML algorithms. After reading this tutorial, you should be able to explore and build the example environments.

![Balance Ball](images/balance.png)

This walkthrough uses the **3D Balance Ball** environment. 3D Balance Ball contains a number of platforms and balls (which are all copies of each other). Each platform tries to keep its ball from falling by rotating either horizontally or vertically. In this environment, a platform is an **agent** that receives a reward for every step that it balances the ball. An agent is also penalized with a negative reward for dropping the ball. The goal of the training process is to have the platforms learn to never drop the ball.

Let's get started!

## Installation

In order to install and set up ML Agents, the Python dependencies and Unity, see the [installation instructions](installation.md).

## Understanding a Unity Environment (Balance Ball)

An agent is an autonomous actor that observes and interacts with an _environment_. In the context of Unity, an environment is a scene containing an Academy and one or more Brain and Agent objects, and, of course, the other entities that an agent interacts with.  

[screenshot showing Scene Hierarchy]

**Note:** In Unity, the base object of everything in a scene is the _GameObject_. The GameObject is essentially a container for everything else, including behaviors, graphics, physics, etc. To see the components that make up a GameObject, select the GameObject in the Scene window, and open the Inspector window. The Inspector shows every component on a GameObject. 
 
The first thing you may notice after opening the 3D Balance Ball scene is that it contains not one, but several platforms.  Each platform in the scene is an independent agent, but they all share the same brain.  Balance Ball does this to speed up training since all twelve agents contribute to training in parallel.

### Academy

The Academy object for the scene is placed on the Ball3DAcademy GameObject. When you look at an Academy component in the inspector, you can see several properties that control how the environment works. For example, the __Training__ and __Inference Configuration__  properties set the graphics and timescale properties for the Unity application. The Academy uses the __Training Configuration__  during training and the __Inference Configuration__ when not training. (*Inference* means that the agent is using a trained model or heuristics or direct control — in other words, whenever **not** training.) Typically, you set low graphics quality and a high time scale for the **Training configuration** and a high graphics quality and the timescale to `1.0` for the **Inference Configuration** .

**Note:** if you want to observe the enviornment during training, you can adjust the __Inference Configuration__ settings to use a larger window and a timescale closer to 1:1. Be sure to set these parameters back when training in earnest; otherwise, training can take a very long time.

Another aspect of an environment to look at is the Academy implementation.  Since the base Academy class is abstract, you must always define a subclass. There are three functions you can implement, though they are all optional:

* Academy.InitializeAcademy() — Called once when the environment is launched.
* Academy.AcademyStep() — Called at every simulation step before Agent.AgentStep() (and after the agents collect their state observations).
* Academy.AcademyReset() — Called when the Academy starts or restarts the simulation (including the first time).

The 3D Balance Ball environment does not use these functions — each agent resets itself when needed — but many environments do use these functions to control the environment around the agents.

### Brain

The Ball3DBrain GameObject in the scene, which contains a Brain component, is a child of the Academy object. (All Brain objects in a scene must be children of the Academy.) All the agents in the 3D Balance Ball environment use the same Brain instance. A Brain doesn't save any state about an agent, it just routes the agent's collected state observations to the decision making process and returns the chosen action to the agent. Thus, all agents can share the same brain, but act independently. The Brain settings tell you quite a bit about how an agent works.

The __Brain Type__ determines how an agent makes its decisions. The **External** and **Internal** types work together — use **External** when training your agents; use **Internal** when using the trained model. The **Heuristic** brain allows you to hand-code the agent's logic by extending the Decision class. Finally, the **Player** brain lets you map keyboard commands to actions, which can be useful when testing your agents and environment. If none of these types of brains do what you need, you can implement your own CoreBrain to create your own type.

In this tutorial, you will set the __Brain Type__ to **External** for training; when you embed the trained model in the Unity application, you will change the __Brain Type__ to **Internal**.

__State Observation Space__

Before making a decision, an agent collects its observation about its state in the world. ML Agents classifies observations into two types: **Continuous** and **Discrete**. The **Continuous** state space collects observations in a vector of floating point numbers. The **Discrete** state space is an index into a table of states. Most of the example environments use a continuous state space. 

The Brain instance used in the 3D Balance Ball example uses the **Continuous** state space with a __State Size__ of 8. This means that the feature vector containing the agent's observations contains eight elements: the `x` and `z` components of the platform's rotation and the `x`, `y`, and `z` components of the ball's relative position and velocity. (The state values are defined in the agent's `CollectState()` function.)

__Action Space__

An agent is given instructions from the brain in the form of *actions*. Like states, ML Agents classifies actions into two types: the **Continuous** action space is a vector of numbers that can vary continuously. What each element of the vector means is defined by the agent logic (the PPO training process just learns what values are better given particular state observations based on the rewards received when it tries different values). For example, an element might represent a force or torque applied to a Rigidbody in the agent. The **Discrete** action space defines its actions as a table. A specific action given to the agent is an index into this table. 

The 3D Balance Ball example is programmed to use both types of action space. You can try training with both settings to observe whether there is a difference. (Set the `Action Size` to 4 when using the discrete action space and 2 when using continuous.)
 
### Agent

The Agent is the actor that observes and takes actions in the environment. In the 3D Balance Ball environment, the Agent components are placed on the twelve Platform GameObjects. The base Agent object has a few properties that affect its behavior:

* __Brain__ — Every agent must have a Brain. The brain determines how an agent makes decisions. All the agents in the 3D Balance Ball scene share the same brain.
* __Observations__ — Defines any Camera objects used by the agent to observe its environment. 3D Balance Ball does not use camera observations.
* __Max Step__ — Defines how many simulation steps can occur before the agent decides it is done. In 3D Balance Ball, an agent restarts after 5000 steps.
* __Reset On Done__ — Defines whether an agent starts over when it is finished. 3D Balance Ball sets this true so that the agent restarts after reaching the __Max Step__ count or after dropping the ball.

Perhaps the more interesting aspect of an agent is the Agent subclass implementation. When you create an agent, you must extend the base Agent class. The Ball3DAgent subclass defines the following methods:

* Agent.AgentReset() — Called when the Agent resets, including at the beginning of a session. The Ball3DAgent class uses the reset function to reset the platform and ball. The function randomizes the reset values so that the training generalizes to more than a specific starting position and platform attitude.
* Agent.CollectState() — Called every simulation step. Responsible for collecting the agent's observations of the environment. Since the Brain instance assigned to the agent is set to the continuous state space with a state size of 8, the `CollectState()` function returns a vector (technically a List<float> object) containing 8 elements.
* Agent.AgentStep() — Called every simulation step (unless the brain's `Frame Skip` property is > 0). Receives the action chosen by the brain. The Ball3DAgent example handles both the continuous and the discrete action space types. There isn't actually much difference between the two state types in this environment — both action spaces result in a small change in platform rotation at each step. The `AgentStep()` function assigns a reward to the agent; in this example, an agent receives a small positive reward for each step it keeps the ball on the platform and a larger, negative reward for dropping the ball. An agent is also marked as done when it drops the ball so that it will reset with a new ball for the next simulation step.

## Building the Environment

The first step is to open the Unity scene containing the 3D Balance Ball environment:

1. Launch Unity.
2. On the Projects dialog, choose the **Open** option at the top of the window.
3. Using the file dialog that opens, locate the `unity-environment` folder within the ML Agents project and click **Open**.
4. In the `Project` window, navigate to the folder `Assets/ML-Agents/Examples/3DBall/`.
5. Double-click the `Scene` file to load the scene containing the Balance Ball environment.

![3DBall Scene](images/mlagents-Open3DBall.png)

Since we are going to build this environment to conduct training, we need to set the brain used by the agents to **External**. This allows the agents to communicate with the external training process when making their decisions.

1. In the **Scene** window, click the triangle icon next to the Ball3DAcademy object.
2. Select its child object `Ball3DBrain`.
3. In the Inspector window, set **Brain Type** to `External`.

![Set Brain to External](images/mlagents-SetExternalBrain.png)

Next, we want the set up scene to to play correctly when the training process launches our environment executable. This means:
* The environment application runs in the background
* No dialogs require interaction
* The correct scene loads automatically
 
1. Open Player Settings (menu: **Edit** > **Project Settings** > **Player**).
2. Under **Resolution and Presentation**:
    - Ensure that **Run in Background** is Checked.
    - Ensure that **Display Resolution Dialog** is set to Disabled.
3. Open the Build Settings window (menu:**File** > **Build Settings**).
4. Choose your target platform.
    - (optional) Select “Development Build” to [log debug messages](https://docs.unity3d.com/Manual/LogFiles.html).
5. If any scenes are shown in the **Scenes in Build** list, make sure that the 3DBall Scene is the only one checked. (If the list is empty, than only the current scene is included in the build).
6. Click *Build*:
    a. In the File dialog, navigate to the `python` folder in your ML Agents directory.
    b. Assign a file name and click **Save**.

![Build Window](images/mlagents-BuildWindow.png)

## Training the Brain with Reinforcement Learning

Now that we have a Unity executable containing the simulation environment, we can perform the training. 

### Training with PPO

In order to train an agent to correctly balance the ball, we will use a Reinforcement Learning algorithm called Proximal Policy Optimization (PPO). This is a method that has been shown to be safe, efficient, and more general purpose than many other RL algorithms, as such we have chosen it as the example algorithm for use with ML Agents. For more information on PPO, OpenAI has a recent [blog post](https://blog.openai.com/openai-baselines-ppo/) explaining it.

In order to train the agents within the Ball Balance environment:

1. Open `python/PPO.ipynb` notebook from Jupyter.
2. Set `env_name` to the name of your environment file earlier.
3. (optional) In order to get the best results quickly, set `max_steps` to 50000, set `buffer_size` to 5000, and set `batch_size` to 512.  For this exercise, this will train the model in approximately ~5-10 minutes.
4. (optional) Set `run_path` directory to your choice. When using Tensorboard to observe the training statistics, it helps to set this to a sequential value for each training run. In other words, "BalanceBall1" for the first run, "BalanceBall2" or the second, and so on. If you don't, the summaries for every training run are saved to the same directory and will all be included on the same graph.
5. Run all cells of notebook with the exception of the last one under "Export the trained Tensorflow graph."

### Observing Training Progress
In order to observe the training process in more detail, you can use Tensorboard.
In your command line, enter into `python` directory and then run :

`tensorboard --logdir=summaries`

Then navigate to `localhost:6006`.

From Tensorboard, you will see the summary statistics:

* Lesson - only interesting when performing [curriculum training](link). This is not used in the 3d Balance Ball environment. 
* Cumulative Reward - The mean cumulative episode reward over all agents. Should increase during a successful training session.
* Entropy - How random the decisions of the model are. Should slowly decrease during a successful training process. If it decreases too quickly, the `beta` hyperparameter should be increased.
* Episode Length - The mean length of each episode in the environment for all agents.
* Learning Rate - How large a step the training algorithmn takes as it searches for the optimal policy. Should decrease over time.
* Policy Loss - The mean loss of the policy function update. Correlates to how much the policy (process for deciding actions) is changing. The magnitude of this should decrease during a successful training session.
* Value Estimate - The mean value estimate for all states visited by the agent. Should increase during a successful training session.
* Value Loss - The mean loss of the value function update. Correlates to how well the model is able to predict the value of each state. This should decrease during a successful training session.

![Example Tensorboard Run](images/mlagents-TensorBoard.png)

## Embedding the Trained Brain into the Unity Environment _[Experimental]_

Once the training process completes, and the training process saves the model (denoted by the `Saved Model` message) you can add it to the Unity project and use it with agents having an **Internal** brain type.

### Setting up TensorFlowSharp Support

Because TensorFlowSharp support is still experimental, it is disabled by default. In order to enable it, you must follow these steps. Please note that the `Internal` Brain mode will only be available once completing these steps.

1. Make sure you are using Unity 2017.1 or newer.
2. Make sure the TensorFlowSharp plugin is in your `Assets` folder. A Plugins folder which includes TF# can be downloaded [here](https://s3.amazonaws.com/unity-agents/0.2/TFSharpPlugin.unitypackage). Double click and import it once downloaded.  You can see if this was successfully installed by checking the TensorFlow files in the Project tab under `Assets` -> `ML-Agents` -> `Plugins` -> `Computer`
3. Go to `Edit` -> `Project Settings` -> `Player`
4. For each of the platforms you target (**`PC, Mac and Linux Standalone`**, **`iOS`** or **`Android`**):
	1. Go into `Other Settings`.
	2. Select `Scripting Runtime Version` to `Experimental (.NET 4.6 Equivalent)`
	3. In `Scripting Defined Symbols`, add the flag `ENABLE_TENSORFLOW`.  After typing in, press Enter.
5. Go to `File` -> `Save Project`
6. Restart the Unity Editor.

### Embedding the trained model into Unity

1. Run the final cell of the notebook under "Export the trained TensorFlow graph" to produce an `<env_name >.bytes` file.
2. Move `<env_name>.bytes` from `python/models/ppo/` into `unity-environment/Assets/ML-Agents/Examples/3DBall/TFModels/`.
3. Open the Unity Editor, and select the `3DBall` scene as described above.
4. Select the `Ball3DBrain` object from the Scene hierarchy.
5. Change the `Type of Brain` to `Internal`.
6. Drag the `<env_name>.bytes` file from the Project window of the Editor to the `Graph Model` placeholder in the `3DBallBrain` inspector window.
7. Set the `Graph Placeholder` size to 1 (_Note that step 7 and 8 are done because 3DBall is a continuous control environment, and the TensorFlow model requires a noise parameter to decide actions. In cases with discrete control, epsilon is not needed_).
8. Add a placeholder called `epsilon` with a type of `floating point` and a range of values from `0` to `0`.
9. Press the Play button at the top of the editor.

If you followed these steps correctly, you should now see the trained model being used to control the behavior of the balance ball within the Editor itself. From here you can re-build the Unity binary, and run it standalone with your agent's new learned behavior built right in.

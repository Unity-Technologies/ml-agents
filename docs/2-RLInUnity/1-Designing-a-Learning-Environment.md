# Reinforcement Learning in Unity

Reinforcement learning is an artificial intelligence technique that trains agents to perform tasks by rewarding desirable behavior. During reinforcement learning, an agent explores its environment, observes the state of things, and, based on those observations, takes an action. If the action leads to a better state, the agent receives a positive reward. If it leads to a less desirable state, then the agent receives no reward or a negative reward (punishment). As the agent learns during training, it optimizes its decision making so that it receives the maximum reward over time.

ML Agents uses a reinforcement learning technique called [Proximal Policy Optimization (PPO)](link). PPO uses a neural network to approximate the ideal function that maps an agent's observations to the best action an agent can take in a given state. The ML Agents PPO algorithm is implemented in TensorFlow and runs in a separate Python process (communicating with the running Unity application over a socket). 

**Note:** if you aren't studying machine and reinforcement learning as a subject and just want to train agents to accomplish tasks, you can treat PPO training as a _black box_. There are a few training-related parameters to adjust inside Unity as well as on the Python training side, but you do not need in-depth knowledge of the algorithm itself to successfully create and train agents. Step-by-step procedures for running the training process are provided in the [Training section](link). 

## The Simulation and Training Process

Training and simulation proceed in steps orchestrated by the ML Agents [Academy](link) class. The Academy works with Agent and Brain objects in the scene to step through the simulation. When either the Academy has reached its maximum number of steps or all agents in the scene are _done_, one training episode is finished. 

During training, the external Python PPO process communicates with the Academy to run a series of episodes while it collects data and optimizes its neural network model. The type of Brain assigned to an agent determines whether it participates in training or not. The **External** brain commmunicates with the external process to train the TensorFlow model. When training is completed successfully, you can add the trained model file to your Unity project for use with an **Internal** brain.

The ML Agents [Academy](link) class orchestrates the agent simulation loop as follows:

1. Calls your Academy subclass's `AcademyReset()` function.
2. Calls your `AgentReset()` function for each agent in the scene.
3. Calls your  `CollectState()` function for each agent in the scene.
4. Uses each agent's Brain class to decide on the agent's next action. 
5. Calls your subclass's `AcademyStep()` function.
6. Calls your  `AgentStep()` function for each agent in the scene, passing in the action chosen by the agent's brain.
7. Optionally, the Academy calls the agent's `AgentOnDone()` function when the agent reaches its `Max Step` count or marks itself as `done`.
8. When the Academy reaches its own `Max Step` count, it starts the next episode again by calling your Academy subclass's `AcademyReset()` function.

To create a training environment, extend the Academy and Agent classes to implement the above methods. The [Agent.CollectState()](link) and [Agent.AgentStep()](link) functions are required; the other models are optional — whether you need to implement them depends on your specific scenario.
  
## Organizing the Unity Scene

To train and use ML Agents in a Unity scene, the scene must contain one Academy subclass and as many Brain objects and Agent subclasses as you need. Any Brain instances in the scene must beattached to GameObjects that are children of the Academy in the Unity Scene Hierarchy:

[Screenshot of scene hierarchy]

You must assign a brain to every agent, but you can share brains between multiple agents. Each agent will make its own observations and act independently, but will use the same decision-making logic. 

**Academy**
The [Academy](link) object orchestrates agents and their decision making processes. Only place one Academy object in a scene. You must create a subclass of the Academy class (since the base class is Abstract). 

When you create your Academy subclass, implement the following methods:

* `AcademyReset()` — Prepare the environment and agents for the next training episode. Use this function to place and initialize entities in the scene as necessary.
* `AcademyStep()` — Prepare the environment for the next simulation step. The base Academy class calls this function before calling any `AgentStep()` methods for the current step. You can use this function to update other objects in the scene before the agents take their actions. Note that the agents have already collected their observations and chosen an action before the Academy invokes this method.

  The base Academy classes also defines several important parameters that you can set in the Unity Editor Inspector:
  
  * `Max Steps` — Maximum number of steps per episode. Set to 0 for unlimited length. Once the Academy's step counter reaches this value, it calls the `AcademyReset()` function.
  * `Frames To Skip` — The Unity `FixedUpdate()` frames to skip between calling the agent  `CollectState()` and `AgentStep()` functions. The `AcademyStep()` function is still called very Unity frame.  
  * `Wait Time`— The time in seconds to wait between agent updates. The Academy ignores the wait time setting during training.
  * Training/Inference Configuration — Configures the [Unity Player settings](link) as approriate for training and release (inference).
     * `Width` & `Height` — the width and height of the Unity window. Use a small window during training to decrease rendering time. (This does not affect the rendering of Camera observations).
     * `Quality Level` — The Unity Player Quality level. Use a low value during training to decrease rendering time.
     * `Time Scale` — The Unity [TimeScale](https://docs.unity3d.com/ScriptReference/Time-timeScale.html). A scale of 1.0 is realtime. Set to a higher scale during training to decrease training time. 
     * `Target Frame Rate`— The Unity [TargetFrameRate](https://docs.unity3d.com/ScriptReference/Application-targetFrameRate.html) setting. Use a low value for `TargetFrameRate` and a high value for `TimeScale` during training to decrease rendering time and let the physics simulation run as fast as possible.
 * `Default Reset Parameter` — Allows you to define custom parameters that you can pass to the external process when using an External brain. See [topic](link).
 
 To set these parameters, select the GameObject containing your Academy subclass in the Unity Scene hierarchy (open the **Inspector** window if necessary with menu: **Window** > **Inspector**).
 
 [screenshot of Academy Inspector]

**Brain** 
The Brain encapsulates the decision making process. Every Agent must be assigned a Brain, but you can use the same Brain with more than one Agent. Brain objects must be children of the Academy in the Unity scene hierarchy. 


**Agent**
The Agent represents an actor in the scene that collects observations and carries out actions. The Agent class is typically attached to the GameObject in the scene that otherwise represents the actor. Every Agent must be assigned a Brain.  


## Environments




# Making a new Learning Environment

This tutorial walks through the process of creating a Unity Environment. A Unity Environment is an application built using the Unity Engine which can be used to train Reinforcement Learning agents.

## Setting up the Unity Project

1. Open an existing Unity project, or create a new one and import the RL interface package:
 * [ML-Agents package without TensorflowSharp](https://s3.amazonaws.com/unity-agents/0.2/ML-AgentsNoPlugin.unitypackage)
 * [ML-Agents package with TensorflowSharp](https://s3.amazonaws.com/unity-agents/0.2/ML-AgentsWithPlugin.unitypackage)

2. Rename `TemplateAcademy.cs` (and the contained class name) to the desired name of your new academy class. All Template files are in the folder `Assets -> Template -> Scripts`. Typical naming convention is `YourNameAcademy`.

3. Attach `YourNameAcademy.cs` to a new empty game object in the currently opened scene (`Unity` -> `GameObject` -> `Create Empty`) and rename this game object to `YourNameAcademy`. Since `YourNameAcademy` will be used to control all the environment logic, ensure the attached-to object is one which will remain in the scene regardless of the environment resetting, or other within-environment behavior.

4. Attach `Brain.cs` to a new empty game object and rename this game object to `YourNameBrain1`. Set this game object as a child of `YourNameAcademy` (Drag `YourNameBrain1` into `YourNameAcademy`). Note that you can have multiple brains in the Academy but they all must have different names.

5. Disable Window Resolution dialogue box and Splash Screen.
    1. Go to `Edit` -> `Project Settings` -> `Player` -> `Resolution and Presentation`.
    2. Set `Display Resolution Dialogue` to `Disabled`.
    3.Check `Run In Background`.
    4. Click `Splash Image`.
    5. Uncheck `Show Splash Screen` _(Unity Pro only)_.

6. If you will be using Tensorflow Sharp in Unity, you must:
	1. Make sure you are using Unity 2017.1 or newer.
	2. Make sure the TensorflowSharp [plugin](https://s3.amazonaws.com/unity-agents/0.2/TFSharpPlugin.unitypackage) is in your Asset folder.
	3. Go to `Edit` -> `Project Settings` -> `Player`
	4. For each of the platforms you target (**`PC, Mac and Linux Standalone`**, **`iOS`** or **`Android`**):
		1. Go into `Other Settings`.
		2. Select `Scripting Runtime Version` to `Experimental (.NET 4.6 Equivalent)`
		3. In `Scripting Defined Symbols`, add the flag `ENABLE_TENSORFLOW`
	5. Note that some of these changes will require a Unity Restart

# Implementing `YourNameAcademy`

1. Click on the game object **`YourNameAcademy`**.

2. In the inspector tab, you can modify the characteristics of the academy:
 * **`Max Steps`** Maximum length of each episode (set to 0 if you do not want the environment to reset after a certain time).
 * **`Wait Time`** Real-time between steps when running environment in test-mode.
 * **`Frames To Skip`** Number of frames (or physics updates) to skip between steps. The agents will act at every frame but get new actions only at every step.
 * **`Training Configuration`** and **`Inference Configuration`** The first defines the configuration of the Engine at training time and the second at test / inference time. The training mode corresponds only to external training when the reset parameter `train_model` was set to True. The adjustable parameters are as follows:
    * `Width` and `Height` Correspond to the width and height in pixels of the window (must be both greater than 0). Typically set it to a small size during training, and a larger size for visualization during inference.
    * `Quality Level` Determines how mesh rendering is performed. Typically set to small value during training and higher value for visualization during inference.
    * `Time Scale` Physics speed. If environment utilized physics calculations, increase this during training, and set to `1.0f` during inference. Otherwise, set it to `1.0f`.
    * `Target Frame Rate` Frequency of frame rendering. If environment utilizes observations, increase this during training, and set to `60` during inference. If no observations are used, this can be set to `1` during training.
 * **`Default Reset Parameters`** You can set the default configuration to be passed at reset. This will be a mapping from strings to float values that you can call in the academy with `resetParameters["YourDefaultParameter"]`

3. Within **`InitializeAcademy()`**, you can define the initialization of the Academy. Note that this command is ran only once at the beginning of the training session. Do **not** use `Awake()`, `Start()` or `OnEnable()`

3. Within **`AcademyStep()`**, you can define the environment logic each step. Use this function to modify the environment for the agents that will live in it.

4. Within **`AcademyReset()`**, you can reset the environment for a new episode. It should contain environment-specific code for setting up the environment. Note that `AcademyReset()` is called at the beginning of the training session to ensure the first episode is similar to the others.

## Implementing `YourNameBrain`
For each Brain game object in your academy :

1. Click on the game object `YourNameBrain`

2. In the inspector tab, you can modify the characteristics of the brain in  **`Brain Parameters`**
    * `State Size` Number of variables within the state provided to the agent(s).
    * `Action Size` The number of possible actions for each individual agent to take.
    * `Memory Size` The number of floats the agents will remember each step.
    * `Camera Resolutions` A list of flexible length that contains resolution parameters : `height` and `width` define the number dimensions of the camera outputs in pixels. Check `Black And White` if you want the camera outputs to be black and white.
    * `Action Descriptions` A list describing in human-readable language the meaning of each available action.
    * `State Space Type` and `Action Space Type`. Either `discrete` or `continuous`.
        * `discrete` corresponds to describing the action space with an `int`.
        * `continuous` corresponds to describing the action space with an array of `float`.

3. You can choose what kind of brain you want `YourNameBrain` to be. There are four possibilities:
 * `External` : You need at least one of your brains to be external if you wish to interact with your environment from python.
 * `Player` : To control your agents manually. If the action space is discrete, you must map input keys to their corresponding integer values. If the action space is continuous, you must map input keys to their corresponding indices and float values.
 * `Heuristic` : You can have your brain automatically react to the observations and states in a customizable way. You will need to drag a `Decision` script into `YourNameBrain`. To create a custom reaction, you must :
   *  Rename `TemplateDecision.cs` (and the contained class name) to the desired name of your new reaction. Typical naming convention is `YourNameDecision`.
   *  Implement `Decide`: Given the state, observation and memory of an agent, this function must return an array of floats corresponding to the actions taken by the agent. If the action space type is discrete, the array must be of size 1.
   *  Optionally, implement `MakeMemory`: Given the state, observation and memory of an agent, this function must return an array of floats corresponding to the new memories of the agent.
 * `Internal` : Note that you must have Tensorflow Sharp setup (see top of this page). Here are the fields that must be completed:
   *  `Graph Model` : This must be the `bytes` file corresponding to the pretrained Tensorflow graph. (You must first drag this file into your Resources folder and then from the Resources folder into the inspector)
   *  `Graph Scope` : If you set a scope while training your tensorflow model, all your placeholder name will have a prefix. You must specify that prefix here.
   *  `Batch Size Node Name` : If the batch size is one of the inputs of your graph, you must specify the name if the placeholder here. The brain will make the batch size equal to the number of agents connected to the brain automatically.
   *  `State Node Name` : If your graph uses the state as an input, you must specify the name if the placeholder here.
   *  `Recurrent Input Node Name` : If your graph uses a recurrent input / memory as input and outputs new recurrent input / memory, you must specify the name if the input placeholder here.
   *  `Recurrent Output Node Name` : If your graph uses a recurrent input / memory as input and outputs new recurrent input / memory, you must specify the name if the output placeholder here.
   * `Observation Placeholder Name` : If your graph uses observations as input, you must specify it here. Note that the number of observations is equal to the length of `Camera Resolutions` in the brain parameters.
   * `Action Node Name` : Specify the name of the placeholder corresponding to the actions of the brain in your graph. If the action space type is continuous, the output must be a one dimensional tensor of float of length `Action Space Size`, if the action space type is discrete, the output must be a one dimensional tensor of int of length 1.
   * `Graph Placeholder` : If your graph takes additional inputs that are fixed (example: noise level) you can specify them here. Note that in your graph, these must correspond to one dimensional tensors of int or float of size 1.
     * `Name` : Corresponds to the name of the placeholder.
     * `Value Type` : Either Integer or Floating Point.
     * `Min Value` and 'Max Value' : Specify the minimum and maximum values (included) the placeholder can take. The value will be sampled from the uniform distribution at each step. If you want this value to be fixed, set both `Min Value` and `Max Value` to the same number.

## Implementing `YourNameAgent`

1. Rename `TemplateAgent.cs` (and the contained class name) to the desired name of your new agent. Typical naming convention is `YourNameAgent`.

2. Attach `YourNameAgent.cs` to the game object that represents your agent. (Example: if you want to make a self-driving car, attach `YourNameAgent.cs` to a car looking game object)

3. In the inspector menu of your agent, drag the brain game object you want to use with this agent into the corresponding `Brain` box. Please note that you can have multiple agents with the same brain. If you want to give an agent a brain or change his brain via script, please use the method `ChangeBrain()`.

4. In the inspector menu of your agent, you can specify what cameras, your agent will use as its observations. To do so, drag the desired number of cameras into the `Observations` field. Note that if you want a camera to move along your agent, you can set this camera as a child of your agent

5. If `Reset On Done` is checked, `Reset()` will be called when the agent is done. Else, `AgentOnDone()` will be called. Note that if `Reset On Done` is unchecked, the agent will remain "done" until the Academy resets. This means that it will not take actions in the environment.

6. Implement the following functions in `YourNameAgent.cs` :
 * `InitializeAgent()` : Use this method to initialize your agent. This method is called when the agent is created. Do **not** use `Awake()`, `Start()` or `OnEnable()`.
 * `CollectState()` : Must return a list of floats corresponding to the state the agent is in. If the state space type is discrete, return a list of length 1 containing the float equivalent of your state.
 * `AgentStep()` : This function will be called every frame, you must define what your agent will do given the input actions. You must also specify the rewards and whether or not the agent is done. To do so, modify the public fields of the agent `reward` and `done`.
 * `AgentReset()` : This function is called at start, when the Academy resets and when the agent is done (if `Reset On Done` is checked).
 * `AgentOnDone()` : If `Reset On Done` is not checked, this function will be called when the agent is done. `Reset()` will only be called when the Academy resets.

If you create Agents via script, we recommend you save them as prefabs and instantiate them either during steps or resets. If you do, you can use `GiveBrain(brain)` to have the agent subscribe to a specific brain. You can also use `RemoveBrain()` to unsubscribe from a brain.

# Defining the reward function
The reward function is the set of circumstances and event which we want to reward or punish the agent for making happen. Here are some examples of positive and negative rewards:
* Positive
    * Reaching a goal
    * Staying alive
    * Defeating an enemy
    * Gaining health
    * Finishing a level
* Negative
    * Taking damage
    * Failing a level
    * The agent’s death

Small negative rewards are also typically used each step in scenarios where the optimal agent behavior is to complete an episode as quickly as possible.

Note that the reward is reset to 0 at every step, you must add to the reward (`reward += rewardIncrement`). If you use `skipFrame` in the Academy and set your rewards instead of incrementing them, you might lose information since the reward is sent at every step, not at every frame.

# Getting Started with the Balance Ball Example

![Balance Ball](../images/balance.png)

This tutorial will walk through the end-to-end process of installing Unity Agents, building an example environment, training an agent in it, and finally embedding the trained model into the Unity environment.

Unity ML Agents contains a number of example environments which can be used as templates for new environments, or as ways to test a new ML algorithm to ensure it is functioning correctly.

In this walkthrough we will be using the **3D Balance Ball** environment. The environment contains a number of platforms and balls. Platforms can act to keep the ball up by rotating either horizontally or vertically. Each platform is an agent which is rewarded the longer it can keep a ball balanced on it, and provided a negative reward for dropping the ball. The goal of the training process is to have the platforms learn to never drop the ball.

Let's get started!

## Installation

In order to install and set-up the Python and Unity environments, see the instructions [here](installation.md).

## Building Unity Environment
Launch the Unity Editor, and log in, if necessary.

1. Open the `unity-environment` folder using the Unity editor.  *(If this is not first time running Unity, you'll be able to skip most of these immediate steps, choose directly from the list of recently opened projects)*
    - On the initial dialog, choose `Open` on the top options
    - On the file dialog, choose `unity-environment` and click `Open` *(It is safe to ignore any warning message about non-matching editor installation)*
    - Once the project is open, on the `Project` panel (bottom of the tool), navigate to the folder `Assets/ML-Agents/Examples/3DBall/`
    - Double-click the `Scene` icon (Unity logo) to load all environment assets
2. Go to `Edit -> Project Settings -> Player`
    - Ensure that `Resolution and Presentation -> Run in Background` is Checked.
    - Ensure that `Resolution and Presentation -> Display Resolution Dialog` is set to Disabled.
3. Expand the `Ball3DAcademy` GameObject and locate its child object `Ball3DBrain` within the Scene hierarchy in the editor. Ensure Type of Brain for this object is set to `External`.
4. *File -> Build Settings*
5. Choose your target platform:
    - (optional) Select “Development Build” to log debug messages.
6. Click *Build*:
    - Save environment binary to the `python` sub-directory of the cloned repository *(you may need to click on the down arrow on the file chooser to be able to select that folder)*

## Training the Brain with Reinforcement Learning

### Testing Python API

To launch jupyter, run in the command line:

`jupyter notebook`

Then navigate to `localhost:8888` to access the notebooks. If you're new to jupyter, check out the [quick start guide](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/execute.html) before you continue.

To ensure that your environment and the Python API work as expected, you can use the `python/Basics` Jupyter notebook. This notebook contains a simple walkthrough of the functionality of the API. Within `Basics`, be sure to set `env_name` to the name of the environment file you built earlier.

### Training with PPO
In order to train an agent to correctly balance the ball, we will use a Reinforcement Learning algorithm called Proximal Policy Optimization (PPO). This is a method that has been shown to be safe, efficient, and more general purpose than many other RL algorithms, as such we have chosen it as the example algorithm for use with ML Agents. For more information on PPO, OpenAI has a recent [blog post](https://blog.openai.com/openai-baselines-ppo/) explaining it.

In order to train the agents within the Ball Balance environment:

1. Open `python/PPO.ipynb` notebook from Jupyter.
2. Set `env_name` to the name of your environment file earlier.
3. (optional) In order to get the best results quickly, set `max_steps` to 50000, set `buffer_size` to 5000, and set `batch_size` to 512.  For this exercise, this will train the model in approximately ~5-10 minutes.
4. (optional) Set `run_path` directory to your choice.
5. Run all cells of notebook with the exception of the last one under "Export the trained Tensorflow graph."

### Observing Training Progress
In order to observe the training process in more detail, you can use Tensorboard.
In your command line, enter into `python` directory and then run :

`tensorboard --logdir=summaries`

Then navigate to `localhost:6006`.

From Tensorboard, you will see the summary statistics of six variables:
* Cumulative Reward - The mean cumulative episode reward over all agents. Should increase during a successful training session.
* Value Loss - The mean loss of the value function update. Correlates to how well the model is able to predict the value of each state. This should decrease during a successful training session.
* Policy Loss - The mean loss of the policy function update. Correlates to how much the policy (process for deciding actions) is changing. The magnitude of this should decrease during a successful training session.
* Episode Length - The mean length of each episode in the environment for all agents.
* Value Estimates - The mean value estimate for all states visited by the agent. Should increase during a successful training session.
* Policy Entropy - How random the decisions of the model are. Should slowly decrease during a successful training process. If it decreases too quickly, the `beta` hyperparameter should be increased.

## Embedding Trained Brain into Unity Environment _[Experimental]_
Once the training process displays an average reward of ~75 or greater, and there has been a recently saved model (denoted by the `Saved Model` message) you can choose to stop the training process by stopping the cell execution. Once this is done, you now have a trained TensorFlow model. You must now convert the saved model to a Unity-ready format which can be embedded directly into the Unity project by following the steps below.

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

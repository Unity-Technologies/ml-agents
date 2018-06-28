# Python API

The ML-Agents toolkit provides a Python API for controlling the agent simulation loop of a environment or game built with Unity. This API is used by the ML-Agent training algorithms (run with `learn.py`), but you can also write your Python programs using this API. 

The key objects in the Python API include:

* **UnityEnvironment** — the main interface between the Unity application and your code. Use UnityEnvironment to start and control a simulation or training session.
* **BrainInfo** — contains all the data from agents in the simulation, such as observations and rewards.
* **BrainParameters** — describes the data elements in a BrainInfo object. For example, provides the array length of an observation in BrainInfo.

These classes are all defined in the `python/unityagents` folder of the ML-Agents SDK.

To communicate with an agent in a Unity environment from a Python program, the agent must either use an **External** brain or use a brain that is broadcasting (has its **Broadcast** property set to true). Your code is expected to return actions for agents with external brains, but can only observe broadcasting brains (the information you receive for an agent is the same in both cases). See [Using the Broadcast Feature](Learning-Environment-Design-Brains.md#using-the-broadcast-feature).

For a simple example of using the Python API to interact with a Unity environment, see the Basic [Jupyter](Background-Jupyter.md) notebook (`python/Basics.ipynb`), which opens an environment, runs a few simulation steps taking random actions, and closes the environment. 

_Notice: Currently communication between Unity and Python takes place over an open socket without authentication. As such, please make sure that the network where training takes place is secure. This will be addressed in a future release._

## Loading a Unity Environment

Python-side communication happens through `UnityEnvironment` which is located in `python/unityagents`. To load a Unity environment from a built binary file, put the file in the same directory as `unityagents`. For example, if the filename of your Unity environment is 3DBall.app, in python, run:

```python
from unityagents import UnityEnvironment
env = UnityEnvironment(file_name="3DBall", worker_id=0, seed=1)
```

* `file_name` is the name of the environment binary (located in the root directory of the python project).
* `worker_id` indicates which port to use for communication with the environment. For use in parallel training regimes such as A3C.
* `seed` indicates the seed to use when generating random numbers during the training process. In environments which do not involve physics calculations, setting the seed enables reproducible experimentation by ensuring that the environment and trainers utilize the same random seed.

If you want to directly interact with the Editor, you need to use `file_name=None`, then press the :arrow_forward: button in the Editor when the message _"Start training by pressing the Play button in the Unity Editor"_ is displayed on the screen

## Interacting with a Unity Environment

A BrainInfo object contains the following fields:

* **`visual_observations`** : A list of 4 dimensional numpy arrays. Matrix n of the list corresponds to the n<sup>th</sup> observation of the brain. 
* **`vector_observations`** : A two dimensional numpy array of dimension `(batch size, vector observation size)` if the vector observation space is continuous and `(batch size, 1)` if the vector observation space is discrete.
* **`text_observations`** : A list of string corresponding to the agents text observations.
* **`memories`** : A two dimensional numpy array of dimension `(batch size, memory size)` which corresponds to the memories sent at the previous step.
* **`rewards`** : A list as long as the number of agents using the brain containing the rewards they each obtained at the previous step. 
* **`local_done`** : A list as long as the number of agents using the brain containing  `done` flags (whether or not the agent is done). 
* **`max_reached`** : A list as long as the number of agents using the brain containing true if the agents reached their max steps.
* **`agents`** : A list of the unique ids of the agents using the brain.
* **`previous_actions`** : A two dimensional numpy array of dimension `(batch size, vector action size)` if the vector action space is continuous and `(batch size, 1)` if the vector action space is discrete.

Once loaded, you can use your UnityEnvironment object, which referenced by a variable named `env` in this example, can be used in the following way:  
- **Print : `print(str(env))`**  
Prints all parameters relevant to the loaded environment and the external brains.  
- **Reset : `env.reset(train_model=True, config=None)`**  
Send a reset signal to the environment, and provides a dictionary mapping brain names to BrainInfo objects.  
    - `train_model` indicates whether to run the environment in train (`True`) or test (`False`) mode.
    - `config` is an optional dictionary of configuration flags specific to the environment. For generic environments, `config` can be ignored. `config` is a dictionary of strings to floats where the keys are the names of the `resetParameters` and the values are their corresponding float values. Define the reset parameters on the [Academy Inspector](Learning-Environment-Design-Academy.md#academy-properties) window in the Unity Editor.
- **Step : `env.step(action, memory=None, text_action=None)`**  
Sends a step signal to the environment using the actions. For each brain : 
    - `action` can be one dimensional arrays or two dimensional arrays if you have multiple agents per brains.
    - `memory` is an optional input that can be used to send a list of floats per agents to be retrieved at the next step.
    - `text_action` is an optional input that be used to send a single string per agent.

    Returns a dictionary mapping brain names to BrainInfo objects.
    
    For example, to access the BrainInfo belonging to a brain called 'brain_name', and the BrainInfo field 'vector_observations':
    ```python
    info = env.step()
    brainInfo = info['brain_name']
    observations = brainInfo.vector_observations
    ``` 

    Note that if you have more than one external brain in the environment, you must provide dictionaries from brain names to arrays for     `action`, `memory` and `value`. For example: If you have two external brains named `brain1` and `brain2` each with one agent taking     two continuous actions, then you can have:
    ```python
    action = {'brain1':[1.0, 2.0], 'brain2':[3.0,4.0]}
    ```

Returns a dictionary mapping brain names to BrainInfo objects.  
- **Close : `env.close()`**  
Sends a shutdown signal to the environment and closes the communication socket.

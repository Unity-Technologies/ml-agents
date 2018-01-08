# Python API

_Notice: Currently communication between Unity and Python takes place over an open socket without authentication. As such, please make sure that the network where training takes place is secure. This will be addressed in a future release._

## Loading a Unity Environment

Python-side communication happens through `UnityEnvironment` which is located in `python/unityagents`. To load a Unity environment from a built binary file, put the file in the same directory as `unityagents`. In python, run:


```python
from unityagents import UnityEnvironment
env = UnityEnvironment(file_name=filename, worker_id=0)
```

* `file_name` is the name of the environment binary (located in the root directory of the python project). 
* `worker_id` indicates which port to use for communication with the environment. For use in parallel training regimes such as A3C.

## Interacting with a Unity Environment

A BrainInfo object contains the following fields:

* **`observations`** : A list of 4 dimensional numpy arrays. Matrix n of the list corresponds to the n<sup>th</sup> observation of the brain. 
* **`states`** : A two dimensional numpy array of dimension `(batch size, state size)` if the state space is continuous and `(batch size, 1)` if the state space is discrete.
* **`memories`** : A two dimensional numpy array of dimension `(batch size, memory size)` which corresponds to the memories sent at the previous step.
* **`rewards`** : A list as long as the number of agents using the brain containing the rewards they each obtained at the previous step. 
* **`local_done`** : A list as long as the number of agents using the brain containing  `done` flags (wether or not the agent is done). 
* **`agents`** : A list of the unique ids of the agents using the brain.

Once loaded, `env` can be used in the following way:  
- **Print : `print(str(env))`**  
Prints all parameters relevant to the loaded environment and the external brains.  
- **Reset : `env.reset(train_model=True, config=None)`**  
Send a reset signal to the environment, and provides a dictionary mapping brain names to BrainInfo objects.  
    - `train_model` indicates whether to run the environment in train (`True`) or test (`False`) mode.
    - `config` is an optional dictionary of configuration flags specific to the environment. For more information on adding optional config flags to an environment, see [here](Making-a-new-Unity-Environment.md#implementing-yournameacademy). For generic environments, `config` can be ignored. `config` is a dictionary of strings to floats where the keys are the names of the `resetParameters` and the values are their corresponding float values.  
- **Step : `env.step(action, memory=None, value = None)`**  
Sends a step signal to the environment using the actions. For each brain : 
    - `action` can be one dimensional arrays or two dimensional arrays if you have multiple agents per brains.
    - `memory` is an optional input that can be used to send a list of floats per agents to be retrieved at the next step.
    - `value` is an optional input that be used to send a single float per agent to be displayed if and `AgentMonitor.cs` component is attached to the agent. 

Note that if you have more than one external brain in the environment, you must provide dictionaries from brain names to arrays for `action`, `memory` and `value`. For example: If you have two external brains named `brain1` and `brain2` each with one agent taking two continuous actions, then you can have:
```python
action = {'brain1':[1.0, 2.0], 'brain2':[3.0,4.0]}
```

Returns a dictionary mapping brain names to BrainInfo objects.  
- **Close : `env.close()`**  
Sends a shutdown signal to the environment and closes the communication socket.


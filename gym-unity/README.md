# Unity ML-Agents Gym Wrapper

A common way in which machine learning researchers interact with simulation
environments is via a wrapper provided by OpenAI called `gym`. For more
information on the gym interface, see [here](https://github.com/openai/gym).

We provide a a gym wrapper, and instructions for using it with existing machine
learning algorithms which utilize gyms. Both wrappers provide interfaces on top
of our `UnityEnvironment` class, which is the default way of interfacing with a
Unity environment via Python.

## Installation

The gym wrapper can be installed using:

```sh
pip install gym_unity
```

or by running the following from the `/gym-unity` directory of the repository:

```sh
pip install .
```

## Using the Gym Wrapper

The gym interface is available from `gym_unity.envs`. To launch an environmnent
from the root of the project repository use:

```python
from gym_unity.envs import UnityEnv

env = UnityEnv(environment_filename, worker_id, default_visual, multiagent)
```

* `environment_filename` refers to the path to the Unity environment.
* `worker_id` refers to the port to use for communication with the environment.
  Defaults to `0`.
* `use_visual` refers to whether to use visual observations (True) or vector
  observations (False) as the default observation provided by the `reset` and
  `step` functions. Defaults to `False`.
* `multiagent` refers to whether you intent to launch an environment which
  contains more than one agent. Defaults to `False`.

The returned environment `env` will function as a gym.

For more on using the gym interface, see our
[Jupyter Notebook tutorial](../notebooks/getting-started-gym.ipynb).

## Limitation

* It is only possible to use an environment with a single Brain.
* By default the first visual observation is provided as the `observation`, if
  present. Otherwise vector observations are provided.
* All `BrainInfo` output from the environment can still be accessed from the
  `info` provided by `env.step(action)`.
* Stacked vector observations are not supported.
* Environment registration for use with `gym.make()` is currently not supported.

## Running OpenAI Baselines Algorithms

OpenAI provides a set of open-source maintained and tested Reinforcement
Learning algorithms called the [Baselines](https://github.com/openai/baselines).

Using the provided Gym wrapper, it is possible to train ML-Agents environments
using these algorithms. This requires the creation of custom training scripts to
launch each algorithm. In most cases these scripts can be created by making
slightly modifications to the ones provided for Atari and Mujoco environments.

### Example - DQN Baseline

In order to train an agent to play the `GridWorld` environment using the
Baselines DQN algorithm, you first need to install the baselines package using 
pip:

```
pip install git+git://github.com/openai/baselines
```

Next, create a file called `train_unity.py`. Then create an `/envs/` directory 
and build the GridWorld environment to that directory. For more information on 
building Unity environments, see 
[here](../docs/Learning-Environment-Executable.md). Add the following code to 
the `train_unity.py` file:

```python
import gym

from baselines import deepq
from gym_unity.envs import UnityEnv

def main():
    env = UnityEnv("./envs/GridWorld", 0, use_visual=True)
    act = deepq.learn(
        env,
        "mlp",
        lr=1e-3,
        total_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10
    )
    print("Saving model to unity_model.pkl")
    act.save("unity_model.pkl")


if __name__ == '__main__':
    main()
```

To start the training process, run the following from the root of the baselines
repository:

```sh
python -m train_unity
```

### Other Algorithms

Other algorithms in the Baselines repository can be run using scripts similar to
the examples from the baselines package. In most cases, the primary changes needed 
to use a Unity environment are to import `UnityEnv`, and to replace the environment
creation code, typically `gym.make()`, with a call to `UnityEnv(env_path)`
passing the environment binary path.

A typical rule of thumb is that for vision-based environments, modification
should be done to Atari training scripts, and for vector observation
environments, modification should be done to Mujoco scripts.

Some algorithms will make use of `make_env()` or `make_mujoco_env()`
functions. You can define a similar function for Unity environments.  An example of 
such a method using the PPO2 baseline:

```python
from gym_unity.envs import UnityEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.bench import Monitor
from baselines import logger
import baselines.ppo2.ppo2 as ppo2

import os

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

def make_unity_env(env_directory, num_env, visual, start_index=0):
    """
    Create a wrapped, monitored Unity environment.
    """
    def make_env(rank, use_visual=True): # pylint: disable=C0111
        def _thunk():
            env = UnityEnv(env_directory, rank, use_visual=use_visual)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            return env
        return _thunk
    if visual:
        return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])
    else:
        rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
        return make_env(rank, use_visual=False)

def main():
    env = make_unity_env('./envs/GridWorld', 4, True)
    ppo2.learn(
        network="mlp",
        env=env,
        total_timesteps=100000,
        lr=1e-3,
    )

if __name__ == '__main__':
    main()
```

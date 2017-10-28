![alt text](../images/banner.png "Unity ML - Agents")

# Unity ML - Agents (Python API)

## Python Setup

### Requirements
* Jupyter
* docopt
* Matplotlib
* numpy
* Pillow
* Python (2 or 3)
* Tensorflow (1.0+)

### Installing Dependencies
To install dependencies, run:

`pip install .`

or 

`pip3 install .`

If your Python environment doesn't include `pip`, see these [instructions](https://packaging.python.org/guides/installing-using-linux-tools/#installing-pip-setuptools-wheel-with-linux-package-managers) on installing it.

## Provided Jupyter Notebooks

* **Basic** - Demonstrates usage of `UnityEnvironment` class for launching and interfacing with Unity Environments.
* **PPO** - Used for training agents. Contains an implementation of Proximal Policy Optimization Reinforcement Learning algorithm. 

### Running each notebook

To launch jupyter, run:

`jupyter notebook` 

Then navigate to `localhost:8888` to access each training notebook.

To monitor training progress, run the following from the root directory of this repo:

`tensorboard --logdir=summaries`

Then navigate to `localhost:6006` to monitor progress with Tensorboard.

## Training PPO directly

To train using PPO without the notebook, run: `python3 ppo.py <env_name> --train`

Where `<env_name>` corresponds to the name of the built Unity environment.

For a list of additional hyperparameters, run: `python3 ppo.py --help`

## Using Python API
See this [documentation](../docs/Unity-Agents---Python-API.md) for a detailed description of the functions and uses of the Python API.

## Training on AWS
See this related [blog post](https://medium.com/towards-data-science/how-to-run-unity-on-amazon-cloud-or-without-monitor-3c10ce022639) for a description of how to run Unity Environments on AWS EC2 instances with the GPU.

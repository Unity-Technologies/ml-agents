# Installation & Set-up

## Install **Unity 2017.1** or later (required)

[Download](https://store.unity.com/download) and install Unity.

## Clone the ml-agents repository

Once installed, you will want to clone the Agents GitHub repository. 

    git clone git@github.com:Unity-Technologies/ml-agents.git

The `unity-environment` directory in this repository contains the Unity Assets to add to your projects. The `python` directory contains the training code. Both directories are located at the root of the repository. 

## Install Python

In order to train an agent within the ML Agents framework, you need Python 2 or 3 along with the dependencies described below.

### Windows Users

If you are a Windows user who is new to Python/TensorFlow, follow [this guide](https://unity3d.college/2017/10/25/machine-learning-in-unity3d-setting-up-the-environment-tensorflow-for-agentml-on-windows-10/) to set up your Python environment.

### Requirements
* Jupyter
* Matplotlib
* numpy
* Pillow
* Python (2 or 3; 64bit required)
* docopt (Training)
* TensorFlow (1.0+) (Training)

### Install Dependencies

To install dependencies, go into the `python` sub-directory of the repository, and run (depending on your python version) from the command line:

    pip install .

or 

    pip3 install .

If your Python environment doesn't include `pip`, see these [instructions](https://packaging.python.org/guides/installing-using-linux-tools/#installing-pip-setuptools-wheel-with-linux-package-managers) on installing it.

Once the requirements are successfully installed, the next step is to check out the [Getting Started guide](Getting-Started-with-Balance-Ball.md).

## Installation Help

### Using Jupyter Notebook

For a walkthrough of how to use Jupyter notebook, see [Running the Jupyter Notebook](http://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/execute.html) in the _Jupyter/IPython Quick Start Guide_.

#### Testing Python API

To launch jupyter, run in the command line:

`jupyter notebook`

Then navigate to `localhost:8888` to access the notebooks. If you're new to jupyter, check out the [quick start guide](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/execute.html) before you continue.

To ensure that your environment and the Python API work as expected, you can use the `python/Basics` Jupyter notebook. This notebook contains a simple walkthrough of the functionality of the API. Within `Basics`, be sure to set `env_name` to the name of the environment file you built earlier.

### General Issues

If you run into issues while installing or running Unity ML Agents, see [Limitations & Common Issues](Limitations-and-Common-Issues.md) for a list of common issues and solutions.

If you have an issue that isn't covered here, feel free to contact us at ml-agents@unity3d.com. Alternatively, feel free to create an issue on the repository.
Be sure to include relevant information on OS, Python version, and exact error message if possible.

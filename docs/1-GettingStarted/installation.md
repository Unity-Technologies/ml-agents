# Installation & Set-up

## Install **Unity 2017.1** or later (required)

Download link available [here](https://store.unity.com/download?ref=update).

## Clone the repository
Once installed, you will want to clone the Agents GitHub repository. References will be made 
throughout to `unity-environment` and `python` directories. Both are located at the root of the repository. 

## Installing Python API
In order to train an agent within the framework, you will need to install Python 2 or 3, and the dependencies described below.

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

### Installing Dependencies
To install dependencies, go into the `python` sub-directory of the repository, and run (depending on your python version) from the command line:

`pip install .`

or 

`pip3 install  .`

If your Python environment doesn't include `pip`, see these [instructions](https://packaging.python.org/guides/installing-using-linux-tools/#installing-pip-setuptools-wheel-with-linux-package-managers) on installing it.

Once the requirements are successfully installed, the next step is to check out the [Getting Started guide](Getting-Started-with-Balance-Ball.md).

## Installation Help

### Using Jupyter Notebook

For a walkthrough of how to use Jupyter notebook, see [here](http://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/execute.html).

### General Issues

If you run into issues while attempting to install and run Unity ML Agents, see [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Limitations-&-Common-Issues.md) for a list of common issues and solutions.

If you have an issue that isn't covered here, feel free to contact us at ml-agents@unity3d.com. Alternatively, feel free to create an issue on the repository.
Be sure to include relevant information on OS, Python version, and exact error message if possible.

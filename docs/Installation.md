# Installation & Set-up

To install and use ML Agents, you need install Unity, clone this repository
and install Python with additional dependencies. Each of the subsections
below overviews each step, in addition to an experimental Docker set-up.

## Install **Unity 2017.1** or Later

[Download](https://store.unity.com/download) and install Unity.

## Clone the ml-agents Repository

Once installed, you will want to clone the ML Agents GitHub repository. 

    git clone git@github.com:Unity-Technologies/ml-agents.git

The `unity-environment` directory in this repository contains the Unity Assets
to add to your projects. The `python` directory contains the training code.
Both directories are located at the root of the repository. 

## Install Python

In order to use ML Agents, you need Python (2 or 3; 64 bit required) along with
the dependencies listed in the [requirements file](../python/requirements.txt).
Some of the primary dependencies include:
- [TensorFlow](https://www.tensorflow.org/)
- [Jupyter](http://jupyter.org/)
- [NumPy](http://www.numpy.org/)

### Windows Users

If you are a Windows user who is new to Python and TensorFlow, follow 
[this guide](https://unity3d.college/2017/10/25/machine-learning-in-unity3d-setting-up-the-environment-tensorflow-for-agentml-on-windows-10/)
to set up your Python environment.

### Mac and Unix Users

If your Python environment doesn't include `pip`, see these 
[instructions](https://packaging.python.org/guides/installing-using-linux-tools/#installing-pip-setuptools-wheel-with-linux-package-managers)
on installing it.

To install dependencies, go into the `python` subdirectory of the repository,
and run (depending on your Python version) from the command line:

    pip install .

or 

    pip3 install .

## Docker-based Installation _[Experimental]_

If you'd like to use Docker for ML Agents, please follow [this guide](Docker.md). 

## Help

If you run into any problems installing ML Agents, 
[submit an issue](https://github.com/Unity-Technologies/ml-agents/issues) and
make sure to cite relevant information on OS, Python version, and
exact error message (whenever possible). 

### Using TensorFlow (and TensorBoard)

TensorFlow is used for training the machine learning models in ML Agents. 
Unless you are implementing new algorithms, the use of TensorFlow
is mostly abstracted away and behind the scenes. However, as you will come
across later, one component of training models with TensorFlow is setting the
values of certain model attributes (called _hyperparameters_). Finding the
right values of these hyperparameters can require a few iterations.
Consequently, we leverage a visualization tool within TensorFlow called
[TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard). 
It allows the visualization of certain agent attributes (e.g. reward)
throughout training which can be helpful in both building
intuitions for the different hyperparameters and setting the optimal values for 
your Unity environment. We provide more details on setting the hyperparameters
in later parts of the documentation, but, in the meantime, if you are 
unfamiliar with TensorBoard we recommend this 
[tutorial](https://github.com/dandelionmane/tf-dev-summit-tensorboard-tutorial).

### Using Jupyter

Jupyter is a fantastic tool for writing code with embedded visualizations. 
We provide several such notebooks for testing your Python
installation and training behaviors. For a walkthrough of how to use 
Jupyter, see
[Running the Jupyter Notebook](http://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/execute.html)
in the _Jupyter/IPython Quick Start Guide_.

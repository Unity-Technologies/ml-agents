# Installation

To install and use ML-Agents, you need install Unity, clone this repository
and install Python with additional dependencies. Each of the subsections
below overviews each step, in addition to a Docker set-up.

## Install **Unity 2017.1** or Later

[Download](https://store.unity.com/download) and install Unity. If you would
like to use our Docker set-up (introduced later), make sure to select the 
_Linux Build Support_ component when installing Unity.

<p align="center">
    <img src="images/unity_linux_build_support.png" 
        alt="Linux Build Support" 
        width="500" border="10" />
</p>

## Clone the Ml-Agents Repository

Once installed, you will want to clone the ML-Agents Toolkit GitHub repository. 

    git clone https://github.com/Unity-Technologies/ml-agents.git

The `unity-environment` directory in this repository contains the Unity Assets
to add to your projects. The `python` directory contains the training code.
Both directories are located at the root of the repository. 

## Install Python (with Dependencies)

In order to use ML-Agents toolkit, you need Python 3.5 or 3.6 along with
the dependencies listed in the [requirements file](../python/requirements.txt).
Some of the primary dependencies include:
- [TensorFlow](Background-TensorFlow.md) 
- [Jupyter](Background-Jupyter.md) 

### Windows Users

If you are a Windows user who is new to Python and TensorFlow, follow [this guide](Installation-Windows.md) to set up your Python environment.

### Mac and Unix Users

[Download](https://www.python.org/downloads/) and install Python 3 if you do not already have it.

If your Python environment doesn't include `pip`, see these 
[instructions](https://packaging.python.org/guides/installing-using-linux-tools/#installing-pip-setuptools-wheel-with-linux-package-managers)
on installing it.

To install dependencies, **go into the `python` subdirectory** of the repository,
and run from the command line:

    pip3 install .

## Docker-based Installation

If you'd like to use Docker for ML-Agents, please follow 
[this guide](Using-Docker.md). 

## Next Steps

The [Basic Guide](Basic-Guide.md) page contains several short 
tutorials on setting up the ML-Agents toolkit within Unity, running a pre-trained model, in
addition to building and training environments.

## Help

If you run into any problems regarding ML-Agents, refer to our [FAQ](FAQ.md) and our [Limitations](Limitations.md) pages. If you can't find anything please
[submit an issue](https://github.com/Unity-Technologies/ml-agents/issues) and
make sure to cite relevant information on OS, Python version, and exact error 
message (whenever possible). 

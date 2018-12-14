# Installation

To install and use ML-Agents, you need to install Unity, clone this repository and
install Python with additional dependencies. Each of the subsections below
overviews each step, in addition to a Docker set-up.

## Install **Unity 2017.4** or Later

[Download](https://store.unity.com/download) and install Unity. If you would
like to use our Docker set-up (introduced later), make sure to select the _Linux
Build Support_ component when installing Unity.

<p align="center">
  <img src="images/unity_linux_build_support.png"
       alt="Linux Build Support"
       width="500" border="10" />
</p>

## Windows Users
For setting up your environment on Windows, we have created a [detailed
guide](Installation-Windows.md) to setting up your env. For Mac and Linux,
continue with this guide.

## Mac and Unix Users

### Clone the ML-Agents Toolkit Repository

Once installed, you will want to clone the ML-Agents Toolkit GitHub repository.

```sh
git clone https://github.com/Unity-Technologies/ml-agents.git
```

The `UnitySDK` subdirectory contains the Unity Assets to add to your projects.
It also contains many [example environments](Learning-Environment-Examples.md)
to help you get started.

The `ml-agents` subdirectory contains Python packages which provide
trainers and a Python API to interface with Unity.

The `gym-unity` subdirectory contains a package to interface with OpenAI Gym.

### Install Python and mlagents Package

In order to use ML-Agents toolkit, you need Python 3.6 along with the
dependencies listed in the [setup.py file](../ml-agents/setup.py).
Some of the primary dependencies include:

- [TensorFlow](Background-TensorFlow.md)
- [Jupyter](Background-Jupyter.md)

[Download](https://www.python.org/downloads/) and install Python 3.6 if you do not
already have it.

If your Python environment doesn't include `pip3`, see these
[instructions](https://packaging.python.org/guides/installing-using-linux-tools/#installing-pip-setuptools-wheel-with-linux-package-managers)
on installing it.

To install the dependencies and `mlagents` Python package, enter the
`ml-agents/` subdirectory and run from the command line:

```sh
pip3 install -e .
```

If you installed this correctly, you should be able to run
`mlagents-learn --help`

**Notes:**

- We do not currently support Python 3.7 or Python 3.5.
- If you are using Anaconda and are having trouble with TensorFlow, please see
  the following
  [note](https://www.tensorflow.org/install/install_mac#installing_with_anaconda)
  on how to install TensorFlow in an Anaconda environment.

## Docker-based Installation

If you'd like to use Docker for ML-Agents, please follow
[this guide](Using-Docker.md).

## Next Steps

The [Basic Guide](Basic-Guide.md) page contains several short tutorials on
setting up the ML-Agents toolkit within Unity, running a pre-trained model, in
addition to building and training environments.

## Help

If you run into any problems regarding ML-Agents, refer to our [FAQ](FAQ.md) and
our [Limitations](Limitations.md) pages. If you can't find anything please
[submit an issue](https://github.com/Unity-Technologies/ml-agents/issues) and
make sure to cite relevant information on OS, Python version, and exact error
message (whenever possible).

# Installation

The ML-Agents Toolkit contains several components:
* Unity package (`com.unity.ml-agents`) contains the Unity C# SDK that will be integrated into your Unity scene.
* Three Python packages:
  ** [`ml-agents`](../ml-agents/) contains the machine learning algorithms that enables you to train behaviors in your Unity scene. Most users of ML-Agents will only need to directly install `ml-agents`.
  ** `ml-agents-envs` contains a Python API to interact with a Unity scene. It is a foundational layer that facilitates
data messaging between Unity scene and the Python machine learning algorithms. Consequently, `ml-agents` depends on `ml-agents-envs`.
  ** `gym-unity` provides a Python-wrapper for your Unity scene that supports the OpenAI Gym interface.
* Unity project that contains several [example environments](Learning-Environment-Examples.md) that highlight the various features of the toolkit.

Consequently, to install and use ML-Agents you will need to:
* Install Unity (2018.4 or later)
* Install Python (3.6.1 or higher)
* Clone this repository
* Install the `com.unity.ml-agents` Unity package
* Install the `ml-agents` Python package

Each step above has a dedicated section below.

## Install **Unity 2018.4** or Later

[Download](https://unity3d.com/get-unity/download) and install Unity. We strongly recommend
that you install Unity through the Unity Hub as it will enable you to manage multiple Unity
versions. 

## Install **Python 3.6.1** or Higher

In order to use ML-Agents toolkit, you need Python 3.6.1 or higher. We recommend [installing]((https://www.python.org/downloads/)) a 3.6.x or 3.7.x version of Python.

If your Python environment doesn't include `pip3`, see these
[instructions](https://packaging.python.org/guides/installing-using-linux-tools/#installing-pip-setuptools-wheel-with-linux-package-managers)
on installing it.

Although we do not support the Anaconda installation path of ML-Agents for Windows, the previous 
[Windows Anaconda Installation (Deprecated) guide](Installation-Windows.md)
is still available. 

## Clone the ML-Agents Toolkit Repository

Now that you have installed Unity and Python, you will need to clone the 
ML-Agents Toolkit GitHub repository to install the Unity package.

```sh
git clone --branch latest_release https://github.com/Unity-Technologies/ml-agents.git
```
The `--branch latest_release` option will switch to the tag of the latest stable release.
Omitting that will get the `master` branch which is potentially unstable.

The repository contains a few key high-level sub-directories worth highlighting:

* `Project/` includes the Unity project with contains many [example environments](Learning-Environment-Examples.md)
to help you get started.

* `com.unity.ml-agents/` contains the Unity package

* `ml-agents/` contains the `ml-agents` Python package

* `ml-agents-envs/` contains the `ml-agents-envs` Python package

* `gym-unity/` contains the `gym-unity` Python package

## Install the `com.unity.ml-agents` Unity package
The Unity ML-Agents C# SDK is now a Unity Package. We are working on getting it added to the
official Unity package registery which will enable you to install the `com.unity.ml-agents` package
without cloning this repository. Until then, you can add the `com.unity.ml-agents` package to 
your project by navigating to the menu `Window`  -> `Package Manager`.  In the package manager 
window click on the `+` button.

<p align="center">
  <img src="images/unity_package_manager_window.png"
       alt="Linux Build Support"
       width="500" border="10" />
</p>

**NOTE:** In Unity 2018.4 it's on the bottom right of the packages list, and in Unity 2019.3 it's 
on the top left of the packages list.

Select `Add package from disk...` and navigate into the
`com.unity.ml-agents` folder and select the `package.json` folder.

<p align="center">
  <img src="images/unity_package_json.png"
       alt="Linux Build Support"
       width="500" border="10" />
</p>

If you are going to follow the examples from our documentation, you can open the `Project`
folder in Unity and start tinkering immediately.

### Install the `ml-agents` Python package

Installing the `ml-agents` Python package involves installing other Python package
that `ml-agents` depends on. So you may run into installation issues if your machine 
has older versions of any of those dependencies already installed. Consequently, our
supported path for installing `ml-agents` is to leverage Python Virtual Environments.
Virtual Environments provide a mechanim for isolating the dependencies for each project 
and are supported on Mac / Windows / Linux. We offer a dedicated 
[guide on Virtual Environments](Using-Virtual-Environment.md).

To install the `mlagents` Python package, activate your virtual environment and run from the command line:

```sh
pip3 install mlagents
```

Note that this will install `ml-agents` from PyPi, _not_ from the cloned repo.
If you installed this correctly, you should be able to run
`mlagents-learn --help`, after which you will see the Unity logo and the command line
parameters you can use with `mlagents-learn`.

By installing the `mlagents` package, the dependencies listed in the [setup.py file](../ml-agents/setup.py) are also installed.
Some of the primary dependencies include:

- [TensorFlow](Background-TensorFlow.md) (Requires a CPU w/ AVX support)
- [Jupyter](Background-Jupyter.md)

**Notes:** If you are using Anaconda and are having trouble with TensorFlow, please see
the following [link](https://www.tensorflow.org/install/pip) on how to install TensorFlow 
in an Anaconda environment.

### Installing for Development

If you intend to make modifications to `ml-agents` or `ml-agents-envs`, you should install
the packages from the cloned repo rather than from PyPi. To do this, you will need to install
 `ml-agents` and `ml-agents-envs` separately. From the repo's root directory, run:

```sh
cd ml-agents-envs
pip3 install -e ./
cd ..
cd ml-agents
pip3 install -e ./
```

Running pip with the `-e` flag will let you make changes to the Python files directly and have those
reflected when you run `mlagents-learn`. It is important to install these packages in this order as the
`mlagents` package depends on `mlagents_envs`, and installing it in the other
order will download `mlagents_envs` from PyPi.

## Docker

If you would
like to use our Docker set-up (introduced later), make sure to select the _Linux
Build Support_ component when installing Unity.

<p align="center">
  <img src="images/unity_linux_build_support.png"
       alt="Linux Build Support"
       width="500" border="10" />
</p>


## Next Steps

The [Basic Guide](Basic-Guide.md) page contains several short tutorials on
setting up the ML-Agents Toolkit within Unity, running a pre-trained model, in
addition to building and training environments.

## Help

If you run into any problems regarding ML-Agents, refer to our [FAQ](FAQ.md) and
our [Limitations](Limitations.md) pages. If you can't find anything please
[submit an issue](https://github.com/Unity-Technologies/ml-agents/issues) and
make sure to cite relevant information on OS, Python version, and exact error
message (whenever possible).

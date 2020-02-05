# Installation

To install and use ML-Agents, you need to install Unity, clone this repository and
install Python with additional dependencies. Each of the subsections below
overviews each step, in addition to a Docker set-up.

## Install **Unity 2018.4** or Later

[Download](https://store.unity.com/download) and install Unity. If you would
like to use our Docker set-up (introduced later), make sure to select the _Linux
Build Support_ component when installing Unity.

<p align="center">
  <img src="images/unity_linux_build_support.png"
       alt="Linux Build Support"
       width="500" border="10" />
</p>

## Environment Setup
We now support a single mechanism for installing ML-Agents on Mac/Windows/Linux using Virtual
Environments. For more information on Virtual Environments and installation instructions,
follow this [guide](Using-Virtual-Environment.md).

Although we don't support Anaconda installation path of ML-Agents for Windows, the previous guide
is still in the docs folder.  Please refer to [Windows Installation (Deprecated)](Installation-Windows.md).

### Clone the ML-Agents Toolkit Repository

Once installed, you will want to clone the ML-Agents Toolkit GitHub repository.

```sh
git clone --branch latest_release https://github.com/Unity-Technologies/ml-agents.git
```
The `--branch latest_release` option will switch to the tag of the latest stable release.
Omitting that will get the `master` branch which is potentially unstable.

The `com.unity.ml-agents` subdirectory contains the core code to add to your projects.
The `Project` subdirectory contains many [example environments](Learning-Environment-Examples.md)
to help you get started.

### Package Installation
ML-Agents C# SDK is transitioning to a Unity Package.  While we are working on getting into the
official packages list, you can add the `com.unity.ml-agents` package to your project by
navigating to the menu `Window`  -> `Package Manager`.  In the package manager window click
on the `+` button.

<p align="center">
  <img src="images/unity_package_manager_window.png"
       alt="Linux Build Support"
       width="500" border="10" />
</p>

**NOTE:** In Unity 2018.4 it's on the bottom right of the packages list, and in Unity 2019.3 it's on the top left of the packages list.

Select `Add package from disk...` and navigate into the
`com.unity.ml-agents` folder and select the `package.json` folder.

<p align="center">
  <img src="images/unity_package_json.png"
       alt="Linux Build Support"
       width="500" border="10" />
</p>

If you are going to follow the examples from our documentation, you can open the `Project`
folder in Unity and start tinkering immediately.


The `ml-agents` subdirectory contains a Python package which provides deep reinforcement
learning trainers to use with Unity environments.

The `ml-agents-envs` subdirectory contains a Python API to interface with Unity, which
the `ml-agents` package depends on.

The `gym-unity` subdirectory contains a package to interface with OpenAI Gym.

### Install Python and mlagents Package

In order to use ML-Agents toolkit, you need Python 3.6.1 or higher.
[Download](https://www.python.org/downloads/) and install the latest version of Python if you do not already have it.

If your Python environment doesn't include `pip3`, see these
[instructions](https://packaging.python.org/guides/installing-using-linux-tools/#installing-pip-setuptools-wheel-with-linux-package-managers)
on installing it.

To install the `mlagents` Python package, run from the command line:

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

**Notes:**

- We do not currently support Python 3.5 or lower.
- If you are using Anaconda and are having trouble with TensorFlow, please see
  the following
  [link](https://www.tensorflow.org/install/pip)
  on how to install TensorFlow in an Anaconda environment.

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

# Installation

The ML-Agents Toolkit contains several components:
* Unity package ([`com.unity.ml-agents`](../com.unity.ml-agents/)) contains the Unity C#
SDK that will be integrated into your Unity scene.
* Three Python packages:
  * [`mlagents`](../ml-agents/) contains the machine learning algorithms that enables you
  to train behaviors in your Unity scene. Most users of ML-Agents will only need to
  directly install `mlagents`.
  * [`mlagents_envs`](../ml-agents-envs/) contains a Python API to interact with a Unity
  scene. It is a foundational layer that facilitates data messaging between Unity scene
  and the Python machine learning algorithms. Consequently, `mlagents` depends on `mlagents_envs`.
  * [`gym_unity`](../gym-unity/) provides a Python-wrapper for your Unity scene that
  supports the OpenAI Gym interface.
* Unity [Project](../Project/) that contains several
[example environments](Learning-Environment-Examples.md) that highlight the various features
of the toolkit to help you get started.

Consequently, to install and use ML-Agents you will need to:
* Install Unity (2018.4 or later)
* Install Python (3.6.1 or higher)
* Clone this repository
* Install the `com.unity.ml-agents` Unity package
* Install the `mlagents` Python package

### Install **Unity 2018.4** or Later

[Download](https://unity3d.com/get-unity/download) and install Unity. We strongly recommend
that you install Unity through the Unity Hub as it will enable you to manage multiple Unity
versions.

### Install **Python 3.6.1** or Higher

We recommend [installing](https://www.python.org/downloads/) Python 3.6 or 3.7. If your Python
environment doesn't include `pip3`, see these
[instructions](https://packaging.python.org/guides/installing-using-linux-tools/#installing-pip-setuptools-wheel-with-linux-package-managers)
on installing it.

Although we do not provide support for Anaconda installation on Windows, the previous
[Windows Anaconda Installation (Deprecated) guide](Installation-Anaconda-Windows.md)
is still available.

### Clone the ML-Agents Toolkit Repository

Now that you have installed Unity and Python, you will need to clone the
ML-Agents Toolkit GitHub repository to install the Unity package (the Python
packages can be installed directly from PyPi - a Python package registry).

```sh
git clone --branch latest_release https://github.com/Unity-Technologies/ml-agents.git
```
The `--branch latest_release` option will switch to the tag of the latest stable release.
Omitting that will get the `master` branch which is potentially unstable.

### Install the `com.unity.ml-agents` Unity package

The Unity ML-Agents C# SDK is a Unity Package. We are working on getting it added to the
official Unity package registry which will enable you to install the `com.unity.ml-agents` package
[directly from the registry](https://docs.unity3d.com/Manual/upm-ui-install.html) without cloning
this repository. Until then, you will need to
[install it from the local package](https://docs.unity3d.com/Manual/upm-ui-local.html) that you
just cloned. You can add the `com.unity.ml-agents` package to
your project by navigating to the menu `Window`  -> `Package Manager`. In the package manager
window click on the `+` button. Select `Add package from disk...` and navigate into the
`com.unity.ml-agents` folder and select the `package.json` folder.

**NOTE:** In Unity 2018.4 it's on the bottom right of the packages list, and in Unity 2019.3 it's
on the top left of the packages list.

<p align="center">
  <img src="images/unity_package_manager_window.png"
       alt="Unity Package Manager Window"
       height="340" border="10" />
  <img src="images/unity_package_json.png"
     alt="package.json"
     height="340" border="10" />
</p>

If you are going to follow the examples from our documentation, you can open the `Project`
folder in Unity and start tinkering immediately.

### Install the `mlagents` Python package

Installing the `mlagents` Python package involves installing other Python packages
that `mlagents` depends on. So you may run into installation issues if your machine
has older versions of any of those dependencies already installed. Consequently, our
supported path for installing `mlagents` is to leverage Python Virtual Environments.
Virtual Environments provide a mechanim for isolating the dependencies for each project
and are supported on Mac / Windows / Linux. We offer a dedicated
[guide on Virtual Environments](Using-Virtual-Environment.md).

To install the `mlagents` Python package, activate your virtual environment and run from the
command line:

```sh
pip3 install mlagents
```

Note that this will install `mlagents` from PyPi, _not_ from the cloned repo.
If you installed this correctly, you should be able to run
`mlagents-learn --help`, after which you will see the Unity logo and the command line
parameters you can use with `mlagents-learn`.

By installing the `mlagents` package, the dependencies listed in the
[setup.py file](../ml-agents/setup.py) are also installed. These include
[TensorFlow](Background-TensorFlow.md) (Requires a CPU w/ AVX support) and
[Jupyter](Background-Jupyter.md).

#### Advanced: Installing for Development

If you intend to make modifications to `mlagents` or `mlagents_envs`, you should install
the packages from the cloned repo rather than from PyPi. To do this, you will need to install
 `mlagents` and `mlagents_envs` separately. From the repo's root directory, run:

```sh
pip3 install -e ./ml-agents-envs
pip3 install -e ./ml-agents
```

Running pip with the `-e` flag will let you make changes to the Python files directly and have
those reflected when you run `mlagents-learn`. It is important to install these packages in this
order as the `mlagents` package depends on `mlagents_envs`, and installing it in the other
order will download `mlagents_envs` from PyPi.

## Next Steps

The [Getting Started](Getting-Started.md) guide contains several short tutorials on
setting up the ML-Agents Toolkit within Unity, running a pre-trained model, in
addition to building and training environments.

## Help

If you run into any problems regarding ML-Agents, refer to our [FAQ](FAQ.md) and
our [Limitations](Limitations.md) pages. If you can't find anything please
[submit an issue](https://github.com/Unity-Technologies/ml-agents/issues) and
make sure to cite relevant information on OS, Python version, and exact error
message (whenever possible).

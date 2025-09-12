# Installation
To install and use the ML-Agents Toolkit, follow the steps below. Detailed instructions for each step are provided later on this page.

1. Install Unity (6000.0 or later)
2. Install Python (>= 3.10.1, <=3.10.12) - we recommend using 3.10.12
3. Install the `com.unity.ml-agents` Unity package; or clone this repository and install locally (recommended for the latest version and bug fixes)
4. Install `mlagents-envs`
5. Install `mlagents`

### Install **Unity 6000.0** or Later

[Download](https://unity3d.com/get-unity/download) and install Unity. We strongly recommend that you install Unity through the Unity Hub as it will enable you to manage multiple Unity versions.

### Install **Python 3.10.12**

We recommend [installing](https://www.python.org/downloads/) Python 3.10.12. If you are using Windows, please install the x86-64 version and not x86. If your Python environment doesn't include `pip3`, see these [instructions](https://packaging.python.org/guides/installing-using-linux-tools/#installing-pip-setuptools-wheel-with-linux-package-managers) on installing it. We also recommend using [conda](https://docs.conda.io/en/latest/) or [mamba](https://github.com/mamba-org/mamba) to manage your python virtual environments.

#### Conda python setup

Once conda has been installed in your system, open a terminal and execute the following commands to setup a python 3.10.12 virtual environment and activate it.

```shell
conda create -n mlagents python=3.10.12 && conda activate mlagents
```

### Install the `com.unity.ml-agents` Unity package

The Unity ML-Agents C# SDK is a Unity Package. You can install the `com.unity.ml-agents` package [directly from the Package Manager registry](https://docs.unity3d.com/Manual/upm-ui-install.html). Please make sure you enable 'Preview Packages' in the 'Advanced' dropdown in order to find the latest Preview release of the package.

**NOTE:** If you do not see the ML-Agents package listed in the Package Manager please follow the advanced installation instructions below.

#### Advanced: Local Installation for Development

You will need to clone the repository if you plan to modify or extend the ML-Agents Toolkit for your purposes, or if you'd like to download our example environments. Some of our tutorials / guides assume you have access to our example environments.

Use the command below to clone the repository

```sh
git clone --branch release_23 https://github.com/Unity-Technologies/ml-agents.git
```

The `--branch release_23` option will switch to the tag of the latest stable release. Omitting that will get the `develop` branch which is potentially unstable. However, if you find that a release branch does not work, the recommendation is to use the `develop` branch as it may have potential fixes for bugs and dependency issues.

(Optional to get bleeding edge)

```sh
git clone https://github.com/Unity-Technologies/ml-agents.git
```

If you plan to contribute those changes back, make sure to clone the `develop` branch (by omitting `--branch release_23` from the command above). See our [Contributions Guidelines](CONTRIBUTING.md) for more information on contributing to the ML-Agents Toolkit.

You can [add the local](https://docs.unity3d.com/Manual/upm-ui-local.html) `com.unity.ml-agents` package (from the repository that you just cloned) to your project by:

1. navigating to the menu `Window` -> `Package Manager`.
2. In the package manager window click on the `+` button on the top left of the packages list).
3. Select `Add package from disk...`
4. Navigate into the `com.unity.ml-agents` folder.
5. Select the `package.json` file.

<p align="center"> <img src="images/unity_package_manager_window.png" alt="Unity Package Manager Window" height="150" border="10" /> <img src="images/unity_package_json.png" alt="package.json" height="150" border="10" /> </p>

If you are going to follow the examples from our documentation, you can open the
`Project` folder in Unity and start tinkering immediately.

### Install Python package

Installing the `mlagents` Python package involves installing other Python packages that `mlagents` depends on. So you may run into installation issues if your machine has older versions of any of those dependencies already installed. Consequently, our supported path for installing `mlagents` is to leverage Python Virtual Environments. Virtual Environments provide a mechanism for isolating the dependencies for each project and are supported on Mac / Windows / Linux. We offer a dedicated [guide on Virtual Environments](Using-Virtual-Environment.md).

#### Installing `mlagents` from PyPi

You can install the ML-Agents Python package directly from PyPi. This is the recommended approach if you installed the C# package via the Package Manager registry.

**Important:** Ensure you install a Python package version that matches your Unity package version. Check the [release history](https://github.com/Unity-Technologies/ml-agents/releases) to find compatible versions.

To install, activate your virtual environment and run the following command:

```shell
python -m pip install mlagents==1.1.0
```

which will install the latest version of ML-Agents Python packages and associated dependencies available on PyPi. If building the wheel for `grpcio` fails, run the following command before installing `mlagents` with pip:

```shell
conda install "grpcio=1.48.2" -c conda-forge
```

When you install the Python package, the dependencies listed in the [setup.py file](https://github.com/Unity-Technologies/ml-agents/blob/release/4.0.0/ml-agents/setup.py) are also installed. These include [PyTorch](Background-PyTorch.md).


#### Advanced: Local Installation for Development

##### (Windows) Installing PyTorch

On Windows, you'll have to install the PyTorch package separately prior to installing ML-Agents in order to make sure the cuda-enabled version is used, rather than the CPU-only version. Activate your virtual environment and run from the command line:

```sh
pip3 install torch~=2.2.1 --index-url https://download.pytorch.org/whl/cu121
```

Note that on Windows, you may also need Microsoft's Visual C++ Redistributable if you don't have it already. See the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for more installation options and versions.

##### All Platforms

To install the `mlagents` Python package, activate your virtual environment and run from the command line:

```sh
cd /path/to/ml-agents
python -m pip install ./ml-agents-envs
python -m pip install ./ml-agents
```

Note that this will install `mlagents` from the cloned repository, _not_ from the PyPi repository. If you installed this correctly, you should be able to run `mlagents-learn --help`, after which you will see the command line parameters you can use with `mlagents-learn`.



If you intend to make modifications to `mlagents` or `mlagents_envs`, from the repository's root directory, run:

```sh
pip3 install torch -f https://download.pytorch.org/whl/torch_stable.html
pip3 install -e ./ml-agents-envs
pip3 install -e ./ml-agents
```

Running pip with the `-e` flag will let you make changes to the Python files directly and have those reflected when you run `mlagents-learn`. It is important to install these packages in this order as the `mlagents` package depends on `mlagents_envs`, and installing it in the other order will download `mlagents_envs` from PyPi.

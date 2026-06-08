# Install the ML-Agents Toolkit
Set up your system to use the ML-Agents Toolkit to train and run machine-learning agents in Unity projects.

This process includes installing Unity, configuring Python, and installing the ML-Agents packages. Follow the steps in order to ensure compatibility between Unity and the ML-Agents components.



##  Install Unity

Install Unity 6000.0 or later to use the ML-Agents Toolkit.

To install Unity, follow these steps:

1. [Download Unity](https://unity3d.com/get-unity/download).
2. Use **Unity Hub** to manage installations and versions.
   Unity Hub makes it easier to manage multiple Unity versions and associated projects.
3. Verify that the Unity Editor version is 6000.0 or later.

## Install Python 3.10.12 using Conda

Use Conda or Mamba to install and manage your Python environment. This ensures that ML-Agents dependencies are isolated and version-controlled.

To install Python, follow these steps:

1. Install [Conda](https://docs.conda.io/en/latest/) or [Mamba](https://github.com/mamba-org/mamba).
2. Open a terminal and create a new Conda environment with Python 3.10.12:

```shell
conda create -n mlagents python=3.10.12 && conda activate mlagents
```

3. On **Windows**, install PyTorch separately to ensure CUDA support:

```shell
pip3 install torch~=2.2.1 --index-url https://download.pytorch.org/whl/cu121
```
If prompted, install Microsoft Visual C++ Redistributable. For more installation options and versions, refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/).


## Install ML-Agents
You can install ML-Agents in two ways:

* [Package installation](#install-ml-agents-package-installation): Recommended for most users who want to use ML-Agents without modifying the source code or using the example environments.
* [Advanced installation](#install-ml-agents-advanced-installation): For contributors, developers extending ML-Agents, or users who want access to the example environments.

### Install ML-Agents (Package installation)

Use this method if you don’t plan to modify the toolkit or need the example environments.

#### Install the ML-Agents Unity package

To install the package, follow these steps:

1. In Unity, open **Window** > **Package Manager**.
2. Select **+** > **Add package by name**.
3. Enter `com.unity.ml-agents`.
4. Enable **Preview Packages** under the **Advanced** drop-down list if the package doesn’t appear.

If the package isn’t listed, follow the [Advanced Installation](#install-ml-agents-advanced-installation) method instead.



#### Install the ML-Agents Python package

Install the ML-Agents Python package to enable communication between Unity and your machine learning training environment.

Using a Python virtual environment helps isolate project dependencies and prevent version conflicts across your system. Virtual environments are supported on macOS, Windows, and Linux. For more information, refer to [Using Virtual Environments](Using-Virtual-Environment.md).

1. Before installing ML-Agents, activate the Conda environment you created.


2. Install the ML-Agents Python package from the Python Package Index (PyPI):

```shell
python -m pip install mlagents==1.1.0
```
Make sure to install a Python package version that matches your Unity ML-Agents package version. For information on compatible versions, refer to the [ML-Agents release history](https://github.com/Unity-Technologies/ml-agents/releases).

3. If you encounter an error while building the `grpcio` wheel, install it separately before reinstalling `mlagents`:

```shell
conda install "grpcio=1.48.2" -c conda-forge
```
This step resolves dependency conflicts that can occur with older versions of `grpcio`.

4. When the installation completes successfully, all the required Python dependencies listed in the [setup.py file](https://github.com/Unity-Technologies/ml-agents/blob/release/4.0.0/ml-agents/setup.py), including [PyTorch](Background-PyTorch.md) are automatically configured.


### Install ML-Agents (Advanced Installation)

Use the advanced installation method if you plan to modify or extend the ML-Agents Toolkit, or if you want to download and use the example environments included in the repository.

#### Clone the ML-Agents repository

Clone the ML-Agents repository to access the source code, sample environments, and development branches.

To clone the latest stable release, run:

```sh
git clone --branch release_23 https://github.com/Unity-Technologies/ml-agents.git
```

The `--branch release_23` flag checks out the latest stable release.
If you omit this option, the `develop` branch is cloned instead, which may contain experimental or unstable changes.
If the release branch does not work as expected, switch to the develop branch. It may include fixes for dependency or compatibility issues.
To clone the bleeding-edge development version (optional), run:
```sh
git clone https://github.com/Unity-Technologies/ml-agents.git
```
If you plan to contribute your changes, clone the develop branch (omit the `--branch` flag) and refer to the [Contribution Guidelines](CONTRIBUTING.md) for details.


#### Add the ML-Agents Unity package

After cloning the repository, add the `com.unity.ml-agents` Unity package to your project.

To add the local package, follow these steps:

1. In the Unity Editor, go to **Window** > **Package Manager**.
2. In the **Package Manager** window, select **+**.
3. Select **Add package from disk**.
4. Navigate to the cloned repository and open the `com.unity.ml-agents` folder.
5. Select the `package.json` file.

Unity adds the ML-Agents package to your project.

> [!NOTE]
> If you plan to use the example environments provided in the repository, open the **Project** folder in Unity to explore and experiment with them. Refer to [Configure example environments](Examples-setup.md) for full set up instructions.


<p align="center"> <img src="images/unity_package_manager_window.png" alt="Unity Package Manager Window" height="150" border="10" /> <img src="images/unity_package_json.png" alt="package.json" height="150" border="10" /> </p>


#### Install the ML-Agents Python package

Install the Python packages from the cloned repository to enable training and environment communication.

1. From the root of the cloned repository, activate your virtual environment and run:

```sh
cd /path/to/ml-agents
python -m pip install ./ml-agents-envs
python -m pip install ./ml-agents
```

This installs the ML-Agents packages directly from the cloned source, _not_ from PyPi.

2. To confirm a successful installation, run:
`mlagents-learn --help`

If the command lists available parameters, your setup is complete.

3. If you plan to modify the ML-Agents source code or contribute changes, install the packages in editable mode.
Editable installs let you make live changes to the Python files and test them immediately.

From the repository’s root directory, run:

```sh
pip3 install torch -f https://download.pytorch.org/whl/torch_stable.html
pip3 install -e ./ml-agents-envs
pip3 install -e ./ml-agents
```

> [!NOTE]
> Install the packages in this order because the `mlagents` package depends on `mlagents_envs`.
> Installing them in the order will download `mlagents_envs` from PyPi, which can cause version mismatches.

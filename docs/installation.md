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

#### If You're Using Conda
- Create a new Conda Environment
    - `conda create --name unity_ml_env`
    - `source activate unity_ml_env` or if you're on Windows `activate unity_ml_env`
- Install requirements.
    - `conda install Matplotlib`
    - `conda install numpy`
    - `conda install Pillow`
    - `conda install docopt`
    - [Go here for the latest TensorFlow install info](https://www.tensorflow.org/install/)
    - The other requirements are part of Conda already

- Allow Jupyter Notebook to see your Conda Environment
    - `conda install ipykernel`
    - `python -m ipykernel install --user --name myenv --display-name "Python (unity_ml_env)"`

- Launch the *PPO.ipynb* notebook from the `python` sub-directory of the repository
    - `jupyter notebook PPO.ipynb`
    - Select the Conda Env you just worked so hard to make. *Kernel->Change kernel->Python (unity_ml_env)*
    - If you cannot see your env in the list check the documentation [here](http://ipython.readthedocs.io/en/stable/install/kernel_install.html)

- You're good to go!

#### If You're NOT Using Conda
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

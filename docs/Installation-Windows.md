# Installing ML-Agents for Windows
We have provided a step by step guide to get ML-Agents working on Windows.  

In order to get ML-Agents working with Windows, you will need to have Windows 10 installed.  While it is possible for ML-Agents to work on other versions of Windows, we have only tested with a local installation of Windows 10 (not using VM like Bootcamp or Parallels).

Additionally, you will need to check if your GPU is CUDA compatible.  Please check Nvidia's page [here](https://developer.nvidia.com/cuda-gpus).

## Step 1: Install Nvidia CUDA toolkit
<a href="https://developer.nvidia.com/cuda-toolkit-archive" target="_blank">Download</a> and install the CUDA toolkit from Nvidia's archive.  The toolkit includes GPU-accelerated libraries, debugging and optimization tools, a C/C++ compiler and a runtime library and is needed to run ML-Agents.  You can select the latest or previous releases.  In this guide, we are using version 9.1.85.3 ([direct link](https://developer.nvidia.com/compute/cuda/9.1/Prod/patches/3/cuda_9.1.85.3_windows)).  

_Before installing, please make sure you __close any running instances of Unity or Visual Studio.___

Run the installer and select the Express option.  Note the directory where you installed the CUDA toolkit.  In this guide, we installed in the directory `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1`

## Step 2: Install Nvidia cuDNN library
<a href="https://developer.nvidia.com/cudnn" target="_blank">Download</a> and install the cuDNN library from Nvidia.  cuDNN is is a GPU-accelerated library of primitives for deep neural networks.  Before you can download, you will need to sign up for free to the Nvidia Developer Program.

<p align="center">
    <img src="images/cuDNN_membership_required.png" 
        alt="cuDNN membership required" 
        width="500" border="10" />
</p>

Once you've signed up, go back to the cuDNN <a href="https://developer.nvidia.com/cudnn" target="_blank">downloads page</a>.  You may or may not be asked to fill out a short survey.  When you get to the list cuDNN releases, __make sure you are downloading the right version for the CUDA toolkit you installed in Step 1.__  In this guide, we are using version 7.1.1 for CUDA toolkit version 9.1+ ([direct link](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.1.1/prod/9.1_20180214/cudnn-9.1-windows10-x64-v7.1)).

After you have downloaded the cuDNN files, you will need to extract the files into the CUDA toolkit directory.  In the cuDNN zip file, copy 3 folders called `bin`, `include`, and `lib` into the CUDA toolkit directory.  In this guide, the CUDA toolkit directory is located at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1`

## Step 4: Set Environment Variables
You will need to add one environment variable and two path variables.

To set the environment variable, type `environment variables` in the search bar (this can be reached by hitting the Windows key or the bottom left Windows button).  You should see an option called __Edit the system environment variables__.  From here, click the __Environment Variables__ button.  Click __New__ to add a new system variable.  For __Variable Name__, enter `CUDA_HOME`.  For the variable value, put the directory location for the CUDA toolkit.  In this guide, the directory location is `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1`.  Press __OK__ once.

To set the two path variables, inside the same __Enviornment Variables__ window and under the second box called __System Variables__, find a variable called `PATH` and click __Edit__.  You will add  two directories to the list.  For this guide, the two entries would look like:

    C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\lib\x64
    C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\extras\CUPTI\libx64

Make sure to replace the relevant directory location with the one you have installed.  _Please note that case sensitivity matters_.

## Step 4: Install Python via Anaconda
<a href="https://www.anaconda.com/download/#windows" target="_blank">Download</a> and install Anaconda for Windows.  By using Anaconda, you can use manage separate environments for different distributions of Python.  We **strongly** recommend using Python 3 as we do not guarantee supporting Python 2 in future releases.  In this guide, we are using Python version 3.6 and Anaconda version 5.1 ([64-bit](https://repo.continuum.io/archive/Anaconda3-5.1.0-Windows-x86_64.exe) or [32-bit](https://repo.continuum.io/archive/Anaconda3-5.1.0-Windows-x86.exe) direct links).  

## Step 5: Setup a New Conda Environment
You will create a new Conda environment from our installation of Anaconda.  This new environment will be specifically for ML-Agents.  This means that all packages that you install are localized to just this environment.  It will not affect any other installation of Python.  Whenever you want to run ML-Agents, you can do some from this new Conda environment.

Open a new CMD or Powershell prompt and type in the following command:

    conda create -n ml-agents python=3.6

This will create a new Conda environment called ml-agents using Python version 3.6.  To use this environment, you will need to activate it.  _(In the future, if you need to use this environment again, you can run the same command)_.  In the same CMD or Powershell prompt, type in the following command:

    activate ml-agents

After this, you will need to install `tensorflow-gpu.`  This can be installed by using `pip` - which is a package management system used to install Python packages.  In the same CMD or Powershell prompt, type in the following command:

    pip install tensorflow-gpu

Lastly, you should test to see if everything installed properly.  You can do this to see if TensorFlow can identify your GPU. In the same CMD or Powershell prompt, type in the following command: 

    python
    
    import tensorflow as tf
    
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

You should see something similar to:

```
Found device 0 with properties ...
```

## Step 6: Install Required Python Packages
ML-Agents includes a list of Python dependencies needed to run.  Similarly in Step 4, you will be using `pip` to install these Python packages.

If you haven't already, make sure to clone the repository.  You can do this using Git ([download here](https://git-scm.com/download/win)) and running the following commands in a new or same CMD or Powershell window from Step 4:

    git clone git@github.com:Unity-Technologies/ml-agents.git

If you don't want to use Git, you can always directly download all the files [here](https://github.com/Unity-Technologies/ml-agents/archive/master.zip).

In our example, the files are located in `C:\Downloads`.  After you have either cloned or downloaded the files, in the same CMD or Powershell window from Step 4, change to the python directory inside the ML-agents directory (_if you closed this window, you can activate the right Conda environment by typing `activate ml-agents`_):

    cd C:\Downloads\ml-agents\python

Make sure you are connected to the internet and then type in the following two commands:

    conda install –force html5lib 
    
    pip install .

This will complete the installation of all the required Python packages to run ML-Agents.  

## Step 6: Acknowledgements

We would like to thank [Jason Weimann](https://unity3d.college/2017/10/25/machine-learning-in-unity3d-setting-up-the-environment-tensorflow-for-agentml-on-windows-10/) and [Nitish S. Mutha](http://blog.nitishmutha.com/tensorflow/2017/01/22/TensorFlow-with-gpu-for-windows.html) for writing the original articles which were used to create this guide.


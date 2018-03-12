# Installing ML-Agents for Windows
To get ML-Agents working with Windows, you will need to have Windows 10 installed.  While it is possible for ML-Agents to work on other versions of Windows, we have only tested with a local installation of Windows 10 (not using VM like Bootcamp or Parallels)

## Step 1: Install Nvidia CUDA toolkit
<a href="https://developer.nvidia.com/cuda-toolkit-archive" target="_blank">Download</a> and install the CUDA toolkit from Nvidia's archive.  The toolkit includes GPU-accelerated libraries, debugging and optimization tools, a C/C++ compiler and a runtime library and is needed to run ML-Agents.  You can select the latest or previous releases.  In this guide, we are using version 9.1.85.3 ([direct link](https://developer.nvidia.com/compute/cuda/9.1/Prod/patches/3/cuda_9.1.85.3_windows)).  

_Before installing, please make sure you __close any running instances of Unity or Visual Studio.___

## Step 2: Install Nvidia cuDNN library
<a href="https://developer.nvidia.com/cudnn" target="_blank">Download</a> and install the cuDNN library from Nvidia.  cuDNN is is a GPU-accelerated library of primitives for deep neural networks.  Before you can download, you will need to sign up for free to the Nvidia Developer Program.

<p align="center">
    <img src="images/cuDNN_membership_required.png" 
        alt="cuDNN membership required" 
        width="500" border="10" />
</p>

Once you've signed up, go back to the cuDNN <a href="https://developer.nvidia.com/cudnn" target="_blank">downloads page</a>.  You may or may not be asked to fill out a short survey.  When you get to the list cuDNN releases, __make sure you are downloading the right version for the CUDA toolkit you installed in Step 1.__  In this guide, we are using version 7.1.1 for CUDA toolkit version 9.1+ ([direct link](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.1.1/prod/9.1_20180214/cudnn-9.1-windows10-x64-v7.1)).  

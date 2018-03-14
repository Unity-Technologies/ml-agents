# Training on Amazon Web Service

This page contains instructions for setting up an EC2 instance on Amazon Web Service for training ML-Agents environments. You can run "headless" training if none of the agents in the environment use visual observations. 

## Pre-Configured AMI
A public pre-configured AMI is available with the ID: `ami-30ec184a` in the `us-east-1` region. It was created as a modification of the Amazon Deep Learning [AMI](https://aws.amazon.com/marketplace/pp/B01M0AXXQB). 

## Configuring your own Instance

1. To begin with, you will need an EC2 instance which contains the latest Nvidia drivers, CUDA8, and cuDNN.  There are a number of external tutorials which describe this, such as:
    * [Getting CUDA 8 to Work With openAI Gym on AWS and Compiling TensorFlow for CUDA 8 Compatibility](https://davidsanwald.github.io/2016/11/13/building-tensorflow-with-gpu-support.html)
    * [Installing TensorFlow on an AWS EC2 P2 GPU Instance](http://expressionflow.com/2016/10/09/installing-tensorflow-on-an-aws-ec2-p2-gpu-instance/)
    * [Updating Nvidia CUDA to 8.0.x in Ubuntu 16.04 – EC2 Gx instance](https://aichamp.wordpress.com/2016/11/09/updating-nvidia-cuda-to-8-0-x-in-ubuntu-16-04-ec2-gx-instance/)

## Installing ML-Agents

2. Move `python` sub-folder of this ml-agents repo to the remote ECS instance, and set it as the working directory.
2. Install the required packages with `pip3 install .`.

## Testing

To verify that all steps worked correctly:

1. In the Unity Editor, load a project containing an ML-Agents environment (you can use one of the example environments if you have not created your own).
2. Open the Build Settings window (menu: File > Build Settings).
3. Select Linux as the Target Platform, and x64_86 as the target architecture.
4. Check Headless Mode (unless you have enabled a virtual screen following the instructions below).
5. Click Build to build the Unity environment executable.
6. Upload the executable to your EC2 instance.
7. Test the instance setup from Python using:

```python
from unityagents import UnityEnvironment

env = UnityEnvironment(<your_env>)
```
Where `<your_env>` corresponds to the path to your environment executable.
 
You should receive a message confirming that the environment was loaded successfully.

## (Optional) Enabling a virtual screen

_Instructions here are adapted from this [Medium post](https://medium.com/towards-data-science/how-to-run-unity-on-amazon-cloud-or-without-monitor-3c10ce022639) on running general Unity applications in the cloud._

Current limitations of the Unity Engine require that a screen be available to render to when using visual observations. In order to make this possible when training on a remote server, a virtual screen is required. We can do this by installing Xorg and creating a virtual screen. Once installed and created, we can display the Unity environment in the virtual environment, and train as we would on a local machine. Ensure that `headless` mode is disabled when building linux executables which use visual observations.

1. Run the following commands to install Xorg:

    ```
    sudo apt-get update
    sudo apt-get install -y xserver-xorg mesa-utils
    sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024
    ```

2. Restart the EC2 instance.

3. Make sure there are no Xorg processes running. To kill the Xorg processes, run `sudo killall Xorg`.  
Note that you might have to run this command multiple times depending on how Xorg is configured.  
If you run `nvidia-smi`, you will have a list of processes running on the GPU, Xorg should not be in the list. 

4. Run:

    ```
    sudo /usr/bin/X :0 &
    export DISPLAY=:0
    ```
 
5. To ensure the installation was successful, run `glxgears`. If there are no errors, then Xorg is correctly configured.
# Training on Amazon Web Service

This page contains instructions for setting up an EC2 instance on Amazon Web Service for use in training ML-Agents environments. Current limitations of the Unity Engine require that a screen be available to render to. In order to make this possible when training on a remote server, a virtual screen is required. We can do this by installing Xorg and creating a virtual screen. Once installed and created, we can display the Unity environment in the virtual environment, and train as we would on a local machine. 

## Pre-Configured AMI
A public pre-configured AMI is available with the ID: `ami-30ec184a` in the `us-east-1` region. It was created as a modification of the Amazon Deep Learning [AMI](https://aws.amazon.com/marketplace/pp/B01M0AXXQB). 

## Configuring your own Instance
Instructions here are adapted from this [Medium post](https://medium.com/towards-data-science/how-to-run-unity-on-amazon-cloud-or-without-monitor-3c10ce022639) on running general Unity applications in the cloud.

1. To begin with, you will need an EC2 instance which contains the latest Nvidia drivers, CUDA8, and cuDNN.  There are a number of external tutorials which describe this, such as:
    * [Getting CUDA 8 to Work With openAI Gym on AWS and Compiling Tensorflow for CUDA 8 Compatibility](https://davidsanwald.github.io/2016/11/13/building-tensorflow-with-gpu-support.html)
    * [Installing TensorFlow on an AWS EC2 P2 GPU Instance](http://expressionflow.com/2016/10/09/installing-tensorflow-on-an-aws-ec2-p2-gpu-instance/)
    * [Updating Nvidia CUDA to 8.0.x in Ubuntu 16.04 – EC2 Gx instance](https://aichamp.wordpress.com/2016/11/09/updating-nvidia-cuda-to-8-0-x-in-ubuntu-16-04-ec2-gx-instance/)
2. Move `python` to remote instance.
2. Install the required packages with `pip install .`.
3. Run the following commands to install Xorg:
    ```
    sudo apt-get update
    sudo apt-get install -y xserver-xorg mesa-utils
    sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024
    ```
4. Restart the EC2 instance.

## Launching your instance

1. Make sure there are no Xorg processes running. To kill the Xorg processes, run `sudo killall Xorg`.  
Note that you might have to run this command multiple times depending on how Xorg is configured.  
If you run `nvidia-smi`, you will have a list of processes running on the GPU, Xorg should not be in the list. 

2. Run:
    ```
    sudo /usr/bin/X :0 &
    export DISPLAY=:0
    ```
3. To ensure the installation was succesful, run `glxgears`. If there are no errors, then Xorg is correctly configured.
4. There is a bug in _Unity 2017.1_ which requires the uninstallation of `libxrandr2`, which can be removed with :
```
sudo apt-get remove --purge libwxgtk3.0-0v5
sudo apt-get remove --purge libxrandr2
```
This is scheduled to be fixed in 2017.3.

## Testing

If all steps worked correctly, upload an example binary built for Linux to the instance, and test it from python with:
```python
from unityagents import UnityEnvironment

env = UnityEnvironment(your_env)
```

You should receive a message confirming that the environment was loaded succesfully.

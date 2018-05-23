# Training on Amazon Web Service

This page contains instructions for setting up an EC2 instance on Amazon Web Service for training ML-Agents environments.

## Recommended starting AMI
You can get started with an EC2 instance with the Deep Learning AMI (Ubuntu) listed under AWS Marketplace [AMI](https://aws.amazon.com/marketplace/pp/B077GCH38C). Choose the python3 environment within that ami which gives you the python3 and CUDA 9.0 environment.

## Configuring your own Instance

1. To begin with, you will need an EC2 instance which contains the latest Nvidia drivers, CUDA9, and cuDNN.  There are a number of external tutorials which describe this although they are targeted for CUDA8, such as:
    * [Getting CUDA 8 to Work With openAI Gym on AWS and Compiling TensorFlow for CUDA 8 Compatibility](https://davidsanwald.github.io/2016/11/13/building-tensorflow-with-gpu-support.html)
    * [Installing TensorFlow on an AWS EC2 P2 GPU Instance](http://expressionflow.com/2016/10/09/installing-tensorflow-on-an-aws-ec2-p2-gpu-instance/)
    * [Updating Nvidia CUDA to 8.0.x in Ubuntu 16.04 – EC2 Gx instance](https://aichamp.wordpress.com/2016/11/09/updating-nvidia-cuda-to-8-0-x-in-ubuntu-16-04-ec2-gx-instance/)

## Installing ML-Agents

1. Move `python` sub-folder of this ml-agents repo to the remote ECS instance, and set it as the working directory.
2. Install the required packages with `pip3 install .`.

## Testing

To verify that all steps worked correctly:

1. In the Unity Editor, load a project containing an ML-Agents environment (you can use one of the example environments if you have not created your own).
2. Open the Build Settings window (menu: File > Build Settings).
3. Select Linux as the Target Platform, and x86_64 as the target architecture.
4. Check Headless Mode. 
5. Click Build to build the Unity environment executable.
6. Upload the executable to your EC2 instance.
7. Test the instance setup from Python using:

```python
from unityagents import UnityEnvironment

env = UnityEnvironment(<your_env>)
```
Where `<your_env>` corresponds to the path to your environment executable.
 
You should receive a message confirming that the environment was loaded successfully.

# Setting up a Custom Instance on Microsoft Azure for Training (works with the ML-Agents toolkit v0.3)

This page contains instructions for setting up a custom Virtual Machine on Microsoft Azure so you can running ML-Agents training in the cloud.

1.  Start by [deploying an Azure VM](https://docs.microsoft.com/azure/virtual-machines/linux/quick-create-portal) with Ubuntu Linux (tests were done with 16.04 LTS).  To use GPU support, use a N-Series VM.
2.  SSH into your VM.
3.  Start with the following commands to install the Nvidia driver:

```
wget http://us.download.nvidia.com/tesla/375.66/nvidia-diag-driver-local-repo-ubuntu1604_375.66-1_amd64.deb 

sudo dpkg -i nvidia-diag-driver-local-repo-ubuntu1604_375.66-1_amd64.deb 

sudo apt-get update 

sudo apt-get install cuda-drivers 

sudo reboot 
```

4.  After a minute you should be able to reconnect to your VM and install the CUDA toolkit:

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb 

sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb 

sudo apt-get update 

sudo apt-get install cuda-8-0 
```

5.  You'll next need to download cuDNN from the Nvidia developer site.  This requires a registered account.

6.  Navigate to [http://developer.nvidia.com](http://developer.nvidia.com) and create an account and verify it.

7.  Download (to your own computer) cuDNN from [this url](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v6/prod/8.0_20170307/Ubuntu16_04_x64/libcudnn6_6.0.20-1+cuda8.0_amd64-deb).  

8.  Copy the deb package to your VM: ```scp libcudnn6_6.0.21-1+cuda8.0_amd64.deb <VMUserName>@<VMIPAddress>:libcudnn6_6.0.21-1+cuda8.0_amd64.deb ```

9.  SSH back to your VM and execute the following:

```
sudo dpkg -i libcudnn6_6.0.21-1+cuda8.0_amd64.deb 

export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH 
. ~/.profile 

sudo reboot 
```

10.  After a minute, you should be able to SSH back into your VM.  After doing so, run the following:

```
sudo apt install python-pip 
sudo apt install python3-pip
```

11.  At this point, you need to install TensorFlow.  The version you install should be tied to if you are using GPU to train:

```
pip3 install tensorflow-gpu==1.4.0 keras==2.0.6 
```
Or CPU to train:
```
pip3 install tensorflow==1.4.0 keras==2.0.6 
```

12.  You'll then need to install additional dependencies:
```
pip3 install pillow 
pip3 install numpy 
pip3 install docopt 
```

13.  You can now return to the [main Azure instruction page](Training-on-Microsoft-Azure.md).
# Training on Amazon Web Service

This page contains instructions for setting up an EC2 instance on Amazon Web Service for training ML-Agents environments. 

## Preconfigured AMI

We've prepared an preconfigured AMI for you with the ID: `ami-18642967` in the `us-east-1` region. It was created as a modification of [Deep Learning AMI (Ubuntu)](https://aws.amazon.com/marketplace/pp/B077GCH38C). If you want to do training without the headless mode, you need to enable X Server on it. After launching your EC2 instance using the ami and ssh into it, run the following commands to enable it:

```
//Start the X Server, press Enter to come to the command line
sudo /usr/bin/X :0 &

//Check if Xorg process is running
//You will have a list of processes running on the GPU, Xorg should be in the list, as shown below
nvidia-smi
/*
 * Thu Jun 14 20:27:26 2018
 * +-----------------------------------------------------------------------------+
 * | NVIDIA-SMI 390.67                 Driver Version: 390.67                    |
 * |-------------------------------+----------------------+----------------------+
 * | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
 * | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
 * |===============================+======================+======================|
 * |   0  Tesla K80           On   | 00000000:00:1E.0 Off |                    0 |
 * | N/A   35C    P8    31W / 149W |      9MiB / 11441MiB |      0%      Default |
 * +-------------------------------+----------------------+----------------------+
 *
 * +-----------------------------------------------------------------------------+
 * | Processes:                                                       GPU Memory |
 * |  GPU       PID   Type   Process name                             Usage      |
 * |=============================================================================|
 * |    0      2331      G   /usr/lib/xorg/Xorg                             8MiB |
 * +-----------------------------------------------------------------------------+
 */

//Make the ubuntu use X Server for display
export DISPLAY=:0
```

## Configuring your own instance

You could also choose to configure your own instance. To begin with, you will need an EC2 instance which contains the latest Nvidia drivers, CUDA9, and cuDNN. In this tutorial we used the [Deep Learning AMI (Ubuntu)](https://aws.amazon.com/marketplace/pp/B077GCH38C) listed under AWS Marketplace with a p2.xlarge instance. 

### Installing the ML-Agents toolkit on the instance

After launching your EC2 instance using the ami and ssh into it:

1. Activate the python3 environment

    ```
    source activate python3
    ```

2. Clone the ML-Agents repo and install the required python packages

    ```
    git clone https://github.com/Unity-Technologies/ml-agents.git
    cd ml-agents/python
    pip3 install .
    ```

### Setting up X Server (optional)

X Server setup is only necessary if you want to do training that requires visual observation input. _Instructions here are adapted from this [Medium post](https://medium.com/towards-data-science/how-to-run-unity-on-amazon-cloud-or-without-monitor-3c10ce022639) on running general Unity applications in the cloud._

Current limitations of the Unity Engine require that a screen be available to render to when using visual observations. In order to make this possible when training on a remote server, a virtual screen is required. We can do this by installing Xorg and creating a virtual screen. Once installed and created, we can display the Unity environment in the virtual environment, and train as we would on a local machine. Ensure that `headless` mode is disabled when building linux executables which use visual observations.

1. Install and setup Xorg:

    ```
    //Install Xorg
    sudo apt-get update
    sudo apt-get install -y xserver-xorg mesa-utils
    sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024

    //Get the BusID information
    nvidia-xconfig --query-gpu-info

    //Add the BusID information to your /etc/X11/xorg.conf file
    sudo sed -i 's/    BoardName      "Tesla K80"/    BoardName      "Tesla K80"\n    BusID          "0:30:0"/g' /etc/X11/xorg.conf

    //Remove the Section "Files" from the /etc/X11/xorg.conf file
    sudo vim /etc/X11/xorg.conf //And remove two lines that contain Section "Files" and EndSection
    ```

2. Update and setup Nvidia driver:

    ```
    //Download and install the latest Nvidia driver for ubuntu
    wget http://download.nvidia.com/XFree86/Linux-x86_64/390.67/NVIDIA-Linux-x86_64-390.67.run
    sudo /bin/bash ./NVIDIA-Linux-x86_64-390.67.run --accept-license --no-questions --ui=none

    //Disable Nouveau as it will clash with the Nvidia driver
    sudo echo 'blacklist nouveau'  | sudo tee -a /etc/modprobe.d/blacklist.conf
    sudo echo 'options nouveau modeset=0'  | sudo tee -a /etc/modprobe.d/blacklist.conf
    sudo echo options nouveau modeset=0 | sudo tee -a /etc/modprobe.d/nouveau-kms.conf
    sudo update-initramfs -u
    ```

2. Restart the EC2 instance:

    ```
    sudo reboot now
    ```

3. Make sure there are no Xorg processes running:

   ```
   //Kill any possible running Xorg processes
   //Note that you might have to run this command multiple times depending on how Xorg is configured.
   sudo killall Xorg

   //Check if there is any Xorg process left
   //You will have a list of processes running on the GPU, Xorg should not be in the list, as shown below.
   nvidia-smi
   /*
    * Thu Jun 14 20:21:11 2018
    * +-----------------------------------------------------------------------------+
    * | NVIDIA-SMI 390.67                 Driver Version: 390.67                    |
    * |-------------------------------+----------------------+----------------------+
    * | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    * | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    * |===============================+======================+======================|
    * |   0  Tesla K80           On   | 00000000:00:1E.0 Off |                    0 |
    * | N/A   37C    P8    31W / 149W |      0MiB / 11441MiB |      0%      Default |
    * +-------------------------------+----------------------+----------------------+
    *
    * +-----------------------------------------------------------------------------+
    * | Processes:                                                       GPU Memory |
    * |  GPU       PID   Type   Process name                             Usage      |
    * |=============================================================================|
    * |  No running processes found                                                 |
    * +-----------------------------------------------------------------------------+
    */
   ```

4. Start X Server and make the ubuntu use X Server for display:

    ```
    //Start the X Server, press Enter to come to the command line
    sudo /usr/bin/X :0 &

    //Check if Xorg process is running
    //You will have a list of processes running on the GPU, Xorg should be in the list.
    nvidia-smi

    //Make the ubuntu use X Server for display
    export DISPLAY=:0
    ```

5. Ensure the Xorg is correctly configured:

    ```
    //For more information on glxgears, see ftp://www.x.org/pub/X11R6.8.1/doc/glxgears.1.html. 
    glxgears
    //If Xorg is configured correctly, you should see the following message
    /*
     * Running synchronized to the vertical refresh.  The framerate should be
     * approximately the same as the monitor refresh rate.
     * 137296 frames in 5.0 seconds = 27459.053 FPS
     * 141674 frames in 5.0 seconds = 28334.779 FPS
     * 141490 frames in 5.0 seconds = 28297.875 FPS
     */
    ```

## Training on EC2 instance

1. In the Unity Editor, load a project containing an ML-Agents environment (you can use one of the example environments if you have not created your own).
2. Open the Build Settings window (menu: File > Build Settings).
3. Select Linux as the Target Platform, and x86_64 as the target architecture.
4. Check Headless Mode (If you haven't setup the X Server).
5. Click Build to build the Unity environment executable.
6. Upload the executable to your EC2 instance within `ml-agents/python` folder.
7. Test the instance setup from Python using:

    ```python
    from unityagents import UnityEnvironment

    env = UnityEnvironment(<your_env>)
    ```
    Where `<your_env>` corresponds to the path to your environment executable.

    You should receive a message confirming that the environment was loaded successfully.
8. Train the executable

    ```
    //cd into your ml-agents/python folder
    chmod +x <your_env>.x86_64
    python learn.py <your_env> --train
    ```
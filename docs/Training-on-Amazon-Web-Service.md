# Training on Amazon Web Service

Note: We no longer use this guide ourselves and so it may not work correctly. We've
decided to keep it up just in case it is helpful to you.

This page contains instructions for setting up an EC2 instance on Amazon Web
Service for training ML-Agents environments.

## Pre-configured AMI

We've prepared a pre-configured AMI for you with the ID: `ami-016ff5559334f8619` in the
`us-east-1` region. It was created as a modification of [Deep Learning AMI
(Ubuntu)](https://aws.amazon.com/marketplace/pp/B077GCH38C). The AMI has been
tested with p2.xlarge instance. Furthermore, if you want to train without
headless mode, you need to enable X Server.

After launching your EC2 instance using the ami and ssh into it, run the
following commands to enable it:

```sh
# Start the X Server, press Enter to come to the command line
$ sudo /usr/bin/X :0 &

# Check if Xorg process is running
# You will have a list of processes running on the GPU, Xorg should be in the
# list, as shown below
$ nvidia-smi

# Thu Jun 14 20:27:26 2018
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 390.67                 Driver Version: 390.67                    |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  Tesla K80           On   | 00000000:00:1E.0 Off |                    0 |
# | N/A   35C    P8    31W / 149W |      9MiB / 11441MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+
#
# +-----------------------------------------------------------------------------+
# | Processes:                                                       GPU Memory |
# |  GPU       PID   Type   Process name                             Usage      |
# |=============================================================================|
# |    0      2331      G   /usr/lib/xorg/Xorg                             8MiB |
# +-----------------------------------------------------------------------------+

# Make the ubuntu use X Server for display
$ export DISPLAY=:0
```

## Configuring your own instance

You could also choose to configure your own instance. To begin with, you will
need an EC2 instance which contains the latest Nvidia drivers, CUDA9, and cuDNN.
In this tutorial we used the
[Deep Learning AMI (Ubuntu)](https://aws.amazon.com/marketplace/pp/B077GCH38C)
listed under AWS Marketplace with a p2.xlarge instance.

### Installing the ML-Agents toolkit on the instance

After launching your EC2 instance using the ami and ssh into it:

1. Activate the python3 environment

    ```sh
    source activate python3
    ```

2. Clone the ML-Agents repo and install the required Python packages

    ```sh
    git clone https://github.com/Unity-Technologies/ml-agents.git
    cd ml-agents/ml-agents/
    pip3 install -e .
    ```

### Setting up X Server (optional)

X Server setup is only necessary if you want to do training that requires visual
observation input. _Instructions here are adapted from this
[Medium post](https://medium.com/towards-data-science/how-to-run-unity-on-amazon-cloud-or-without-monitor-3c10ce022639)
on running general Unity applications in the cloud._

Current limitations of the Unity Engine require that a screen be available to
render to when using visual observations. In order to make this possible when
training on a remote server, a virtual screen is required. We can do this by
installing Xorg and creating a virtual screen. Once installed and created, we
can display the Unity environment in the virtual environment, and train as we
would on a local machine. Ensure that `headless` mode is disabled when building
linux executables which use visual observations.

#### Install and setup Xorg:

    ```sh
    # Install Xorg
    $ sudo apt-get update
    $ sudo apt-get install -y xserver-xorg mesa-utils
    $ sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024

    # Get the BusID information
    $ nvidia-xconfig --query-gpu-info

    # Add the BusID information to your /etc/X11/xorg.conf file
    $ sudo sed -i 's/    BoardName      "Tesla K80"/    BoardName      "Tesla K80"\n    BusID          "0:30:0"/g' /etc/X11/xorg.conf

    # Remove the Section "Files" from the /etc/X11/xorg.conf file
    # And remove two lines that contain Section "Files" and EndSection
    $ sudo vim /etc/X11/xorg.conf
    ```

#### Update and setup Nvidia driver:

    ```sh
    # Download and install the latest Nvidia driver for ubuntu
    # Please refer to http://download.nvidia.com/XFree86/Linux-#x86_64/latest.txt
    $ wget http://download.nvidia.com/XFree86/Linux-x86_64/390.87/NVIDIA-Linux-x86_64-390.87.run
    $ sudo /bin/bash ./NVIDIA-Linux-x86_64-390.67.run --accept-license --no-questions --ui=none

    # Disable Nouveau as it will clash with the Nvidia driver
    $ sudo echo 'blacklist nouveau'  | sudo tee -a /etc/modprobe.d/blacklist.conf
    $ sudo echo 'options nouveau modeset=0'  | sudo tee -a /etc/modprobe.d/blacklist.conf
    $ sudo echo options nouveau modeset=0 | sudo tee -a /etc/modprobe.d/nouveau-kms.conf
    $ sudo update-initramfs -u
    ```

#### Restart the EC2 instance:

    ```sh
    sudo reboot now
    ```

#### Make sure there are no Xorg processes running:

   ```sh
   # Kill any possible running Xorg processes
   # Note that you might have to run this command multiple times depending on
   # how Xorg is configured.
   $ sudo killall Xorg

   # Check if there is any Xorg process left
   # You will have a list of processes running on the GPU, Xorg should not be in
   # the list, as shown below.
   $ nvidia-smi

   # Thu Jun 14 20:21:11 2018
   # +-----------------------------------------------------------------------------+
   # | NVIDIA-SMI 390.67                 Driver Version: 390.67                    |
   # |-------------------------------+----------------------+----------------------+
   # | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
   # | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
   # |===============================+======================+======================|
   # |   0  Tesla K80           On   | 00000000:00:1E.0 Off |                    0 |
   # | N/A   37C    P8    31W / 149W |      0MiB / 11441MiB |      0%      Default |
   # +-------------------------------+----------------------+----------------------+
   #
   # +-----------------------------------------------------------------------------+
   # | Processes:                                                       GPU Memory |
   # |  GPU       PID   Type   Process name                             Usage      |
   # |=============================================================================|
   # |  No running processes found                                                 |
   # +-----------------------------------------------------------------------------+

   ```

#### Start X Server and make the ubuntu use X Server for display:

    ```console
    # Start the X Server, press Enter to come back to the command line
    $ sudo /usr/bin/X :0 &

    # Check if Xorg process is running
    # You will have a list of processes running on the GPU, Xorg should be in the list.
    $ nvidia-smi

    # Make the ubuntu use X Server for display
    $ export DISPLAY=:0
    ```

#### Ensure the Xorg is correctly configured:

    ```sh
    # For more information on glxgears, see ftp://www.x.org/pub/X11R6.8.1/doc/glxgears.1.html.
    $ glxgears
    # If Xorg is configured correctly, you should see the following message

    # Running synchronized to the vertical refresh.  The framerate should be
    # approximately the same as the monitor refresh rate.
    # 137296 frames in 5.0 seconds = 27459.053 FPS
    # 141674 frames in 5.0 seconds = 28334.779 FPS
    # 141490 frames in 5.0 seconds = 28297.875 FPS

    ```

## Training on EC2 instance

1. In the Unity Editor, load a project containing an ML-Agents environment (you
   can use one of the example environments if you have not created your own).
2. Open the Build Settings window (menu: File > Build Settings).
3. Select Linux as the Target Platform, and x86_64 as the target architecture
(the default x86 currently does not work).
4. Check Headless Mode if you have not setup the X Server. (If you do not use
Headless Mode, you have to setup the X Server to enable training.)
5. Click Build to build the Unity environment executable.
6. Upload the executable to your EC2 instance within `ml-agents` folder.
7. Change the permissions of the executable.

    ```sh
    chmod +x <your_env>.x86_64
    ```
8. (Without Headless Mode) Start X Server and use it for display:

    ```sh
    # Start the X Server, press Enter to come back to the command line
    $ sudo /usr/bin/X :0 &

    # Check if Xorg process is running
    # You will have a list of processes running on the GPU, Xorg should be in the list.
    $ nvidia-smi

    # Make the ubuntu use X Server for display
    $ export DISPLAY=:0
    ```
9. Test the instance setup from Python using:

    ```python
    from mlagents.envs.environment import UnityEnvironment

    env = UnityEnvironment(<your_env>)
    ```

    Where `<your_env>` corresponds to the path to your environment executable.

    You should receive a message confirming that the environment was loaded successfully.
10. Train your models

    ```console
    mlagents-learn <trainer-config-file> --env=<your_env> --train
    ```

## FAQ

### The <Executable_Name>_Data folder hasn't been copied cover

If you've built your Linux executable, but forget to copy over the corresponding <Executable_Name>_Data folder, you will see error message like the following:

```sh
Set current directory to /home/ubuntu/ml-agents/ml-agents
Found path: /home/ubuntu/ml-agents/ml-agents/3dball_linux.x86_64
no boot config - using default values

(Filename:  Line: 403)

There is no data folder
```

### Unity Environment not responding

If you didn't setup X Server or hasn't launched it properly, or your environment somehow crashes, or you haven't `chmod +x` your Unity Environment, all of these will cause connection between Unity and Python to fail. Then you will see something like this:

```console
Logging to /home/ubuntu/.config/unity3d/<Some_Path>/Player.log
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/ubuntu/ml-agents/ml-agents/mlagents/envs/environment.py", line 63, in __init__
    aca_params = self.send_academy_parameters(rl_init_parameters_in)
  File "/home/ubuntu/ml-agents/ml-agents/mlagents/envs/environment.py", line 489, in send_academy_parameters
    return self.communicator.initialize(inputs).rl_initialization_output
  File "/home/ubuntu/ml-agents/ml-agents/mlagents/envs/rpc_communicator.py", line 60, in initialize
mlagents.envs.exception.UnityTimeOutException: The Unity environment took too long to respond. Make sure that :
         The environment does not need user interaction to launch
         The environment and the Python interface have compatible versions.
```

It would be also really helpful to check your /home/ubuntu/.config/unity3d/<Some_Path>/Player.log to see what happens with your Unity environment.

### Could not launch X Server

When you execute:

```sh
sudo /usr/bin/X :0 &
```

You might see something like:

```sh
X.Org X Server 1.18.4
...
(==) Log file: "/var/log/Xorg.0.log", Time: Thu Oct 11 21:10:38 2018
(==) Using config file: "/etc/X11/xorg.conf"
(==) Using system config directory "/usr/share/X11/xorg.conf.d"
(EE)
Fatal server error:
(EE) no screens found(EE)
(EE)
Please consult the The X.Org Foundation support
         at http://wiki.x.org
 for help.
(EE) Please also check the log file at "/var/log/Xorg.0.log" for additional information.
(EE)
(EE) Server terminated with error (1). Closing log file.
```

And when you execute:

```sh
nvidia-smi
```

You might see something like:

```sh
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.
```
This means the NVIDIA's driver needs to be updated. Refer to [this section](Training-on-Amazon-Web-Service.md#update-and-setup-nvidia-driver) for more information.

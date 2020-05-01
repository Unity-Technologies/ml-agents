# Training on Microsoft Azure (works with ML-Agents Toolkit v0.3)

:warning: **Note:** We no longer use this guide ourselves and so it may not work
correctly. We've decided to keep it up just in case it is helpful to you.

This page contains instructions for setting up training on Microsoft Azure
through either
[Azure Container Instances](https://azure.microsoft.com/services/container-instances/)
or Virtual Machines. Non "headless" training has not yet been tested to verify
support.

## Pre-Configured Azure Virtual Machine

A pre-configured virtual machine image is available in the Azure Marketplace and
is nearly completely ready for training. You can start by deploying the
[Data Science Virtual Machine for Linux (Ubuntu)](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-dsvm.ubuntu-1804)
into your Azure subscription.

Note that, if you choose to deploy the image to an
[N-Series GPU optimized VM](https://docs.microsoft.com/azure/virtual-machines/linux/sizes-gpu),
training will, by default, run on the GPU. If you choose any other type of VM,
training will run on the CPU.

## Configuring your own Instance

Setting up your own instance requires a number of package installations. Please
view the documentation for doing so [here](#custom-instances).

## Installing ML-Agents

1. [Move](https://docs.microsoft.com/en-us/azure/virtual-machines/linux/copy-files-to-linux-vm-using-scp)
   the `ml-agents` sub-folder of this ml-agents repo to the remote Azure
   instance, and set it as the working directory.
2. Install the required packages with `pip3 install .`.

## Testing

To verify that all steps worked correctly:

1. In the Unity Editor, load a project containing an ML-Agents environment (you
   can use one of the example environments if you have not created your own).
2. Open the Build Settings window (menu: File > Build Settings).
3. Select Linux as the Target Platform, and x86_64 as the target architecture.
4. Check Headless Mode.
5. Click Build to build the Unity environment executable.
6. Upload the resulting files to your Azure instance.
7. Test the instance setup from Python using:

```python
from mlagents_envs.environment import UnityEnvironment

env = UnityEnvironment(<your_env>)
```

Where `<your_env>` corresponds to the path to your environment executable.

You should receive a message confirming that the environment was loaded
successfully.

## Running Training on your Virtual Machine

To run your training on the VM:

1. [Move](https://docs.microsoft.com/en-us/azure/virtual-machines/linux/copy-files-to-linux-vm-using-scp)
   your built Unity application to your Virtual Machine.
2. Set the directory where the ML-Agents Toolkit was installed to your working
   directory.
3. Run the following command:

```sh
mlagents-learn <trainer_config> --env=<your_app> --run-id=<run_id> --train
```

Where `<your_app>` is the path to your app (i.e.
`~/unity-volume/3DBallHeadless`) and `<run_id>` is an identifier you would like
to identify your training run with.

If you've selected to run on a N-Series VM with GPU support, you can verify that
the GPU is being used by running `nvidia-smi` from the command line.

## Monitoring your Training Run with TensorBoard

Once you have started training, you can
[use TensorBoard to observe the training](Using-Tensorboard.md).

1. Start by
   [opening the appropriate port for web traffic to connect to your VM](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/nsg-quickstart-portal).

   - Note that you don't need to generate a new `Network Security Group` but
     instead, go to the **Networking** tab under **Settings** for your VM.
   - As an example, you could use the following settings to open the Port with
     the following Inbound Rule settings:
     - Source: Any
     - Source Port Ranges: \*
     - Destination: Any
     - Destination Port Ranges: 6006
     - Protocol: Any
     - Action: Allow
     - Priority: (Leave as default)

2. Unless you started the training as a background process, connect to your VM
   from another terminal instance.
3. Run the following command from your terminal
   `tensorboard --logdir=summaries --host 0.0.0.0`
4. You should now be able to open a browser and navigate to
   `<Your_VM_IP_Address>:6060` to view the TensorBoard report.

## Running on Azure Container Instances

[Azure Container Instances](https://azure.microsoft.com/services/container-instances/)
allow you to spin up a container, on demand, that will run your training and
then be shut down. This ensures you aren't leaving a billable VM running when it
isn't needed. Using ACI enables you to offload training of your models without
needing to install Python and TensorFlow on your own computer.

## Custom Instances

This page contains instructions for setting up a custom Virtual Machine on
Microsoft Azure so you can running ML-Agents training in the cloud.

1. Start by
   [deploying an Azure VM](https://docs.microsoft.com/azure/virtual-machines/linux/quick-create-portal)
   with Ubuntu Linux (tests were done with 16.04 LTS). To use GPU support, use a
   N-Series VM.
2. SSH into your VM.
3. Start with the following commands to install the Nvidia driver:

   ```sh
   wget http://us.download.nvidia.com/tesla/375.66/nvidia-diag-driver-local-repo-ubuntu1604_375.66-1_amd64.deb

   sudo dpkg -i nvidia-diag-driver-local-repo-ubuntu1604_375.66-1_amd64.deb

   sudo apt-get update

   sudo apt-get install cuda-drivers

   sudo reboot
   ```

4. After a minute you should be able to reconnect to your VM and install the
   CUDA toolkit:

   ```sh
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb

   sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb

   sudo apt-get update

   sudo apt-get install cuda-8-0
   ```

5. You'll next need to download cuDNN from the Nvidia developer site. This
   requires a registered account.

6. Navigate to [http://developer.nvidia.com](http://developer.nvidia.com) and
   create an account and verify it.

7. Download (to your own computer) cuDNN from
   [this url](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v6/prod/8.0_20170307/Ubuntu16_04_x64/libcudnn6_6.0.20-1+cuda8.0_amd64-deb).

8. Copy the deb package to your VM:

   ```sh
   scp libcudnn6_6.0.21-1+cuda8.0_amd64.deb <VMUserName>@<VMIPAddress>:libcudnn6_6.0.21-1+cuda8.0_amd64.deb
   ```

9. SSH back to your VM and execute the following:

   ```console
   sudo dpkg -i libcudnn6_6.0.21-1+cuda8.0_amd64.deb

   export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
   . ~/.profile

   sudo reboot
   ```

10. After a minute, you should be able to SSH back into your VM. After doing so,
    run the following:

    ```sh
    sudo apt install python-pip
    sudo apt install python3-pip
    ```

11. At this point, you need to install TensorFlow. The version you install
    should be tied to if you are using GPU to train:

    ```sh
    pip3 install tensorflow-gpu==1.4.0 keras==2.0.6
    ```

    Or CPU to train:

    ```sh
    pip3 install tensorflow==1.4.0 keras==2.0.6
    ```

12. You'll then need to install additional dependencies:

    ```sh
    pip3 install pillow
    pip3 install numpy
    ```

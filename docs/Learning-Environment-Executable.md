# Using an Environment Executable

This section will help you create and use built environments rather than the Editor to interact with an environment. Using an executable has some advantages over using the Editor : 

 * You can exchange executable with other people without having to share your entire repository.
 * You can put your executable on a remote machine for faster training.
 * You can use `Headles` mode for faster training.
 * You can keep using the Unity Editor for other tasks while the agents are training.

## Building the environment

The first step is to open the Unity scene containing the 3D Balance Ball
environment:

1. Launch Unity.
2. On the Projects dialog, choose the **Open** option at the top of the window.
3. Using the file dialog that opens, locate the `unity-environment` folder 
within the ML-Agents project and click **Open**.
4. In the **Project** window, navigate to the folder 
`Assets/ML-Agents/Examples/3DBall/`.
5. Double-click the `3DBall` file to load the scene containing the Balance 
Ball environment.

![3DBall Scene](images/mlagents-Open3DBall.png)

Make sure the Brains in the scene have the right type. For example, if you want to be able to control your agents from Python, you will need to set the corresponding brain to **External**.

1. In the **Scene** window, click the triangle icon next to the Ball3DAcademy 
object.
2. Select its child object **Ball3DBrain**.
3. In the Inspector window, set **Brain Type** to **External**.

![Set Brain to External](images/mlagents-SetExternalBrain.png)

Next, we want the set up scene to play correctly when we 
launch our environment executable. This means:
* The environment application runs in the background
* No dialogs require interaction
* The correct scene loads automatically
 
1. Open Player Settings (menu: **Edit** > **Project Settings** > **Player**).
2. Under **Resolution and Presentation**:
    - Ensure that **Run in Background** is Checked.
    - Ensure that **Display Resolution Dialog** is set to Disabled.
3. Open the Build Settings window (menu:**File** > **Build Settings**).
4. Choose your target platform.
    - (optional) Select “Development Build” to
    [log debug messages](https://docs.unity3d.com/Manual/LogFiles.html).
5. If any scenes are shown in the **Scenes in Build** list, make sure that 
the 3DBall Scene is the only one checked. (If the list is empty, than only the 
current scene is included in the build).
6. Click **Build**:
    a. In the File dialog, navigate to the `python` folder in your ML-Agents 
    directory.
    b. Assign a file name and click **Save**.

![Build Window](images/mlagents-BuildWindow.png)

Now that we have a Unity executable containing the simulation environment, we 
can interact with it.

## Interacting with the Environment
If you want to use the [Python API](Python-API.md) to interact with your executable, you can pass the name of the executable with the argument 'file_name' of the `UnityEnvironment`. For instance :

```python
from unityagents import UnityEnvironment
env = UnityEnvironment(file_name=<env_name>)
```

If you are using `learn.py` you can pass the name of the executable in the command line. For instance : 

```
python3 learn.py <env_name> --run-id=<run-identifier> --train
```


# Limitations and Common Issues

## Unity SDK
### Headless Mode
Currently headless mode is disabled. We hope to address these in a future version of Unity.

### Rendering Speed and Synchronization
Currently the speed of the game physics can only be increased to 100x real-time. The Academy also moves in time with FixedUpdate() rather than Update(), so game behavior tied to frame updates may be out of sync. 

### macOS Metal Support
When running a Unity Environment on macOS using Metal rendering, the application can crash when the lock-screen is open. The solution is to set rendering to OpenGL. This can be done by navigating: `Edit -> Project Settings -> Player`. Clicking on `Other Settings`. Unchecking `Auto Graphics API for Mac`. Setting `OpenGL Core` to be above `Metal` in the priority list.

## Python API

### Environment Permission Error

If you directly import your Unity environment without building it in the editor, you might need to give it additionnal permissions to execute it. 

If you receive such a permission error on macOS, run:

`chmod -R 755 *.app` 

or on Linux:

`chmod -R 755 *.x86_64` 

On Windows, you can find instructions [here](https://technet.microsoft.com/en-us/library/cc754344(v=ws.11).aspx).

### Environment Connection Timeout

If you are able to launch the environment from `UnityEnvironment` but then recieve a timeout error, there may be a number of possible causes.
 * _Cause_: There may be no Brains in your environment which are set to `External`.  In this case, the environment will not attempt to communicate with python. _Solution_: Set the train you wish to externally control through the Python API to `External` from the Unity Editor, and rebuild the environment.
 * _Cause_: On OSX, the firewall may be preventing communication with the environment. _Solution_: Add the built environment binary to the list of exceptions on the firewall by following instructions [here](https://support.apple.com/en-us/HT201642). 

### Filename not found

If you receive a file-not-found error while attempting to launch an environment, ensure that the environment files are in the root repository directory. For example, if there is a sub-folder containing the environment files, those files should be removed from the sub-folder and moved to the root. 

### Communication port {} still in use

If you receive an exception `"Couldn't launch new environment because communication port {} is still in use. "`, you can change the worker number in the python script when calling 

`UnityEnvironment(file_name=filename, worker_id=X)`

### Mean reward : nan

If you recieve a message `Mean reward : nan` when attempting to train a model using PPO, this is due to the episodes of the learning environment not terminating. In order to address this, set `Max Steps` for either the Academy or Agents within the Scene Inspector to a value greater than 0. Alternatively, it is possible to manually set `done` conditions for episodes from within scripts for custom episode-terminating events.

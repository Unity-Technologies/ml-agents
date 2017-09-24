# Limitations and Common Issues

## Unity SDK
### Headless Mode
Currently headless mode is disabled. We hope to address these in a future version of Unity.

### Rendering Speed and Synchronization
Currently the speed of the game physics can only be increased to 100x real-time. The Academy also moves in time with FixedUpdate() rather than Update(), so game behavior tied to frame updates may be out of sync. 

### macOS Metal Support
When running a Unity Environment on macOS using Metal rendering, the application can crash when the lock-screen is open. The solution is to set rendering to OpenGL. This can be done by navigating: `Edit -> Project Settings -> Player`. Clicking on `Other Settings`. Unchecking `Auto Graphics API for Mac`. Setting `OpenGL Core` to be above `Metal` in the priority list.

## Python API

### Unity Environment Permission Error

If you directly import your Unity environment without building it in the editor, you might need to give it additionnal permissions to execute it. 

If you receive such a permission error on macOS, run:

`chmod -R 755 *.app` 

or on Linux:

`chmod -R 755 *.x86_64` 

On Windows, you can find instructions [here](https://technet.microsoft.com/en-us/library/cc754344(v=ws.11).aspx).

### Filename not found

If you receive a file-not-found error while attempting to launch an environment, ensure that the environment files are in the root repository directory. For example, if there is a sub-folder containing the environment files, those files should be removed from the sub-folder and moved to the root. 

### Communication port {} still in use

If you receive an exception `"Couldn't launch new environment because communication port {} is still in use. "`, you can change the worker number in the python script when calling 

`UnityEnvironment(file_name=filename, worker_num=X)`

# Using TensorFlowSharp in Unity (Experimental)

The ML-Agents toolkit allows you to use pre-trained
[TensorFlow graphs](https://www.tensorflow.org/programmers_guide/graphs)
inside your Unity
games. This support is possible thanks to the
[TensorFlowSharp project](https://github.com/migueldeicaza/TensorFlowSharp).
The primary purpose for this support is to use the TensorFlow models produced by
the ML-Agents toolkit's own training programs, but a side benefit is that you
can use any TensorFlow model.

_Notice: This feature is still experimental. While it is possible to embed
trained models into Unity games, Unity Technologies does not officially support
this use-case for production games at this time. As such, no guarantees are
provided regarding the quality of experience. If you encounter issues regarding
battery life, or general performance (especially on mobile), please let us
know._

## Supported devices

* Linux 64 bits
* Mac OS X 64 bits
* Windows 64 bits
* iOS (Requires additional steps)
* Android

## Requirements

* Unity 2017.4 or above
* Unity TensorFlow Plugin ([Download here](https://s3.amazonaws.com/unity-ml-agents/0.5/TFSharpPlugin.unitypackage))

## Using TensorFlowSharp with ML-Agents

Go to `Edit` -> `Player Settings` and add `ENABLE_TENSORFLOW` to the `Scripting
Define Symbols` for each type of device you want to use (**`PC, Mac and Linux
Standalone`**, **`iOS`** or **`Android`**).

Set the Brain you used for training to `Learning`. Drag `your_name_graph.bytes`
into Unity and then drag it into The `Model` field in the Brain.

## iOS additional instructions for building

* Before build your game against iOS platform, make sure you've set the
  flag `ENABLE_TENSORFLOW` for it.
* Once you build the project for iOS in the editor, open the .xcodeproj file
  within the project folder using Xcode.
* Set up your ios account following the
  [iOS Account setup page](https://docs.unity3d.com/Manual/iphone-accountsetup.html).
* In **Build Settings** > **Linking** > **Other Linker Flags**:
  * Double click on the flag list to expand the list
  * Add `-force_load`
  * Drag the library `libtensorflow-core.a` from the **Project Navigator** on
    the left under `Libraries/ML-Agents/Plugins/iOS` into the flag list, after
    `-force_load`.


# Unity Inference Engine

The ML-Agents toolkit allows you to use pre-trained neural network models
inside your Unity games. This support is possible thanks to the Unity Inference
Engine. The Unity Inference Engine is using
[compute shaders](https://docs.unity3d.com/Manual/class-ComputeShader.html)
to run the neural network within Unity.

__Note__: The ML-Agents toolkit only supports the models created with our
trainers.

## Supported devices

Scripting Backends : The Unity Inference Engine is generally faster with
__IL2CPP__ than with __Mono__ for Standalone builds.
In the Editor, It is not possible to use the Unity Inference Engine with
GPU device selected when Editor Graphics Emulation is set to __OpenGL(ES)
3.0 or 2.0 emulation__. Also there might be non-fatal build time errors
when target platform includes Graphics API that does not support
__Unity Compute Shaders__.
The Unity Inference Engine supposedly works on any Unity supported platform
but we only tested for the following platforms :

* Linux 64 bits
* Mac OS X 64 bits (`OpenGLCore` Graphics API is not supported)
* Windows 64 bits
* iOS
* Android

## Supported formats
There are currently two supported model formats:
 * Barracuda (`.nn`) files use a proprietary format produced by the [`tensorflow_to_barracuda.py`]() script.
 * ONNX (`.onnx`) files use an [industry-standard open format](https://onnx.ai/about.html) produced by the [tf2onnx package](https://github.com/onnx/tensorflow-onnx).

Export to ONNX is currently considered beta. To enable it, make sure `tf2onnx>=1.5.5` is installed in pip.
tf2onnx does not currently support tensorflow 2.0.0 or later, or earlier than 1.12.0.

## Using the Unity Inference Engine

When using a model, drag the model file into the **Model** field in the Inspector of the Agent.
Select the **Inference Device** : CPU or GPU you want to use for Inference.

**Note:** For most of the models generated with the ML-Agents toolkit, CPU will be faster than GPU.
You should use the GPU only if you use the
ResNet visual encoder or have a large number of agents with visual observations.

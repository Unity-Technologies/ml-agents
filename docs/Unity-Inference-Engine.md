# Unity Inference Engine

The ML-Agents toolkit allows you to use pre-trained neural network models
inside your Unity games. This support is possible thanks to the Unity Inference
Engine. The Unity Inference Engine is using 
[compute shaders](https://docs.unity3d.com/Manual/class-ComputeShader.html) 
to run the neural network within Unity. 

## Supported devices

The Unity Inference Engine supposedly works on any Unity supported platform
but we only tested for the following platforms :

* Linux 64 bits
* Mac OS X 64 bits (`OpenGLCore` Graphics API is not supported)
* Windows 64 bits
* iOS
* Android

## Using the Unity Inference Engine

When using a **Learning Brain**, drag the `.nn` file into the **Model** field 
in the Inspector. 
Uncheck the `Control` checkbox for the corresponding **Brain** in the 
**BroadcastHub** of the Academy.
Select the **Inference Device** : CPU or GPU you want to use for Inference.

_Note: For most of the models generated with the ML-Agents toolkit, CPU will be faster than GPU. Only use GPU if you have a large number of agents using visual observations._

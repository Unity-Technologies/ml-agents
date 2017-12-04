# Using TensorFlowSharp in Unity (Experimental)

Unity now offers the possibility to use pretrained TensorFlow graphs inside of the game engine. This was made possible thanks to [the TensorFlowSharp project](https://github.com/migueldeicaza/TensorFlowSharp).

_Notice: This feature is still experimental. While it is possible to embed trained models into Unity games, Unity Technologies does not officially support this use-case for production games at this time. As such, no guarantees are provided regarding the quality of experience. If you encounter issues regarding battery life, or general performance (especially on mobile), please let us know._

## Supported devices :

 * Linux 64 bits
 * Mac OSX 64 bits
 * Windows 64 bits
 * iOS (Requires additional steps)
 * Android

## Requirements

* Unity 2017.1 or above
* Unity Tensorflow Plugin ([Download here](https://s3.amazonaws.com/unity-agents/0.2/TFSharpPlugin.unitypackage))
# Using TensorflowSharp with ML-Agents

In order to bring a fully trained agent back into Unity, you will need to make sure the nodes of your graph have appropriate names. You can give names to nodes in Tensorflow :
```python
variable= tf.identity(variable, name="variable_name")
```
We recommend using the following naming convention:
 * Name the batch size input placeholder `batch_size`
 * Name the input state placeholder `state`
 * Name the output node `action`
 * Name the recurrent vector (memory) input placeholder `recurrent_in` (if any)
 * Name the recurrent vector (memory) output node `recurrent_out` (if any)
 * Name the observations placeholders input placeholders `observation_i` where `i` is the index of the observation (starting at 0)

You can have additional placeholders for float or integers but they must be placed in placeholders of dimension 1 and size 1. (Be sure to name them)

It is important that the inputs and outputs of the graph are exactly the one you receive / give when training your model with an `External` brain. This means you cannot have any operations such as reshaping outside of the graph.
The object you get by calling `step` or `reset` has fields `states`, `observations` and `memories` which must correspond to the placeholders of your graph. Similarly, the arguments `action` and `memory` you pass to `step` must correspond to the output nodes of your graph.

While training your Agent using the Python API, you can save your graph at any point of the training. Note that the argument `output_node_names` must be the name of the tensor your graph outputs (separated by a coma if multiple outputs). In this case, it will be either `action` or `action,recurrent_out` if you have recurrent outputs.
```python
from tensorflow.python.tools import freeze_graph

freeze_graph.freeze_graph(input_graph = model_path +'/raw_graph_def.pb',
              input_binary = True,
              input_checkpoint = last_checkpoint,
              output_node_names = "action",
              output_graph = model_path +'/your_name_graph.bytes' ,
              clear_devices = True, initializer_nodes = "",input_saver = "",
              restore_op_name = "save/restore_all", filename_tensor_name = "save/Const:0")
```
Your model will be saved with the name `your_name_graph.bytes` and will contain both the graph and associated weights. Note that you must save your graph as a bytes file so Unity can load it.

## Inside Unity

Go to `Edit` -> `Player Settings` and add `ENABLE_TENSORFLOW` to the `Scripting Define Symbols` for each type of device you want to use (**`PC, Mac and Linux Standalone`**, **`iOS`** or **`Android`**).

Set the Brain you used for training to `Internal`. Drag `your_name_graph.bytes` into Unity and then drag it into The `Graph Model` field in the Brain. If you used a scope when training you graph, specify it in the `Graph Scope` field. Specify the names of the nodes you used in your graph. If you followed these instructions well, the agents in your environment that use this brain will use you fully trained network to make decisions.

# iOS additional instructions for building

* Once you build for iOS in the editor, Xcode will launch.
* In `General` -> `Linked Frameworks and Libraries`:
  * Add a framework called `Framework.accelerate`
  * Remove the library `libtensorflow-core.a`
* In `Build Settings`->`Linking`->`Other Linker Flags`:
  * Double Click on the flag list
  * Type `-force_load`
  * Drag the library `libtensorflow-core.a` from the `Project Navigator` on the left under `Libraries/ML-Agents/Plugins/iOS` into the flag list.

# Using TensorflowSharp without ML-Agents

Beyond controlling an in-game agent, you may desire to use TensorFlowSharp for more general computation. The below instructions describe how to generally embed Tensorflow models without using the ML-Agents framework.

You must have a Tensorflow graph `your_name_graph.bytes` made using Tensorflow's `freeze_graph.py`. The process to create such graph is explained above.

## Inside of Unity

Put the file `your_name_graph.bytes` into Resources.

In your C# script :
At the top, add the line
```csharp
using Tensorflow;
```
If you will be building for android, you must add this block at the start of your code :
```csharp
#if UNITY_ANDROID
TensorFlowSharp.Android.NativeBinding.Init();
#endif
```
Put your graph as a text asset in the variable `graphModel`. You can do so in the inspector by making `graphModel` a public variable and dragging you asset in the inspector or load it from the Resources folder :
```csharp
TextAsset graphModel = Resources.Load (your_name_graph) as TextAsset;
```
You then must recreate the graph in Unity by adding the code :
```csharp
graph = new TFGraph ();
graph.Import (graphModel.bytes);
session = new TFSession (graph);
```
Your newly created graph need to get input tensors. Here is an example with a one dimensional tensor of size 2:

```csharp
var runner = session.GetRunner ();
runner.AddInput (graph ["input_placeholder_name"] [0], new float[]{ placeholder_value1, placeholder_value2 });
```
You need to give all required inputs to the graph. There is one input per TensorFlow placeholder.

To retrieve the output of your graph run the following code. Note that this is for an output that would be a two dimensional tensor of floats. Cast to a long array if your outputs are integers.
```csharp
runner.Fetch (graph["output_placeholder_name"][0]);
float[,] recurrent_tensor = runner.Run () [0].GetValue () as float[,];
```

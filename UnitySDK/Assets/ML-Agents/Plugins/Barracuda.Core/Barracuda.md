<!---TODO:
	Advanced topics
	* worker.AddInput(): to prewarm data
	* how to trim networks at runtime (multi brain models)
	* loading model from url: var modelFromDiskOrInternet = ModelLoader.Load(url, verbose); // will download and cache model from url
	* recurrent state
--->

# Barracuda

**Barracuda** is a lightweight and **cross-platform** Neural Net **inference library for Unity**. Barracuda can execute both on GPU and CPU. Currently Barracuda is in the early development stage, so adventures are expected.

## Using Barracuda
Typically the following steps are needed to use Barracuda in application:
1. load model,
2. create inference engine (the worker),
3. execute model and
4. fetch results.

But first you have to convert your TensorFlow (or ONNX) model to Barracuda format with python scripts. Example usage:
```bash
python onnx_to_barracuda.py Models/mnist/model.onnx Destination/mnist.bytes
```
See _Converting models to Barracuda_ paragraph below for more information.

### Load Model into Barracuda
Once you have your TensorFlow (or ONNX) model converted, you can load resulting Barracuda file via `ModelLoader`:
```C#
var model = ModelLoader.LoadFromStreamingAssets(modelName + ".bytes");
```

### Create inference engine (Worker)
Inference engine in Barracuda is called Worker. Worker is responsible for converting model into executable tasks and scheduling them on GPU or CPU.
```C#
var worker = BarracudaWorkerFactory.CreateWorker(BarracudaWorkerFactory.Type.ComputeFast, model)
```

### Execute the model
Inputs can be provided both as sole `Tensor` object (assuming Model has only one input) or as a dictionary of name and `Tensor` pairs.

```C#
var inputs = new Dictionary<string, Tensor>();
inputs[name1] = new Tensor(...);
inputs[name2] = new Tensor(...);
worker.Execute(inputs);
```
Execution is asynchronous for GPU backends. Currently implementation is synchronous for CPU backends, however it is good to assume that execution will be async for all backends in the future.

### Fetch outputs
If model has only single output, then simple `worker.Fetch()` can be used, otherwise output names should be provided.
```C#
var O = worker.Fetch(outputName);
```

### Cleanup
As a Barracuda client you are responsible to `Dispose` _worker_, _inputs_ and _outputs_ you fetched. This is necessary to properly free GPU resources.
```C#
O.Dispose();
worker.Dispose();
```

## Working with data

### Tensor
Barracuda stores data in `batch`,`height`,`width`,`channels` also known as _NHWC_ or _channels-last_ format. You can interact with `Tensor` data via multi-dimensional array operators:
```C#
var tensor = new Tensor(batchCount, height, width, channelCount);
tensor[n, y, x, c] = 1.0f; // as N batches of 3 dimensional data: N x {X, Y, C}
tensor[n,       c] = 2.0f; // as N batches of 1 dimensional data: N x {C}
tensor[         i] = 3.0f; // as flat array
```

There are number of `Tensor` constructors that cover variety of scenarios. By default tensors are initialized with `0` upon construction, unless intialization `Array` is provided.
```C#
tensor = new Tensor(batchCount, height, width, channelCount);    // batch of 3 dimensional data, 0 initialized: batchCount x {height, width, channelCount}
tensor = new Tensor(batchCount, elementCount);                   // batch of 1 dimensional data, 0 initialized: batchCount x {elementCount}

var stridedArray = new float[batchCount * elementCount] { ... };
tensor = new Tensor(batchCount, elementCount, stridedArray);     // batch of 1 dimensional data, initialized from strided array

var jaggedArray = new float[batchCount][elementCount] { ... };
tensor = new Tensor(batchCount, elementCount, jaggedArray);      // batch of 1 dimensional data, initialized from jagged array

Texture2D texture = ...;
tensor = new Tensor(texture);                                    // tensor initialized with texture data: 1 x { texture.width, texture.height, 3}
```

You can query shape of the `Tensor` object, but you can not change it. Shape of the `Tensor` is immutable. If you want to have different shape of `Tensor`, you have to construct the new instance of `Tensor` object.
```C#
var shape = tensor.shape;
Debug.Log(shape + " or " + shape.batch + shape.height + shape.width + shape.channels);
```

### Texture as input
You can directly pass `Texture2D`, `Texture2DArray`, `Texture3D` or `RenderTexture` to Barracuda without accessing individual pixels on CPU:
```C#
var channelCount = 3; // you can treat input pixels as 1 (grayscale), 3 (color) or 4 (color with alpha) channels
var tensor = new Tensor(texture, channelCount);
```
You can batch multiple textures into the single `Tensor` object:
```C#
var textures = new [] { texture0, texture1, texture2, texture3 }; // these textures will form a batch
var tensor = new Tensor(textures, channelCount);
```
Note that to form a batch all textures must have the same width and height dimensions.

### Texture as output
If you want to use Barracuda execution results further in the graphics pipeline, you can copy data from `Tensor` into `RenderTexture` without stalling CPU or GPU:
```C#
	var tensor = worker.Fetch();
	var texture = BarracudaTextureUtils.TensorToRenderTexture(tensor);
```
If you wish, you can reuse the same `RenderTexture` multiple times:
```C#
	var texture = new RenderTexture(width, height, 0);
	// ...
	var tensor = worker.Fetch();
	BarracudaTextureUtils.TensorToRenderTexture(tensor, texture);
```

## Introspecting Barracuda models
Barracuda model has very simple memory representation. Once model is loaded you can query for inputs and outputs:
```C#
string[] inputNames = model.inputs;   // query model inputs
string[] outputNames = model.outputs; // query model outputs
```
Or you can directly iterate through the layers and investigate what model is going to do:
```C#
foreach (var layer in model.layers)
	Debug.Log(layer.name + " does " + layer.type);
```

## Verbose mode
You can turn on verbose mode for different parts of Barracuda:
```C#
bool verbose = true;
var model = ModelLoader.LoadFromStreamingAssets(modelName + ".bytes", verbose); // verbose loader
var worker = BarracudaWorkerFactory.CreateWorker(BarracudaWorkerFactory.Type.ComputeFast, model, verbose); // verbose execution
```

## Converting TensorFlow and ONNX models to Barracuda format
Barracuda comes with dedicated python scripts to convert pre-trained TensorFlow and ONNX models to Barracuda format.

Convert from TensorFlow:
```bash
python tensorflow_to_barracuda.py Models/3DBall-tf-model.pb Destination/3DBall-bc.bytes
```

Convert from ONNX:
```bash
python onnx_to_barracuda.py Models/mnist/model.onnx Destination/mnist-bc.bytes
```

If network has multiple outputs, but you need only particular ones during the inference, there is an optional `-trim` flag to remove unused outputs and calculations.
For example:
```bash
python tensorflow_to_barracuda.py Models/3DBall-tf-model.pb Destination/3DBall-bc.bytes -trim action$
```
Trim will first remove outputs that do not match regular expression from the graph. In this case only output that ends with `action` will be left.
Next trim will strip all nodes that do not participate in the evaluation of the output.


P.S. Python 3.5 or 3.6 is recommended

P.P.S. We plan to migrate Tensorflow and ONNX converters from Python to C# in the future.


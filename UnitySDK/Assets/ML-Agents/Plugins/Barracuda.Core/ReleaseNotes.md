# Release notes

## 0.2.4
- Switched to 2018.4.3f1 as primary Unity version for testing.
- Fixed ScaleBias scheduling issue with large amounts of data (reproduced with MobileNet @ 16 batch)
- Fixed buffer overrun in ThreadGroup SharedMemory when TRANSPOSE_X and/or SHIFTED_X paths are enabled. This should fix GPU worker issues on Windows.
- Added string cache to minimise string concat generated GC pressure.
- Added small fixes for temp memory allocations, saves ~200B per layer.
- Refactored inner loop workings, to avoid GC allocations for delegates.
- Fixed input handling for layers, now inputs are not regenerated with every execution. Static model tensors are to stay forever until worker is disposed.
- Bumped Burst version to 1.1.1.

## 0.2.3
- Rewritten Dense, Conv and some other ops on GPU. Speedup of 33% in most models with batch=1 and over 100% for batch=16.
- Optimizations: reimplemented InstanceNormalization using pyramid approach for calculating mean and variance.

## 0.2.2
- Added support for --print-supported-ops flag for model converters, now it will print approximate list of supported operations. List of supported ops depends on converter.
- Added Keras converter as part of distribution.
- Now compute shaders are loaded only if GPU worker is requested.
- Fixed bug in MaxPool and AvgPool padding. Issue discovered by Yolo faces network.
- Fixed bug in Transpose convolution support for C# backend.
- Fixed TF model conversion with two LSTM cells.
- Fixed case when strided slice end overflows to zero and thus producing negative range.

## 0.2.1
- TF importer: fixed ResizeNearestNeighbor aka Upsample2D scaling factor detection.
- TF importer: optimized node sorting. Should be faster than 0.2.0.
- TF importer: made detection of actual output node from LSTM/GRU pattern more bullet proof by skipping Const nodes.
- TF importer: improved InstanceNormalization handling.
- TF importer: fixed SquareDifference pattern.
- TF importer: fixed Conv2DBackpropInput (transpose convolution) import.
- Fixed Conv2D performance regression on some GPUs.
- Fixed TextureAsTensorData.Download() to work properly with InterpretDepthAs.Channels.
- Fixed bug when identity/nop layers would reuse input as an output and later causing premature release of that tensor as part of intermediate data cleanup.
- Added scale + bias to TenstorToRenderTexture interface, usefull for adjusting network output scale + bias on the fly.
- Fixed double Dispose issue when worker gets garbage collected.

## 0.2.0
- Version bumped to 0.2.0 as it brings breaking API changes, for details look below.
- Significantly reduced temporary memory allocations by introducing internal allocator support. Now memory is re-used between layer execution as much as possible.
- Improved small workload performance on CSharp backend
- Added parallel implementation for multiple activation functions on CSharp backend
- Added `Peek()` function to `IWorker`, it retains object storage in worker's allocator, useful for quick grabbing of output. If you want to preserve content of output tensor between `Execute()` invocations, then use `Fetch()`.
- Fixed ESRGAN model conversion (ONNX importer).
- Fixed Tensor <-> Texture copy for textures/tensors that dimensions are not multiple of 8.
- Added `Summary()` method to `Worker`. Currently returns allocator information.
- Tabs to spaces! Aiming at higher salary (https://stackoverflow.blog/2017/06/15/developers-use-spaces-make-money-use-tabs/).
- Renamed worker type enum members: `CSharp` -> `CSharpRef`, `CSharpFast` -> `CSharp`, `Compute` -> `ComputeRef`, `ComputeFast` -> `Compute`.
- Implemented new optimized `ComputePrecompiled` worker. This worker caches Compute kernels and state beforehand to reduce CPU overhead.
- Added `ExecuteAsync()` to `IWorker` interface, it returns `IEnumerator`, which enables you to control how many layers to schedule per frame (one iteration == one layer).
- Added `Log` op support on Compute workers.
- Optimized activation functions and ScaleBias by accessing tensor as continuous array. Gained ~2.0ms on 4 batch MobileNet (MBP2016).
- Introduced _Loop version of activations to fight 65535 scheduling limit on D3D11.
- Added .nn as Barracuda model file extension for use in Unity Editor. Also added simple editor importer. Now you can declare serializable fields as NNModel to bind them to .nn asset. ModelLoader.Load() now accepts NNModel as a source.
- Compute: Reduce reference GPU implementation.
- TF importer: Expanded Mean support to mean over channels, implemented Pad (as Border2D), implemented SquaredDifference, added InstanceNormalization and LeakyRelu patterns, StridedSlice implementation.
- TF importer: sort model nodes by dependencies before processing.
- Fixed ComputeBuffer leak when using Compute and ComputePrecompiled backends.
- Made to use Conv2D_L1Cached64_RegisterBlock4x4 more often: improves perf ~2x on Vega 16, and ~30% on Nvidia and Intel.

## 0.1.6
- Added activation type print in verbose mode
- Added fast and parallel CPU implementation for Swish, Relu, Add, Sub, Div, Min, Max, Tanh, Exp
- Removed duplicate profiler blocks for ops
- Improved scheduling on CPU for small batches of data
- Fixed compatibility with Unity 2019.2.x

## 0.1.5
- Added Transpose, MatMul and Indentity layer support for models exported from ONNX.
- Added BasicLSTM layer support for models exported from TF. Limited set of LSTM networks should work now.
- Added DepthwiseConv2D layer support. Most of the networks based on the MobileNet should work now.
- Added OneHot layer support for models exported from TF.
- Added optimized path for Conv2D, Dense and Transpose layers with single batch executions. Performance gain up to 100%.
- Fixed FMA performance issue on Metal GFX platforms.
- Added fast optimized path for Sigmoid and Mul layers on CPU.
- Fixed issue when worker is executed with different batch sizes.
- Added ``pip`` requirements file for Python dependencies, check ``Tools/requirements.txt```.
- Added proof of concept Docker wrappers for running model conversion inside of Docker container. Check ``Tools/docker-tensorflow-to-barracuda.sh`` and ``Tools/docker-onnx-to-barracuda.sh``. Currently it was tested only on Mac host.
- Refactored model importers for easier integration with ML Agents.
- Fixed input shape determination for Keras sequential model.
- Added metadata about input shapes to model. Look for ``Model.GetShapeByName()``.
- Added API to query constant Tensors embedded into network, look for ``Model.GetTensorByName()``.
- Added reference implementations for Selu, Abs, Neg, Ceil, Floor, Clip, Rcp, Log layers.
- Added support for Mean, Square, StridedSlice and Border2D layers.
- Added support for Swish activation, now it is automatically detected in models.
- Fixed Tanh NaN issue when large argument is passed.
- RandomNormal and RandomUniform now supports either embedded shape constant OR previous tensor shape for input.
- Fixed Keras/TF/ONNX FusedBatchNorm/BatchNorm import and now it takes ``epsilon`` into account.
- Now Barracuda will fallback to CSharpFast if compute shaders are not supported on the current platform.
- Improved compute kernel interop on Android.
- Implemented Pix2Pix model (.pict) importer.

## 0.1.4
- Implemented fast Conv2DTrans. Useful for GAN type networks.
- Fixed few ComputeBuffer handling issues.
- Simplified way to pass texture via ``Tensor`` constructor.
- Documentation improvements.
- Added Unity Companion License as part of distribution.
- Fixed boundary checks for Compute Copy/Concat operations.
- Improved profiling experience, now each layer will be reported separately in Unity Profiler.
- Fixed Broadcast layer support in ``ModelAnalyzer``.
- Exp, Pow and other layers are now also implemented in Compute. Improves RL model inference performance on GPU.
- Added platform specific BLAS plugin support. Out of the box Barracuda ships with Apple Accelerate framework support for iOS and macOS.
- Added Burst BLAS plugin, greatly improves performance in Unity Editor where native OS BLAS is not available. It's packaged as separate package and requires to have Burst enabled.
- Improved memory handling, now less GC allocations should be made per inference execution.

## 0.1.3
- Improved Barracuda support for Unity Profiler.
- Cleaned up Barracuda APIs.
- Added direct ``Texture`` input support. Look for ``TextureAsTensorData``. The following types of texture supported as input: ``Texture2D``, ``Texture2DArray``, ``Texture3D``, ``RenderTexture``.
- Added ``Tensor`` to ``RenderTexture`` conversion. Look for ``TensorToRenderTexture``.
- Autoencoder type networks can run completely on GPU now. Data roundtrip via CPU is not necessary anymore.
- Vertical flip is applied when converting between ``Texture`` and ``Tensor`` to match conventionts. To override this behavior look for ``TextureAsTensorData.Flip`` enum.
- Removed direct reference to WebCamTexture, now Barracuda compiles for Console targets.
- Fixed _Conv2DTranspose_ layer support. Now GANs using _Conv2DTranspose_ work properly.
- Added automated test for pix2pix GAN.

## 0.1.2
- Barracuda now is also available as preview package. Look for ``com.unity.barracuda`` in https://staging-packages.unity.com registry.
- Conv2D layers are now *up to 30x faster* with ``CSharpFast`` backend (``ComputeFast`` remains best backend for convolutional networks).
- Added profiler sample for ``Fetch()``.
- Fixed compilation issues on Xbox One.
- TexConv2D support was temporary disabled.
- Barracuda logging now can be configured via static fields of ``Barracuda.D`` class, it allows both disable specific logging levels or just disable stack trace collection (helps with performance when profiling).
- Compute Concat implementation now will fall back to C# implementation instead of throwing exception when unsupported configuration is encountered.
- Fixed several ``ComputeBuffer`` release issues.
- Added constructor for ``Tensor`` that allows to pass in data array.
- Improved Flatten handling in TensorFlow models.
- Added helper func ``ModelLoader.LoadFromStreamingAssets``.
- Fixed .meta file packaging.
- Small docs improvements.
- Fixed unnecessary patching of Activation layers in ``ModelLoader``.
- Added output trimming at run-time. See for extra parameters Worker factory.

## 0.1.1
- First internal realease as drop-in package
- Compatibility with ML Agents models: 3DBall, PushBlock, GridWorld, Soccer.

## 0.1.0
- First internal build. Due some bugs encountered wasn't published.

#Contributors
- Renaldas (ReJ) Zioma
- Mantas Puida
- Vladimir Oster
- Aurimas Petrovas
- Martin Sternevald
- Valdemar Bučilko
- Kuba Cupisz
- Povilas Kanapickas
- Paulius Puodžiūnas
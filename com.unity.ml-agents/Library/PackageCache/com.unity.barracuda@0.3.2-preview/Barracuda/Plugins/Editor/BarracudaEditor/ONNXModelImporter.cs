using Google.Protobuf;
using Onnx;
using UnityEngine;
using UnityEngine.Profiling;
using UnityEditor;
using UnityEditor.Experimental.AssetImporters;
using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

[assembly: InternalsVisibleToAttribute("Barracuda.EditorTests")]

namespace Barracuda
{
    /// <summary>
    /// Asset Importer of ONNX models, this will convert to barracuda NN model.
    /// </summary>
    [ScriptedImporter(2, new[] { "onnx" })]
    public class ONNXModelImporter : ScriptedImporter {
        // Configuration
        public bool patchReshapeToSupportBatchSize = true;

        private const string iconName = "ONNXModelIcon";
        private Texture2D iconTexture;

        private readonly Dictionary<string, Action<ModelBuilder, ONNXNodeWrapper>> nodeImporters =
            new Dictionary<string, Action<ModelBuilder, ONNXNodeWrapper>>();
        private void Add(string opType, Action<ModelBuilder, ONNXNodeWrapper> opImportAction)
        {
            nodeImporters.Add(opType, opImportAction);
        }

        public struct ONNXTensor
        {
            public Tensor data;
            public long[] shape;
            public int rank { get { return shape.Length; } }
        }

        // Description of the output of the layer
        public class VariableTensor // TODO: struct
        {
            public int features;
            public int rank;
        }

        private Dictionary<string, ONNXTensor> constantTensors =
            new Dictionary<string, ONNXTensor>();

        private Dictionary<string, VariableTensor> variableTensors =
            new Dictionary<string, VariableTensor>();

        public ONNXModelImporter()
        {
            // TODO: simplify code to avoid passing node.Name over and over again
            Add("Constant", (net, node) => {
                node.UnsupportedAttribute("sparse_value");
                constantTensors[node.Name] = node.ValueAsTensor;
            });
            Add("Reshape", (net, node)  => {
                long[] onnxShape;
                if (node.InputCount > 1) // Reshape-5
                    onnxShape = node.Input1Constant(onnxLayout:"C").readonlyArray.Select(v => (long)v).ToArray();
                else // Reshape-1
                    onnxShape = node.Shape;

                if (node.IsInput0Const)
                { // reshape constant source tensor and store it as the new constant
                    var symbolicShape = ConvertSymbolicShapeToBarracuda(onnxShape, "?");
                    var srcTensor = constantTensors[node.Input0].data;
                    var reshapedTensor = srcTensor.Reshape(symbolicShape);
                    Const(node.Name, reshapedTensor, onnxShape);
                }
                else
                {
                    var symbolicShape = ConvertSymbolicShapeToBarracuda(onnxShape, "NCHW");
                    if (patchReshapeToSupportBatchSize)
                        symbolicShape[0] = 0; // force keep batch size
                    net.Reshape(node.Name, node.Input0, symbolicShape);
                    Output(node, rank:symbolicShape.Length);
                }
            });
            Add("Unsqueeze", (net, node) => {
                if (!node.IsInput0Const)
                    throw new OnnxLayerImportException(
                        $"Currently only constant inputs for node of type {node.OperatorType} are supported. Instead input of {node.Name} is pointing to non constant node {node.Input0}.");

                var onnxShape = constantTensors[node.Input0].shape.ToList();
                foreach (var axis in node.Axes)
                    onnxShape.Insert(axis, 1);
                // reshape constant source tensor and store it as the new constant
                var symbolicShape = ConvertSymbolicShapeToBarracuda(onnxShape.ToArray(), "?");
                var srcTensor = constantTensors[node.Input0].data;
                var reshapedTensor = srcTensor.Reshape(symbolicShape);
                D.Log($"barracudaIN: {srcTensor.shape} barracudaOUT: {reshapedTensor.shape} onnxShapeIN: {string.Join(",", constantTensors[node.Input0].shape)} onnxShapeOUT: {string.Join(",", onnxShape)}");
                Const(node.Name, reshapedTensor, onnxShape.ToArray());

                //var outputRank = node.Input0Rank + node.Axes.Length;
                //if (node.Input0Rank > 1)
                //    Warn($"Unsqeeze on tensors with rank higher than 1 might lead to unexpected results. Got tensor with rank {node.Input0Rank}");
                //if (outputRank > 4)
                //    Warn($"Result of unsqeeze will produce tensor with rank {node.Input0Rank} which is higher than allowed.");
                //node.UnsupportedAttribute("axes", v => v < 3, new int[0]);
                //if (!node.IsInput0Const)
                //{
                //    net.Identity(node.Name, node.Input0);
                //    Output(node, rank:outputRank);
                //}
                //else
                //    Const(node.Name, constantTensors[node.Input0]);
            });
            Add("Squeeze", (net, node) => {
                if (!node.IsInput0Const)
                    throw new OnnxLayerImportException(
                        $"Currently only constant inputs for node of type {node.OperatorType} are supported. Instead input of {node.Name} is pointing to non constant node {node.Input0}.");

                var onnxShape = constantTensors[node.Input0].shape.ToList();
                foreach (var axis in node.Axes)
                    onnxShape.RemoveAt(axis);
                // reshape constant source tensor and store it as the new constant
                var symbolicShape = ConvertSymbolicShapeToBarracuda(onnxShape.ToArray(), "?");
                var srcTensor = constantTensors[node.Input0].data;
                var reshapedTensor = srcTensor.Reshape(symbolicShape);
                Const(node.Name, reshapedTensor, onnxShape.ToArray());
            });
            Add("Flatten", (net, node)  => {
                node.UnsupportedAttribute("axis", 1);
                net.Flatten(node.Name, node.Input0);
                Output(node, rank:2);
            });
            Add("Concat", (net, node)  => {
                node.UnsupportedAttribute("axis", 1);
                net.Concat(node.Name, node.Inputs, axis:-1);
                // TODO: only axis=='channels'
                var featuresConcatenated = node.Inputs.Sum(i => variableTensors[i].features);
                Output(node, features:featuresConcatenated);
            });
            Add("Slice", (net, node) => {
                if (node.IsInput0Const)
                    throw new OnnxLayerImportException(
                        $"Currently only non-constant inputs for node of type {node.OperatorType} are supported. Instead input of {node.Name} is pointing to constant node {node.Input0}.");

                int[] starts, ends, axes, steps;
                if (node.InputCount > 1) // Slice-10
                {
                    var constStarts      = node.Input1Constant(onnxLayout:"C");
                    var constEnds        = node.Input2Constant(onnxLayout:"C");
                    var defaultAxes = new Tensor(constStarts.shape, Enumerable.Range(0, constStarts.length).Select(v => (float)v).ToArray());
                    var constAxes        = node.Input3ConstantOptional(defaultAxes, onnxLayout:"C");
                    var constSteps       = node.Input4ConstantOptional(constStarts.shape, 1.0f, onnxLayout:"C");

                    starts  = constStarts.readonlyArray.Select(v => (int)v).ToArray();
                    ends    = constEnds.readonlyArray.Select(v => (int)v).ToArray();
                    axes    = constAxes.readonlyArray.Select(v => (int)v).ToArray();
                    steps   = constSteps.readonlyArray.Select(v => (int)v).ToArray();
                }
                else // Slice-1
                {
                    starts      = node.Starts;
                    ends        = node.Ends;
                    axes        = node.AxesOptional(Enumerable.Range(0, starts.Length).ToArray());
                    steps       = Enumerable.Repeat(1, starts.Length).ToArray();
                }

                Debug.Assert(starts.Length == ends.Length);
                var onnxRank    = node.Input0Rank;
                var onnxStarts  = Enumerable.Repeat(0L, onnxRank).ToArray();
                var onnxEnds    = Enumerable.Repeat(0L, onnxRank).ToArray(); // by default copy the whole axis till the end, see the notes below
                var onnxSteps   = Enumerable.Repeat(1L, onnxRank).ToArray();

                // NOTE: begin=0, end=0, stride=1  <=  full range from existing axis
                //       begin=0, end=X, stride=1  <=  full range from existing axis, if X==last element on this axis
                //       begin=0, end=0, stride=0  <=  new axis OR shrink axis to single 1st element
                //       begin=N, end=N, stride=0  <=              shrink axis to single Nth element
                // These notes are copied from TensorExtensions.ApplyStridedSlice(...)

                for (int i = 0; i < axes.Length; ++i)
                {
                    var axis = axes[i];
                    if (axis < 0)
                        axis += onnxRank;
                    axis = Math.Min(Math.Max(axis, 0), onnxRank);
                    onnxStarts[axis] = starts[i];
                    onnxEnds[axis]   = ends[i];
                    onnxSteps[axis]  = steps[i];
                }

                net.StridedSlice(node.Name, node.Input0,
                    starts:ConvertSymbolicShapeToBarracuda(onnxStarts, onnxLayout:"NCHW"),
                    ends:ConvertSymbolicShapeToBarracuda(onnxEnds, onnxLayout:"NCHW"),
                    strides:ConvertSymbolicShapeToBarracuda(onnxSteps, onnxLayout:"NCHW"));
            });

            // Activation ops
            Add("Relu", (net, node)     => { net.Relu(node.Name, node.Input0); });
            Add("Softmax", (net, node)  => { net.Softmax(node.Name, node.Input0); node.UnsupportedAttribute("axis", 1); });
            Add("Tanh", (net, node)     => { net.Tanh(node.Name, node.Input0); });
            Add("Sqrt", (net, node)     => { net.Sqrt(node.Name, node.Input0); });
            Add("Sigmoid", (net, node)  => { net.Sigmoid(node.Name, node.Input0); });
            Add("Elu", (net, node)      => { net.Elu(node.Name, node.Input0, node.AlphaOptional(1f)); });
            Add("LeakyRelu",(net, node) => { net.LeakyRelu(node.Name, node.Input0, node.AlphaOptional(0.01f)); });
            Add("Selu", (net, node)     => { net.Selu(node.Name, node.Input0, node.AlphaOptional(1.67326f), node.GammaOptional(1.0507f)); });
            Add("Swish", (net, node)    => { net.Swish(node.Name, node.Input0); });
            Add("PRelu", (net, node)    => { net.PRelu(node.Name, node.Input0, node.Input1); });
            Add("LogSoftmax", (net, node)   => { net.LogSoftmax(node.Name, node.Input0); node.UnsupportedAttribute("axis", 1); });
            Add("Exp", (net, node)    => { net.Exp(node.Name, node.Input0); });
            Add("Log", (net, node)    => { net.Log(node.Name, node.Input0); });
            Add("Neg", (net, node)    => { net.Neg(node.Name, node.Input0); });
            Add("Reciprocal", (net, node)    => { net.Reciprocal(node.Name, node.Input0); });
            // TODO: Add("Hardmax", (net, node)      => { net.Hardmax(node.Name, node.Input0); node.UnsupportedAttribute("axis", 1); });
            // TODO: Add("Softplus", (net, node)     => { net.Softplus(node.Name, node.Input0); });
            // TODO: Add("Softsign", (net, node)     => { net.Softsign(node.Name, node.Input0); });
            // TODO: Add("HardSigmoid", (net, node)  => { net.HardSigmoid(node.Name, node.Input0, node.AlphaOptional(0.2f), node.BetaOptional(0.5f)); });
            Add("Clip", (net, node)    => { net.Clip(node.Name, node.Input0, node.MinOptional(float.MinValue),  node.MaxOptional(float.MaxValue)); });

            // Broadcast ops
            Add("Add", (net, node)     => { net.Add(node.Name, node.Inputs); });
            Add("Sum", (net, node)     => { net.Add(node.Name, node.Inputs); }); // Sum is implemented via Add
            Add("Sub", (net, node)     => { net.Sub(node.Name, node.Inputs); });
            Add("Mul", (net, node)     => { net.Mul(node.Name, node.Inputs); });
            Add("Div", (net, node)     => { net.Div(node.Name, node.Inputs); });
            Add("Pow", (net, node)     => { net.Pow(node.Name, node.Inputs); });
            Add("Min", (net, node)     => { net.Min(node.Name, node.Inputs); });
            Add("Max", (net, node)     => { net.Max(node.Name, node.Inputs); });
            Add("Mean", (net, node)    => { net.Mean(node.Name, node.Inputs); });

            // Logical ops
            Add("Greater", (net, node) => { net.Greater(node.Name, node.Input0, node.Input1); });
            Add("Less", (net, node)    => { net.Less(node.Name, node.Input0, node.Input1); });
            Add("Equal", (net, node)   => { net.Equal(node.Name, node.Input0, node.Input1); });
            Add("Or", (net, node)      => { net.LogicalOr(node.Name, node.Input0, node.Input1); });
            Add("And", (net, node)     => { net.LogicalAnd(node.Name, node.Input0, node.Input1); });
            Add("Not", (net, node)     => { net.LogicalNot(node.Name, node.Input0); });
            Add("Xor", (net, node)     => { net.LogicalXor(node.Name, node.Input0, node.Input1); });

            // Pooling
            Add("Pad", (net, node) =>
            {
                var mode = node.GetOptionalString("mode", "constant");
                switch (mode)
                {
                    case "constant": net.Border2D(node.Name, node.Input0, node.Pads, node.GetOptionalFloat("value", 0.0f)); break;
                    case "reflect": net.Pad2DReflect(node.Name, node.Input0, node.Pads); break;
                    case "edge": net.Pad2DEdge(node.Name, node.Input0, node.Pads); break;
                }
            });

            // Pooling ops
            Add("AveragePool", (net, node) => {
                node.UnsupportedAttribute("ceil_mode", 0);
                node.UnsupportedAttribute("count_include_pad", 0);
                net.AvgPool2D(node.Name, node.Input0, node.KernelShape, node.Strides, node.Pads);
            });
            Add("MaxPool", (net, node) => {
                node.UnsupportedAttribute("ceil_mode", 0);
                node.UnsupportedAttribute("dilations", new[] {1, 1});
                node.UnsupportedAttribute("storage_order", 0);
                net.MaxPool2D(node.Name, node.Input0, node.KernelShape, node.Strides, node.Pads);
            });
            Add("GlobalAveragePool", (net, node) => { net.GlobalAvgPool2D(node.Name, node.Input0); });
            Add("GlobalMaxPool",     (net, node) => { net.GlobalMaxPool2D(node.Name, node.Input0); });
            Add("Upsample", (net, node) => {
                node.UnsupportedAttribute("mode", "nearest");
                // TODO: if (node.Scales < 1) pool

                float[] scales;
                if (node.InputCount > 1)
                {
                    scales = node.Input1Constant(onnxLayout:"C").readonlyArray;
                    if (scales.Length < 2 || scales.Length > 5)
                        throw new OnnxLayerImportException(
                            $"Input scales of unsupported length {scales.Length} in {node.Name} ot fype {node.OperatorType}.");

                    if ((scales[0] != 1) || (scales[1] != 1))
                        Warn(net, node, $"Unsupported scaling, only H and W scaling are supported. Value will be ignored and defaulted to 1.");

                    // skip NC from onnx NCHW layout
                    scales = scales.Skip(2).ToArray();
                }
                else
                    scales = node.Scales;

                if (!scales.All(x => Mathf.Approximately(x, Mathf.Round(x))))
                    Warn(net, node, $"Only integer scale values are currently supported. Scale value will be rounded to closest integer value.");

                net.Upsample2D(node.Name, node.Input0, scales.Select(x => (int)Mathf.Round(x)).ToArray());
            });

            // Tensor ops
            Add("Gemm", (net, node)     => {
                node.UnsupportedAttribute("alpha", 1.0f);
                node.UnsupportedAttribute("beta", 1.0f);
                node.UnsupportedAttribute("transA", 0);
                var onnxLayout = node.TransBOptional() ? "KC" : "CK";
                var weights = node.Input1Constant(onnxLayout);
                var biases  = node.Input2ConstantOptional(Bias(weights.shape), 0.0f, onnxLayout:"C");
                // Change data layout from "channels first" to "channels last"
                weights = SwapSpatialDimensionsAndFeaturesInMatMulWeights(weights, node.Input0Features);
                net.Dense(node.Name, node.Input0, weights, biases);
                Output(node, features:weights.channels, rank:2); // Gemm forces flatten of the input to rank 2
            });
            Add("MatMul", (net, node)   => {
                var weights = node.Input1Constant(onnxLayout:"CK");
                var biases  = node.DefaultTensor(Bias(weights.shape), 0.0f);
                // Change data layout from "channels first" to "channels last"
                weights = SwapSpatialDimensionsAndFeaturesInMatMulWeights(weights, node.Input0Features);
                net.Dense(node.Name, node.Input0, weights, biases);
                Output(node, features:weights.channels, rank:2); // MatMul forces flatten of the input to rank 2
            });
            Add("Conv", (net, node)     => {
                node.UnsupportedAttribute("dilations", new[] {1, 1});
                node.IgnoredAttribute("kernel_shape", "Kernel shape is derived from K tensor weights instead");
                var kernels = node.Input1Constant(onnxLayout:"KCHW");
                var biases  = node.Input2ConstantOptional(Bias(kernels.shape), 0.0f, onnxLayout:"C");

                if (node.GroupOptional() > 1)
                    net.DepthwiseConv2D(node.Name, node.Input0, node.Strides, node.Pads, kernels, biases);
                else
                    net.Conv2D(node.Name, node.Input0, node.Strides, node.Pads, kernels, biases);
                Output(node, features:kernels.channels);
            });
            Add("ConvTranspose", (net, node)     => {
                node.UnsupportedAttribute("dilations", new[] {1, 1});
                node.UnsupportedAttribute("group", 1);
                node.UnsupportedAttribute("output_shape", new int[0]);
                node.IgnoredAttribute("kernel_shape", "Kernel shape is derived from K tensor weights instead");
                var kernels = node.Input1Constant(onnxLayout:"CKHW");
                var biases  = node.Input2ConstantOptional(Bias(kernels.shape), 0.0f, onnxLayout:"C");
                net.Conv2DTrans(node.Name, node.Input0, node.Strides, node.Pads, node.OutputPadding, kernels, biases);
                Output(node, features:kernels.channels);
            });
            Add("BatchNormalization", (net, node) => {
                var variance  = node.Input4Constant(onnxLayout:"C");
                var scale     = node.Input1ConstantOptional(variance.shape, 1.0f, onnxLayout:"C");
                var bias      = node.Input2ConstantOptional(variance.shape, 0.0f, onnxLayout:"C");
                var mean      = node.Input3ConstantOptional(variance.shape, 0.0f, onnxLayout:"C");
                var fusedData = FuseBatchNormWeights(scale, bias, mean, variance, node.EpsilonOptional());
                net.ScaleBias(node.Name, node.Input0, fusedData.Item1, fusedData.Item2);
            });
            Add("InstanceNormalization", (net, node) => {
                var scale     = node.Input1Constant(onnxLayout:"C");
                var bias      = node.Input2ConstantOptional(scale.shape, 0.0f, onnxLayout:"C");
                net.Normalization(node.Name, node.Input0, scale, bias, node.EpsilonOptional());
            });
            // random opps
            Add("RandomNormal", (net, node) => {
                float mean     = node.GetOptionalFloat("mean",  1.0f);
                float scale    = node.GetOptionalFloat("scale",   0.0f);
                float seed     = node.GetOptionalFloat("seed",  0.0f);
                int[] shape    = node.GetRequiredIntArray("shape");
                net.RandomNormal(node.Name, mean, scale, seed, shape);
            });
            Add("RandomNormalLike", (net, node) => {
                float mean     = node.GetOptionalFloat("mean",  1.0f);
                float scale    = node.GetOptionalFloat("scale",   0.0f);
                float seed     = node.GetOptionalFloat("seed",  0.0f);
                net.RandomNormal(node.Name, mean, scale, seed, node.Input0);
            });
            Add("RandomUniform", (net, node) => {
                float high     = node.GetOptionalFloat("high",  1.0f);
                float low      = node.GetOptionalFloat("low",   0.0f);
                float seed     = node.GetOptionalFloat("seed",  0.0f);
                int[] shape    = node.GetRequiredIntArray("shape");
                net.RandomUniform(node.Name, low, high, seed, shape);
            });
            Add("RandomUniformLike", (net, node) => {
                float high     = node.GetOptionalFloat("high",  1.0f);
                float low      = node.GetOptionalFloat("low",   0.0f);
                float seed     = node.GetOptionalFloat("seed",  0.0f);
                net.RandomUniform(node.Name, low, high, seed, node.Input0);
            });

            // Ignore, noop during inference
            Add("Identity", (net, node)     => { net.Identity(node.Name, node.Input0); });
            Add("Cast", (net, node)         => { net.Identity(node.Name, node.Input0); });
            Add("Dropout", (net, node)      => { net.Identity(node.Name, node.Input0); });
        }

        private void Const(string name, Tensor data, long[] onnxShape)
        {
            Const(name, new ONNXTensor { data = data, shape = onnxShape });
        }
        private void Const(string name, ONNXTensor onnxTensor)
        {
            constantTensors[name] = onnxTensor;
            Output(name, onnxTensor);
        }

        private void Output(ONNXNodeWrapper node, int features = -1, int rank = -1)
        {
            if (!variableTensors.ContainsKey(node.Name))
                variableTensors[node.Name] = new VariableTensor();
            variableTensors[node.Name].features = features;
            variableTensors[node.Name].rank = rank;
        }
        private void Output(string nodeId, ONNXTensor onnxTensor)
        {
            if (!variableTensors.ContainsKey(nodeId))
                variableTensors[nodeId] = new VariableTensor();
            variableTensors[nodeId].features = -1;
            variableTensors[nodeId].rank = onnxTensor.rank;
        }
        private void Output(string nodeId, long[] onnxShape, string onnxLayout)
        {
            if (!variableTensors.ContainsKey(nodeId))
                variableTensors[nodeId] = new VariableTensor();
            var onnxRank = onnxShape.Length;
            var permuatations = AxisPermutationsForMappingONNXLayoutToBarracuda(onnxRank, onnxLayout);
            var barracudaChannelIndex = permuatations.Length - 1;
            var onnxChannelIndex = permuatations[barracudaChannelIndex];
            var channels = (onnxLayout != "?" && onnxChannelIndex >= 0) ? (int)onnxShape[onnxChannelIndex]: -1;
            variableTensors[nodeId].features = channels;
            variableTensors[nodeId].rank = onnxRank;
        }

        internal static int[] AxisPermutationsForMappingONNXLayoutToBarracuda(int onnxRank, string onnxLayout="NCHW")
        {
            // Input tensors:           NCHW -> NHWC, NCW -> N1WC, NC -> N11C, C -> 111C
            // Conv kernels:            KCHW -> HWCK, KCW -> 1WCK
            // TransposeConv:           CKHW -> HWCK, CKW -> 1WCK
            // Gemm, MatMul weights:    KC   -> C11K

            const int _ = -1;

            if (onnxRank == 0)
                return new[] {_, _, _, _};

            if (onnxRank > 4)
                throw new OnnxLayerImportException($"Only tensors of rank 4 or less are supported, but got rank {onnxRank}");

            else if (onnxLayout == "NCHW") // -> NHWC
                switch (onnxRank)
                {
                    case 4:
                        return new int[] {0, 2, 3, 1};
                    case 3:
                        return new int[] {0, _, 2, 1};
                    case 2:
                        return new int[] {0, _, _, 1};
                    case 1:
                        return new int[] {_, _, _, 0};
                }
            else if (onnxLayout == "CONST") // -> NHWC
                switch (onnxRank)
                {
                    case 4:
                        return new int[] {0, 2, 3, 1}; // assume NCHW
                    case 3:
                        return new int[] {_, 2, 1, 0}; // assume  CHW
                    case 2:
                        return new int[] {_, _, 1, 0}; // assume   CW
                    case 1:
                        return new int[] {_, _, _, 0}; // assume    C
                }
            else if (onnxLayout == "MCHW" || onnxLayout == "KCHW") // -> HWCK
                switch (onnxRank)
                {
                    case 4:
                        return new int[] {2, 3, 1, 0};
                    case 3:
                        return new int[] {_, 2, 1, 0};
                    case 2:
                    case 1:
                        throw new OnnxLayerImportException($"MCHW layout requires kernel weight tensor of rank 3 or higher, but got {onnxRank}");
                }
            else if (onnxLayout == "CMHW" || onnxLayout == "CKHW") // -> HWCK
                switch (onnxRank)
                {
                    case 4:
                        return new int[] {2, 3, 0, 1};
                    case 3:
                        return new int[] {_, 2, 0, 1};
                    case 1:
                        throw new OnnxLayerImportException($"CMHW layout requires kernel weight tensor of rank 3 or higher, but got {onnxRank}");
                }
            else if (onnxLayout == "CHWM" || onnxLayout == "CHWK") // -> HWCK
                switch (onnxRank)
                {
                    case 4:
                        return new int[] {1, 2, 0, 3};
                    case 3:
                        return new int[] {_, 1, 0, 2};
                    case 1:
                        throw new OnnxLayerImportException($"CHWM layout requires kernel weight tensor of rank 3 or higher, but got {onnxRank}");
                }
            else if (onnxLayout == "CM" || onnxLayout == "CK") // -> C__K
                switch (onnxRank)
                {
                    case 2:
                        return new int[] {0, _, _, 1};
                    case 4:
                    case 3:
                    case 1:
                        throw new OnnxLayerImportException($"CM layout requires weight tensor of rank 2, but got {onnxRank}");
                }
            else if (onnxLayout == "MC" || onnxLayout == "KC") // -> C__K
                switch (onnxRank)
                {
                    case 2:
                        return new int[] {1, _, _, 0};
                    case 4:
                    case 3:
                    case 1:
                        throw new OnnxLayerImportException($"MC layout requires weight tensor of rank 2, but got {onnxRank}");
                }
            else if (onnxLayout == "C") // -> ___C
                switch (onnxRank)
                {
                    case 1:
                        return new int[] {_, _, _, 0};
                    case 4:
                    case 3:
                    case 2:
                        throw new OnnxLayerImportException($"C layout requires tensor of rank 1, but got {onnxRank}");
                }
            else if (onnxLayout == "?")
                switch (onnxRank)
                {
                    case 4:
                        return new int[] {0, 1, 2, 3};
                    case 3:
                        return new int[] {0, 1, 2, _};
                    case 2:
                        return new int[] {0, 1, _, _};
                    case 1:
                        return new int[] {0, _, _, _};
                }
            else
                throw new OnnxLayerImportException($"Unknown tensor layout {onnxLayout}");

            throw new OnnxLayerImportException($"Unsupported combination of tensor layout {onnxLayout} and tensor rank {onnxRank}");
        }

        // Fuse training time BatchNorm tensors into Scale & Bias
        internal static Tuple<Tensor, Tensor> FuseBatchNormWeights(Tensor gamma, Tensor beta, Tensor mean, Tensor variance, float epsilon)
        {
            // https://github.com/Tencent/ncnn/blob/master/src/layer/batchnorm.cpp
            // float sqrt_var = sqrt(var_data[i]);
            // a_data[i] = bias_data[i] - slope_data[i] * mean_data[i] / sqrt_var;
            // b_data[i] = slope_data[i] / sqrt_var;
            // ...
            // ptr[i] = b * ptr[i] + a;
            Debug.Assert(gamma.shape == beta.shape);
            Debug.Assert(gamma.shape == mean.shape);
            Debug.Assert(gamma.shape == variance.shape);
            Tensor scale = new Tensor(gamma.shape);
            Tensor bias = new Tensor(gamma.shape);
            for (int i = 0; i < gamma.length; ++i)
            {
                scale[i] = gamma[i] / Mathf.Sqrt(variance[i] + epsilon);
                bias[i] = beta[i] - gamma[i] * mean[i] / Mathf.Sqrt(variance[i] + epsilon);
            }
            return Tuple.Create(scale, bias);
        }

        // Transpose channels first to channels last data in MatMul/GEMM weight tensor
        internal static Tensor SwapSpatialDimensionsAndFeaturesInMatMulWeights(Tensor weights, int featureCount)
        {
            Debug.Assert(featureCount <= weights.flatHeight);
            if (featureCount != weights.flatHeight)
            {
                var shape = weights.shape;
                var implicitSpatialDimensionsInWeights = shape.flatHeight / featureCount;
                Debug.Assert(shape.flatHeight % featureCount == 0);
                // reshape: C__K -> CHWK
                weights = weights.Reshape(
                    new TensorShape(featureCount, implicitSpatialDimensionsInWeights, 1, shape.channels));
                // permute: CHWK -> HWCK
                weights = Permute(weights, new int[] {1,0,2,3});
                // reshape: HWCK -> C__K
                weights = weights.Reshape(shape);
            }
            return weights;
        }

        internal static TensorShape Bias(TensorShape shape)
        {
            return new TensorShape(1, 1, 1, shape.channels);
        }

        public override void OnImportAsset(AssetImportContext ctx)
        {
            var onnxModel = new ModelProto();
            using (var readStream = new FileStream(ctx.assetPath, FileMode.Open))
            using (var inputStream = new CodedInputStream(readStream))
                onnxModel.MergeFrom(inputStream);

            var irVersion = onnxModel.IrVersion; // legacy
            if (onnxModel.OpsetImport?.Count > 0)
                irVersion = onnxModel.OpsetImport[0].Version;

            var producerName = $"{onnxModel.ProducerName} v{onnxModel.ProducerVersion}";
            var model = ConvertOnnxModel(onnxModel);
            model.ProducerName = producerName;
            model.IrSource = "ONNX";
            model.IrVersion = $"{irVersion}";
            D.Log($"ONNX v{irVersion}. File producer: {producerName}.");
            D.Log($"Barracuda model: {model}");

            var asset = ScriptableObject.CreateInstance<NNModel>();
            using (var memoryStream = new MemoryStream())
            using (var writer = new BinaryWriter(memoryStream))
            {
                ModelWriter.Save(writer, model);
                asset.Value = memoryStream.ToArray();
            }

            ctx.AddObjectToAsset("main obj", asset, LoadIconTexture());
            ctx.SetMainObject(asset);
        }

        private Model ConvertOnnxModel(ModelProto onnxModel)
        {
            var model = new Model();
            var modelBuilder = new ModelBuilder(model);

            // Convert graph inputs & outputs
            var initializersByName = onnxModel.Graph.Initializer.ToDictionary(i => i.Name, i => true);
            foreach (ValueInfoProto i in onnxModel.Graph.Input)
            {
                // skip input tensors that have initializer data, they are constant tensors not global inputs
                if (initializersByName.ContainsKey(i.Name))
                    continue;

                modelBuilder.Input(i.Name, ConvertSymbolicShapeToBarracuda(i.Type.TensorType.Shape, onnxLayout:"NCHW"));
                Output(i.Name, onnxShape:i.Type.TensorType.Shape.Dim.Select(d => d.DimValue).ToArray(), onnxLayout:"NCHW");
            }
            foreach (ValueInfoProto o in onnxModel.Graph.Output)
                modelBuilder.Output(o.Name);

            // TODO: process model (recurrent nodes) memories

            // Read constants from initializer list
            foreach (TensorProto initializer in onnxModel.Graph.Initializer)
                Const(initializer.Name, ReadTensor(initializer));

            // Convert graph nodes
            foreach (NodeProto onnxNode in onnxModel.Graph.Node)
            {
                var node = new ONNXNodeWrapper(onnxNode, onnxModel, constantTensors, variableTensors, model.Warnings);
                var nodeId = node.Name;
                var opType = node.OperatorType;

                Output(node);

                bool injectDummy = false;
                if (nodeImporters.ContainsKey(opType))
                {
                    try
                    {
                        Profiler.BeginSample($"Import {opType} {node.Name}");
                        nodeImporters[opType](modelBuilder, node);
                        Profiler.EndSample();
                    }
                    catch (Exception e)
                    {
                        // We support the layer but something went wrong while importing it
                        // We log the problem and insert an identity layer
                        string message = $"Unexpected error while parsing layer {nodeId} of type {opType}.\n{e.Message}\n\nJson: {onnxNode}\n";
                        Warn(model, nodeId, message);
                        injectDummy = true;
                    }
                }
                else
                {
                    //We don't support this type of layer
                    //We log the problem and insert an identity layer
                    string message = $"Unknown type encountered while parsing layer {nodeId} of type {opType}. We replace by an identity layer.";
                    Warn(model, nodeId, message);
                    injectDummy = true;
                }

                if (injectDummy)
                {
                    var originalLayerHadInputs = (node.InputCount > 0);
                    if (originalLayerHadInputs)
                        modelBuilder.Identity(nodeId, node.Input0);
                    else // if errorneous layer had no inputs, inject dummy constant which does not require any inputs
                        modelBuilder.Const(nodeId, new Tensor());
                }

                 // TODO: extract into separate function
                if (variableTensors[nodeId].features == -1)
                {
                    // TODO: reuse Output code for constants to determine features when onnxLayout is known
                    if (variableTensors.ContainsKey(node.Input0Optional))
                        variableTensors[nodeId].features = variableTensors[node.Input0Optional].features;
                }
                if (variableTensors[nodeId].rank == -1)
                {
                    if (constantTensors.ContainsKey(nodeId))
                        variableTensors[nodeId].rank = constantTensors[nodeId].rank;
                    else if (variableTensors.ContainsKey(node.Input0Optional))
                        variableTensors[nodeId].rank = variableTensors[node.Input0Optional].rank;
                }
            }

            // Convert constant tensors
            int insertionIndex = 0;
            var unconnectedInputs = ModelAnalyzer.FindBrokenLinks(model);
            foreach(var entry in constantTensors)
                if (unconnectedInputs.Any(i => i == entry.Key))
                    modelBuilder.Const(entry.Key,
                        ConvertTensorToBarracuda(entry.Value, onnxLayout:"CONST"),
                        insertionIndex++);

            // model should not contain any broken links in the end
            unconnectedInputs = ModelAnalyzer.FindBrokenLinks(model);
            Debug.Assert(unconnectedInputs.Length == 0);
            if (unconnectedInputs.Length > 0)
            {
                var message = $"Broken links: {string.Join(", ", unconnectedInputs)}";
                Warn(model, "", message);
            }

            return model;
        }

        private Texture2D LoadIconTexture()
        {
            if (iconTexture == null)
            {
                string[] allCandidates = AssetDatabase.FindAssets(iconName);

                if (allCandidates.Length > 0)
                {
                    iconTexture = AssetDatabase.LoadAssetAtPath(AssetDatabase.GUIDToAssetPath(allCandidates[0]), typeof(Texture2D)) as Texture2D;
                }
            }
            return iconTexture;
        }

        private class OnnxLayerImportException : Exception
        {
            public OnnxLayerImportException(string message) : base(message) { }
        }

        private static int[] Permute(long[] shape, string onnxLayout)
        {
            var onnxRank = shape.Length;
            var permutations = AxisPermutationsForMappingONNXLayoutToBarracuda(onnxRank, onnxLayout);
            Debug.Assert(shape.Length <= permutations.Length);
            Debug.Assert(shape.Length == permutations.Count(v => v >= 0));
            var output = new int[permutations.Length];
            for (var i = 0; i < permutations.Length; ++i)
                output[i] = permutations[i] >= 0 ? (int)shape[permutations[i]] : 1;
            return output;
        }

        private static int[] Permute(int[] shape, int[] permutations)
        {
            Debug.Assert(shape.Length <= permutations.Length);
            Debug.Assert(shape.Count(v => v > 1) <= permutations.Count(v => v >= 0));
            var output = new int[permutations.Length];
            for (var i = 0; i < permutations.Length; ++i)
                output[i] = permutations[i] >= 0 ? (int)shape[permutations[i]] : 1;
            return output;
        }

        // slow version - kept just for performance comparison and validation
        internal static Tensor PermuteSlow(Tensor readTensor, int[] permutations) // TODO: unify Permute() arguments
        {
            var outputTensor = new Tensor(Permute(readTensor.shape.ToArray(), permutations));
            Debug.Assert(outputTensor.length == readTensor.length);

            var inShape = readTensor.shape.ToArray();
            for (var n = 0; n < inShape[0]; ++n)
                for (var h = 0; h < inShape[1]; ++h)
                    for (var w = 0; w < inShape[2]; ++w)
                        for (var c = 0; c < inShape[3]; ++c)
                        {
                            var it = new int[] {0, n, h, w, c}; // prepend with 0 to handle "new axis" -1 value in permutations
                            var oN = it[permutations[0] + 1];
                            var oH = it[permutations[1] + 1];
                            var oW = it[permutations[2] + 1];
                            var oC = it[permutations[3] + 1];
                            outputTensor[oN, oH, oW, oC] = readTensor[n, h, w, c];
                        }

            return outputTensor;
        }

        internal static Tensor Permute(Tensor inTensor, int[] permutations) // TODO: unify Permute() arguments
        {
            // See: https://stackoverflow.com/a/32034565

            Profiler.BeginSample("PermuteTensorData");
            var outTensor = new Tensor(Permute(inTensor.shape.ToArray(), permutations));
            Debug.Assert(outTensor.length == inTensor.length);

            // {0, 2, 3, 1} => {0, 3, 1, 2}
            // {2, 3, 1, 0} => {3, 2, 0, 1}
            //              => {find_index(0), find_index(1), find_index(2), find_index(3)}
            var reversePermute = new int[permutations.Length];
            for (var i = 0; i < permutations.Length; ++i)
                reversePermute[i] = Array.IndexOf(permutations, i);

            // outTensor strides
            var outStrideC   =               outTensor.channels;
            var outStrideWC  = outStrideC  * outTensor.width;
            var outStrideHWC = outStrideWC * outTensor.height;

            var outStride = new int[reversePermute.Length];
            for (var i = 0; i < reversePermute.Length; ++i)
                outStride[i] = new[] {0, outStrideHWC, outStrideWC, outStrideC, 1}[reversePermute[i] + 1];

            // inTensor strides
            var inStrideC   =              inTensor.channels;
            var inStrideWC  = inStrideC  * inTensor.width;
            var inStrideHWC = inStrideWC * inTensor.height;

            var inShape = inTensor.shape.ToArray();
            for (var n = 0; n < inShape[0]; ++n)
                for (var h = 0; h < inShape[1]; ++h)
                    for (var w = 0; w < inShape[2]; ++w)
                        for (var c = 0; c < inShape[3]; ++c)
                        {
                            outTensor[n * outStride[0] + h * outStride[1] + w * outStride[2] + c * outStride[3]] =
                                inTensor[n * inStrideHWC + h * inStrideWC + w * inStrideC + c];
                        }

            Profiler.EndSample();
            return outTensor;
/*
zyx     => xzy
4x24x12 => 12x4x24

z*24*12 y*24 x = x*4*24 z*24 y = z*24 y x*4*24
z*24*12 y*24 x = z*s[0] y*s[1] x*s[2] :: s[24, 1, 4*24]


        "NCHW" // -> NHWC
        return new int[] {0, 2, 3, 1};     {0, 2, 3, 1} => {0, 3, 1, 2}

nchw                        =>  nhwc
xyzw                        =>  xzwy
1x4x12x24                       1x12x24x4
n*4x12x24 c*12x24 h*24 w*1  =   n*12x24x4 h*24x4 w*4 c*1
x*4x12x24 y*12x24 z*24 w*1  =   x*12x24x4 z*24x4 w*4 y*1 = xyzw*[12x24x4, 1, 24x4, 4]   [0, 3, 1, 2]

z::4
y::24
x::12
*/
        }

        // TODO: extract common Permute(Tensor) code
        private static Tensor ConvertTensorToBarracuda(ONNXTensor onnxTensor, string onnxLayout)
        {
            Profiler.BeginSample("ConvertTensorToBarracuda");
            if (onnxLayout == "?")
                throw new OnnxLayerImportException("Unknown ONNX layout in not supported when converting constant tensor to Barracuda");

            var onnxShape = onnxTensor.shape;
            Debug.Assert(onnxShape.All(v => v > 0));

            var onnxRank = onnxShape.Length;
            var permutations = AxisPermutationsForMappingONNXLayoutToBarracuda(onnxRank, onnxLayout);
            Debug.Assert(onnxShape.Length <= permutations.Length);
            Debug.Assert(onnxShape.Length == permutations.Count(v => v >= 0));

            var outTensor = Permute(onnxTensor.data, permutations);
            Profiler.EndSample();
            return outTensor;
        }

        private static TensorShape ConvertShapeToBarracuda(long[] onnxShape, string onnxLayout)
        {
            var shape = ConvertSymbolicShapeToBarracuda(onnxShape, onnxLayout);
            if (shape.Any(s => s <= 0))
                throw new OnnxLayerImportException($"Expected ONNX shape with all dimensions known, instead got {string.Join(", ",shape)}");
            return new TensorShape(shape);
        }

        private static int[] ConvertSymbolicShapeToBarracuda(TensorShapeProto shape, string onnxLayout)
        {
            // TODO: use dimension denotation from TensorShapeProto to figure, if this particular tensor has specific data layout
            // https://github.com/onnx/onnx/blob/master/docs/DimensionDenotation.md
            return ConvertSymbolicShapeToBarracuda(shape.Dim.Select(d => d.DimValue).ToArray(), onnxLayout);
        }

        private static int[] ConvertSymbolicShapeToBarracuda(long[] onnxShape, string onnxLayout)
        {
            var permutedShape = Permute(onnxShape, onnxLayout);
            Debug.Assert(permutedShape.Length == 4);
            return Enumerable.Repeat(1, 4 - permutedShape.Length).Concat(permutedShape).ToArray();
        }

        private static ONNXTensor ReadTensor(TensorProto onnxTensor)
        {
            // shape
            var onnxShape = onnxTensor.Dims.ToArray();
            var shape = ConvertShapeToBarracuda(onnxShape, onnxLayout:"?");

            // data
            float[] data;
            if ((onnxTensor.RawData != null) && (!onnxTensor.RawData.IsEmpty))
            {
                var byteArray = new byte[onnxTensor.RawData.Length];
                onnxTensor.RawData.CopyTo(byteArray, 0);

                if (onnxTensor.DataType == (int)TensorProto.Types.DataType.Float)
                {
                    data = new float[shape.length];
                    Debug.Assert((sizeof(float) * shape.length) == onnxTensor.RawData.Length);
                    Buffer.BlockCopy(byteArray, 0, data, 0, byteArray.Length);
                }
                else if (onnxTensor.DataType == (int)TensorProto.Types.DataType.Int32)
                {
                    var typedData = new int[shape.length];
                    Debug.Assert((sizeof(int) * shape.length) == onnxTensor.RawData.Length);
                    Buffer.BlockCopy(byteArray, 0, typedData, 0, byteArray.Length);
                    data = typedData.Select(x => (float)x).ToArray();
                }
                else if (onnxTensor.DataType == (int)TensorProto.Types.DataType.Int64)
                {
                    var typedData = new long[shape.length];
                    Debug.Assert((sizeof(long) * shape.length) == onnxTensor.RawData.Length);
                    Buffer.BlockCopy(byteArray, 0, typedData, 0, byteArray.Length);
                    data = typedData.Select(x => (float)x).ToArray();
                }
                else
                    throw new OnnxLayerImportException($"Tensor data type {(TensorProto.Types.DataType)onnxTensor.DataType} is not supported.");
            }
            else if ((onnxTensor.FloatData != null) && (onnxTensor.FloatData.Count != 0))
            {
                Debug.Assert(shape.length == onnxTensor.FloatData.Count);
                data = new float[shape.length];
                onnxTensor.FloatData.CopyTo(data, 0);
            }
            else if ((onnxTensor.Int32Data != null) && (onnxTensor.Int32Data.Count != 0))
            {
                Debug.Assert(shape.length == onnxTensor.Int32Data.Count);
                data = onnxTensor.Int32Data.Select(x => (float)x).ToArray();
            }
            else if ((onnxTensor.Int64Data != null) && (onnxTensor.Int64Data.Count != 0))
            {
                Debug.Assert(shape.length == onnxTensor.Int64Data.Count);
                data = onnxTensor.Int64Data.Select(x => (float)x).ToArray();
            }
            else
            {
                throw new OnnxLayerImportException("Could not read tensor data for constant tensor.");
            }

            return new ONNXTensor { data = new Tensor(shape, new SharedArrayTensorData(data)), shape = onnxShape};
        }

        // Logging helpers
        private static void Warn(ModelBuilder builder, ONNXNodeWrapper node, string message)
        {
            Warn(builder.model, node.Name, message);
        }

        private static void Warn(Model model, string layerName, string message)
        {
            model.Warnings.Add(new Model.ImporterWarning(layerName,message));
            Debug.LogWarning(message);
        }

        private class ONNXNodeWrapper
        {
            private NodeProto m_ONNXNode;
            private ModelProto m_ONNXModel;
            private Dictionary<string, ONNXTensor> m_ConstantTensors;
            private Dictionary<string, VariableTensor> m_VariableTensors;
            private List<Model.ImporterWarning> m_ImporterWarnings;

            public ONNXNodeWrapper(NodeProto ONNXNode, ModelProto ONNXModel,
                Dictionary<string, ONNXTensor> constantTensors,
                Dictionary<string, VariableTensor> variableTensors,
                List<Model.ImporterWarning> importerWarnings)
            {
                m_ONNXNode = ONNXNode;
                m_ONNXModel = ONNXModel;
                m_ConstantTensors = constantTensors;
                m_VariableTensors = variableTensors;
                m_ImporterWarnings = importerWarnings;
            }

            // Layer identification (name and op)
            public string Name {
                get {
                    // prefer node.output over the node.name
                    return (m_ONNXNode.Output[0].Length > 0) ? m_ONNXNode.Output[0] : m_ONNXNode.Name;
                }
            }
            public string OperatorType { get { return m_ONNXNode.OpType; } }

            // Logging helpers
            public void Warn(string message)
            {
                m_ImporterWarnings.Add(new Model.ImporterWarning(Name, message));
                Debug.LogWarning(message);
            }
            public void UnsupportedAttribute(string name)
            {
                AttributeProto attr;
                if (TryFindAttribute(name, out attr))
                    Warn($"Unsupported attribute {name}, node {Name} of type {OperatorType}. Value will be ignored.");
            }
            public void UnsupportedAttribute(string name, int defaultValue)
            {
                if (GetOptionalInt(name, defaultValue) != defaultValue)
                    Warn($"Unsupported attribute {name}, node {Name} of type {OperatorType}. Value will be ignored and defaulted to {defaultValue}.");
            }
            public void UnsupportedAttribute(string name, float defaultValue)
            {
                if (GetOptionalFloat(name, defaultValue) != defaultValue)
                    Warn($"Unsupported attribute {name}, node {Name} of type {OperatorType}. Value will be ignored and defaulted to {defaultValue}.");
            }
            public void UnsupportedAttribute(string name, string defaultValue)
            {
                if (GetOptionalString(name, defaultValue) != defaultValue)
                    Warn($"Unsupported attribute {name}, node {Name} of type {OperatorType}. Value will be ignored and defaulted to {defaultValue}.");
            }
            public void UnsupportedAttribute(string name, int[] defaultValue)
            {
                var valueArray = GetOptionalIntArray(name, defaultValue);
                if (!Enumerable.SequenceEqual(valueArray, defaultValue))
                    Warn($"Unsupported attribute {name}, node {Name} of type {OperatorType}. Value will be ignored and defaulted to [{string.Join(", ", defaultValue)}].");
            }
            public void UnsupportedAttribute(string name, Func<int, bool> predicate, int[] defaultValue)
            {
                var valueArray = GetOptionalIntArray(name, defaultValue);
                if (!Enumerable.All(valueArray, predicate))
                    Warn($"Unsupported attribute {name}, node {Name} of type {OperatorType}. Value will be ignored and defaulted to [{string.Join(", ", defaultValue)}].");
            }
            public void IgnoredAttribute(string name, string reasonToIgnore)
            {
            }

            // Inputs
            internal string GetRequiredInput(int inputIndex)
            {
                if ((inputIndex >= m_ONNXNode.Input.Count) || (m_ONNXNode.Input[inputIndex] == ""))
                    throw new OnnxLayerImportException($"required Input {inputIndex} was not found.");

                return m_ONNXNode.Input[inputIndex];
            }

            internal bool IsInputConst(int inputIndex)
            {
                // TODO: optimize! Current implementation will allocate and copy constant data, if input points to constant data
                try
                {
                    ReadTensorFromModel(GetRequiredInput(inputIndex));
                    return true;
                }
                catch
                {
                    return false;
                }
            }

            internal Tensor ReadRequiredInputAsConstant(string input, string onnxLayout)
            {
                if (input == "")
                    throw new OnnxLayerImportException("Input value is marked as required, but it is missing in the model.");

                ONNXTensor onnxTensor;
                try
                {
                    onnxTensor = ReadTensorFromModel(input);
                }
                catch (KeyNotFoundException)
                {
                    throw new OnnxLayerImportException(
                        $"Currently only constant inputs for node of type {OperatorType} are supported. Instead input of {Name} is pointing to non constant node {input}.");
                }

                return ConvertTensorToBarracuda(onnxTensor, onnxLayout);
            }

            internal Tensor ReadOptionalInputAsConstant(string input, TensorShape shape, float defaultValue, string onnxLayout)
            {
                try { return ReadRequiredInputAsConstant(input, onnxLayout); } catch (Exception) { return DefaultTensor(shape, defaultValue); }
            }

            public int InputCount { get { return m_ONNXNode.Input.Count;  } }
            public string[] Inputs { get { return m_ONNXNode.Input.ToArray(); } }
            public string Input0 { get { return GetRequiredInput(0); } }
            public string Input1 { get { return GetRequiredInput(1); } }
            public string Input2 { get { return GetRequiredInput(2); } }
            public string Input3 { get { return GetRequiredInput(3); } }
            public string Input4 { get { return GetRequiredInput(4); } }
            public string Input0Optional { get { return InputCount > 0 ? GetRequiredInput(0) : ""; } }
            public string Input1Optional { get { return InputCount > 1 ? GetRequiredInput(1) : ""; } }
            public string Input2Optional { get { return InputCount > 2 ? GetRequiredInput(2) : ""; } }
            public string Input3Optional { get { return InputCount > 3 ? GetRequiredInput(3) : ""; } }
            public string Input4Optional { get { return InputCount > 4 ? GetRequiredInput(4) : ""; } }
            public bool IsInput0Const { get { return IsInputConst(0); } }
            public bool IsInput1Const { get { return IsInputConst(1); } }
            public bool IsInput2Const { get { return IsInputConst(2); } }
            public bool IsInput3Const { get { return IsInputConst(3); } }
            public bool IsInput4Const { get { return IsInputConst(4); } }

            public int Input0Features { get { return m_VariableTensors[Input0].features; } }
            public int Input0Rank { get { return m_VariableTensors[Input0].rank; } }
            public Tensor Input0Constant(string onnxLayout) { return ReadRequiredInputAsConstant(Input0, onnxLayout); }
            public Tensor Input1Constant(string onnxLayout) { return ReadRequiredInputAsConstant(Input1, onnxLayout); }
            public Tensor Input2Constant(string onnxLayout) { return ReadRequiredInputAsConstant(Input2, onnxLayout); }
            public Tensor Input3Constant(string onnxLayout) { return ReadRequiredInputAsConstant(Input3, onnxLayout); }
            public Tensor Input4Constant(string onnxLayout) { return ReadRequiredInputAsConstant(Input4, onnxLayout); }
            public Tensor Input1ConstantOptional(Tensor defaultValue, string onnxLayout) { try { return ReadRequiredInputAsConstant(Input1, onnxLayout); } catch (Exception) { return defaultValue; } }
            public Tensor Input2ConstantOptional(Tensor defaultValue, string onnxLayout) { try { return ReadRequiredInputAsConstant(Input2, onnxLayout); } catch (Exception) { return defaultValue; } }
            public Tensor Input3ConstantOptional(Tensor defaultValue, string onnxLayout) { try { return ReadRequiredInputAsConstant(Input3, onnxLayout); } catch (Exception) { return defaultValue; } }
            public Tensor Input4ConstantOptional(Tensor defaultValue, string onnxLayout) { try { return ReadRequiredInputAsConstant(Input4, onnxLayout); } catch (Exception) { return defaultValue; } }
            public Tensor Input1ConstantOptional(TensorShape shape, float defaultValue, string onnxLayout) { try { return ReadRequiredInputAsConstant(Input1, onnxLayout); } catch (Exception) { return DefaultTensor(shape, defaultValue); } }
            public Tensor Input2ConstantOptional(TensorShape shape, float defaultValue, string onnxLayout) { try { return ReadRequiredInputAsConstant(Input2, onnxLayout); } catch (Exception) { return DefaultTensor(shape, defaultValue); } }
            public Tensor Input3ConstantOptional(TensorShape shape, float defaultValue, string onnxLayout) { try { return ReadRequiredInputAsConstant(Input3, onnxLayout); } catch (Exception) { return DefaultTensor(shape, defaultValue); } }
            public Tensor Input4ConstantOptional(TensorShape shape, float defaultValue, string onnxLayout) { try { return ReadRequiredInputAsConstant(Input4, onnxLayout); } catch (Exception) { return DefaultTensor(shape, defaultValue); } }

            // Read tensor data from ONNX TensorProt
            private ONNXTensor ReadTensorFromModel(string name)
            {
                try
                {
                    return m_ConstantTensors[name];
                }
                catch (KeyNotFoundException)
                {
                    throw new KeyNotFoundException($"Could not find constant tensor {name} in the model.");
                }
            }

            public Tensor DefaultTensor(TensorShape tensorShape, float defaultValue)
            {
                var shape = tensorShape;
                var data = Enumerable.Repeat(defaultValue, tensorShape.length).ToArray();
                return new Tensor(shape, data);
            }

            //Attributes
            internal bool TryFindAttribute(string name, out AttributeProto attr)
            {
                return TryFindAttribute(name, AttributeProto.Types.AttributeType.Undefined, out attr);
            }
            internal bool TryFindAttribute(string name, AttributeProto.Types.AttributeType type, out AttributeProto attr)
            {
                const AttributeProto.Types.AttributeType undefined = AttributeProto.Types.AttributeType.Undefined;
                var attributes = m_ONNXNode.Attribute;
                for (var i = 0; i < attributes.Count; ++i)
                {
                    attr = attributes[i];
                    if (attr.Name == name && (attr.Type == type || attr.Type == undefined || type == undefined))
                        return true;
                }
                attr = null;
                return false;
            }
            internal AttributeProto FindAttribute(string name, AttributeProto.Types.AttributeType type = AttributeProto.Types.AttributeType.Undefined)
            {
                AttributeProto attr = null;
                if (TryFindAttribute(name, type, out attr))
                    return attr;

                throw new OnnxLayerImportException($"Couldn't find attribute {name} of type {type}");
            }
            public float GetOptionalFloat(string name, float defaultValue)
            {
                try { return GetRequiredFloat(name); }
                catch (OnnxLayerImportException) { return defaultValue; }
            }
            public float GetRequiredFloat(string name)
            {
                return FindAttribute(name, AttributeProto.Types.AttributeType.Float).F;
            }
            public float[] GetOptionalFloatArray(string name, float[] defaultValue)
            {
                try { return GetRequiredFloatArray(name); }
                catch (OnnxLayerImportException) { return defaultValue; }
            }
            public float[] GetRequiredFloatArray(string name)
            {
                var attribute = FindAttribute(name,AttributeProto.Types.AttributeType.Floats);
                return attribute.Floats.ToArray();//.Select(x => (float)x).ToArray();
            }
            public ONNXTensor GetRequiredTensor(string name)
            {
                var tensorProto = FindAttribute(name, AttributeProto.Types.AttributeType.Tensor).T;
                return ReadTensor(tensorProto);
            }
            public int GetOptionalInt(string name, int defaultValue)
            {
                try { return GetRequiredInt(name); }
                catch (OnnxLayerImportException) { return defaultValue; }
            }
            public int GetRequiredInt(string name)
            {
                return (int)FindAttribute(name, AttributeProto.Types.AttributeType.Int).I;
            }
            public int[] GetOptionalIntArray(string name, int[] defaultValue)
            {
                try { return GetRequiredIntArray(name); }
                catch (OnnxLayerImportException) { return defaultValue; }
            }
            public int[] GetRequiredIntArray(string name)
            {
                var attribute = FindAttribute(name,AttributeProto.Types.AttributeType.Ints);
                return attribute.Ints.Select(x => (int)x).ToArray();
            }
            public long[] GetOptionalLongArray(string name, long[] defaultValue)
            {
                try { return GetRequiredLongArray(name); }
                catch (OnnxLayerImportException) { return defaultValue; }
            }
            public long[] GetRequiredLongArray(string name)
            {
                var attribute = FindAttribute(name,AttributeProto.Types.AttributeType.Ints);
                return attribute.Ints.ToArray();
            }
            public string GetOptionalString(string name, string defaultValue)
            {
                try { return GetRequiredString(name); }
                catch (OnnxLayerImportException) { return defaultValue; }
            }
            public string GetRequiredString(string name)
            {
                var raw = FindAttribute(name, AttributeProto.Types.AttributeType.String).S;
                return raw.ToStringUtf8();
            }

            public float Alpha { get { return GetRequiredFloat("alpha"); } }
            public float Beta { get { return GetRequiredFloat("beta"); } }
            public float Gamma { get { return GetRequiredFloat("gamma"); } }
            public float Epsilon { get { return GetRequiredFloat("epsilon"); } }
            public float Mean { get { return GetRequiredFloat("mean"); } }
            public float Scale { get { return GetRequiredFloat("scale"); } }
            public float Seed { get { return GetOptionalFloat("seed", 1337f); } } // seed is always optional and defaults to 'auto generated'
            public ONNXTensor ValueAsTensor { get { return GetRequiredTensor("value"); } }
            public int Axis { get { return GetRequiredInt("axis"); } }
            public int Group { get { return GetRequiredInt("group"); } }
            public long[] Shape { get { return GetRequiredLongArray("shape"); } }
            public int[] Starts { get { return GetRequiredIntArray("starts"); } }
            public int[] Ends { get { return GetRequiredIntArray("ends"); } }
            public int[] Axes { get { return GetRequiredIntArray("axes"); } }
            public int[] KernelShape { get { return GetRequiredIntArray("kernel_shape"); } }
            public int[] Strides { get { return GetOptionalIntArray("strides", new[] {1,1}); } }
            public int[] OutputPadding { get { return GetOptionalIntArray("output_padding", new[] {0,0}); } }
            internal bool SupportsAutoPad { get { return OperatorType != "Pad"; } }
            internal bool SupportsSpatialOnlyPads { get { return OperatorType != "Pad"; } }
            public int[] Pads { get {
                var noPadding = new[] {0,0,0,0};
                if (SupportsAutoPad)
                {
                    // known_paddings = {
                    //     'VALID' : [0,0,0,0],
                    //     'SAME_UPPER'  : [-1],
                    //     'SAME_LOWER'  : [-2],
                    // }
                    var autoPad = GetOptionalString("auto_pad", "NOTSET");
                    if (autoPad == "VALID")
                        return noPadding;
                    else if (autoPad == "SAME_UPPER")
                        return new[] { -1 };
                    else if (autoPad == "SAME_LOWER")
                        return new[] { -2 };
                    else {} // TODO: Assert NOTSET
                }

                // NOTE: ONNX has pad layout of [z, y, x ...] while Barracuda is opposite [x, y, z ...]
                var pads = GetOptionalIntArray("pads", noPadding);
                if (SupportsSpatialOnlyPads)
                {
                    // See: https://github.com/onnx/onnx/blob/master/docs/Operators.md#AveragePool
                    // Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0.
                    // The value represent the number of pixels added to the beginning and end part of the corresponding axis.
                    // `pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...],
                    // where xi_begin the number of pixels added at the beginning of axis `i` and xi_end,
                    // the number of pixels added at the end of axis `i`.

                    switch (pads.Length)
                    {
                        case 2: return new [] { pads[0], 0, pads[1], 0 }; // 1D WW => W_W_
                        case 4: return new [] { pads[1], pads[0], pads[3], pads[2] }; // 2D HWHW => WHWH
                        case 6: Warn("3D pads are not supported yet!");
                            return new [] { pads[2], pads[1], pads[0], pads[3], pads[4], pads[5] }; // TODO: 3D DHWDHW => WHDWHD
                        default:
                            throw new OnnxLayerImportException(
                                $"Attribute pads of unsupported length {pads.Length} in {Name} ot fype {OperatorType}.");
                    }
                }

                // See: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Pad
                // `pads` should be a 1D tensor of shape [2 * input_rank].
                // `pads` format should be: [x1_begin, x2_begin,...,x1_end, x2_end,...],
                // where xi_begin is the number of pad values added at the beginning of axis `i` and xi_end,
                // the number of pad values added at the end of axis `i

                Debug.Assert(pads.Length % 2 == 0);
                long[] onnxStarts = new long[pads.Length / 2];//TODO make the semantic diff between Permute(int[] and Permute(long[] more clear.
                long[] onnxEnds = new long[pads.Length / 2];
                for(int i=0; i < onnxStarts.Length; ++i)
                {
                    onnxStarts[i] = (long)pads[i];
                    onnxEnds[i] = (long)pads[i + onnxStarts.Length];
                }
                var starts = Permute(onnxStarts, "NCHW");
                var ends = Permute(onnxEnds, "NCHW");
                if ((starts[0] != 0) || (starts[3] != 0) || (ends[0] != 0) || (ends[3] != 0))
                    Warn($"Unsupported padding, only H and W padding are supported. Value will be ignored and defaulted to 0.");

                return new int[] { starts[2], starts[1], ends[2], ends[1] };
            } }
            public float[] Scales { get {
                var scales = GetOptionalFloatArray("scales", new float[0]);
                if (scales.Length > 0)
                    return scales;
                return new[] {
                    GetRequiredFloat("height_scale"),
                    GetRequiredFloat("width_scale")
                };
            } }
            public float AlphaOptional(float defaultValue) { return GetOptionalFloat("alpha", defaultValue); }
            public float BetaOptional(float defaultValue) { return GetOptionalFloat("beta", defaultValue); }
            public float GammaOptional(float defaultValue) { return GetOptionalFloat("gamma", defaultValue); }
            public float EpsilonOptional(float defaultValue=1e-5f) { return GetOptionalFloat("epsilon", defaultValue); }
            public float MeanOptional(float defaultValue=0f) { return GetOptionalFloat("mean", defaultValue); }
            public float ScaleOptional(float defaultValue=1f) { return GetOptionalFloat("scale", defaultValue); }
            public bool TransAOptional(bool defaultValue=false) { return GetOptionalInt("transA", defaultValue?1:0) != 0;}
            public bool TransBOptional(bool defaultValue=false) { return GetOptionalInt("transB", defaultValue?1:0) != 0;}
            public int AxisOptional(int defaultValue) { return GetOptionalInt("axis", defaultValue); }
            public int GroupOptional(int defaultValue=1) { return GetOptionalInt("group", defaultValue); }
            public int[] KernelShapeOptional(int[] defaultValue) { return GetOptionalIntArray("kernel_shape", defaultValue); }
            public int[] AxesOptional(int[] defaultValue) { return GetOptionalIntArray("axes", defaultValue); }
            public float MinOptional(float defaultValue) { return GetOptionalFloat("min", defaultValue); }
            public float MaxOptional(float defaultValue) { return GetOptionalFloat("max", defaultValue); }
        }
    }
}

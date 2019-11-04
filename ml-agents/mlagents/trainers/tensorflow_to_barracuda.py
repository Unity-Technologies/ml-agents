from __future__ import print_function
import numpy as np
import struct  # convert from Python values and C structs
import tensorflow as tf
import re

# import barracuda
# from barracuda import Struct
from mlagents.trainers import barracuda
from mlagents.trainers.barracuda import Struct
from google.protobuf import descriptor
from google.protobuf.json_format import MessageToJson


if __name__ == "__main__":
    # Handle command line argumengts
    args = barracuda.parse_args(
        description="Convert Tensorflow model to Barracuda binary",
        source_extension=".pb",
        help="input Tensorflow serialized .pb file",
    )
    # Te following code can be used as an example of API used from another module
    # convert() is the main entry point for converter
    import tensorflow_to_barracuda as tf2bc

    tf2bc.convert(args.source_file, args.target_file, args.trim_unused_by_output, args)


# TODO: support more than 1 LSTM layer per model - prepend scope to names and inputs
# TODO: support different activation functions in LSTM
# TODO: strip output Identity node, instead patch upstream layer names
# TODO: use ScaleBias and Pow with alpha when input is constant Tensor
# TODO: support all data format types (curretly only NHWC)
# TODO: support all data types (currently only FLOAT, INT32, BOOL)
# TODO: implement FusedResizeAndPadConv2D

# Important ProtoBuf definitions:
#    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/types.proto
#    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto
#    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/node_def.proto
#
# Node descriptions:
#    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/nn_ops.cc
#    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/math_ops.cc
#    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/random_ops.cc
#
# Class doc:
#    https://www.tensorflow.org/api_docs/cc/
#
known_classes = {
    "Dense": Struct(
        id=1,
        rank=2,
        out_shapes=lambda shapes: [
            [shapes[0][0], 1, 1, shapes[0][1]]
            if len(shapes[0]) > 1
            else [1, 1, 1, 1],  # W
            [1, 1, 1, shapes[-1][-1]],  # B
        ],
        patch_data=lambda data: [data[0], data[1]],
    ),
    "MatMul": Struct(
        id=1,
        rank=2,
        out_shapes=lambda shapes: [
            [shapes[0][0], 1, 1, shapes[0][1]],  # W
            [1, 1, 1, shapes[0][1]],  # B
        ],
        patch_data=lambda data: [data[0], np.zeros(np.shape(data[1]))],
    ),
    "BiasAdd": Struct(
        id=51,  # implemented as ScaleBias
        out_shapes=lambda shapes: [
            [1, 1, 1, shapes[0][0]],  # ONE
            [1, 1, 1, shapes[0][0]],  # B
        ],
        patch_data=lambda data: [np.ones(np.shape(data[0])), data[0]],
    ),
    # TODO: NCHW
    "Conv2D": Struct(
        id=20,
        rank=4,
        out_shapes=lambda shapes: [shapes[0], [1, 1, 1, shapes[-1][-1]]],  # K  # B
        patch_data=lambda data: [data[0], data[1]],
    ),
    "DepthwiseConv2dNative": Struct(  # DepthwiseConv2D
        id=21,
        rank=4,
        out_shapes=lambda s: [
            [
                s[0][0],
                s[0][1],
                s[0][3],
                s[0][2],
            ],  # K TF:[H, W, in_channels, channel_multiplier] => [H, W, 1, in_channels]
            [1, 1, 1, s[-1][-1]] if len(s) > 1 else [1, 1, 1, s[0][2]],  # B
        ],
        patch_data=lambda data: [np.transpose(data[0], (0, 1, 3, 2)), data[1]],
    ),
    "Conv2DBackpropInput": Struct(  # Conv2DTranspose
        id=22,
        rank=4,
        out_shapes=lambda s: [
            [
                s[0][0],
                s[0][1],
                s[0][3],
                s[0][2],
            ],  # K TF:[H, W, in_channels, out_channels] => [H, W, out_channels, in_channels]
            [1, 1, 1, s[-1][-1]] if len(s) > 1 else [1, 1, 1, s[0][2]],  # B
        ],
        patch_data=lambda data: [np.transpose(data[0], (0, 1, 3, 2)), data[1]],
    ),
    "Pad": 29,
    # TODO: 3D
    "ResizeNearestNeighbor": 23,  # implemented as Upsample2D
    "ResizeBilinear": 23,  # implemented as Upsample2D
    "ResizeBicubic": 23,  # implemented as Upsample2D
    "MaxPool": 25,
    "AvgPool": 26,
    "GlobalAveragePool": 28,
    "GlobalAvgPool": 28,
    "Activation": 50,
    "BatchNormalization": Struct(
        id=51,  # after fusion implemented as ScaleBias
        out_shapes=lambda shapes: [
            [1, 1, 1, shapes[0][0]],  # S
            [1, 1, 1, shapes[0][0]],  # B
        ],
        patch_data=lambda data:
        # fuse [gamma, beta, mean, var, epsilon] => [scale, bias]
        # TODO: double-check if epsilon is the last data argument and not the 1st?
        barracuda.fuse_batchnorm_weights(data[0], data[1], data[2], data[3], data[4])
        if len(data) == 5
        else
        # fuse [ONE, beta, mean, var, epsilon] => [scale, bias]
        # TODO: double-check if epsilon is the last data argument and not the 1st?
        barracuda.fuse_batchnorm_weights(
            np.ones(np.shape(data[0])), data[0], data[1], data[2], data[3]
        ),
    ),
    "FusedBatchNorm": Struct(
        id=51,  # after fusion implemented as ScaleBias
        out_shapes=lambda shapes: [
            [1, 1, 1, shapes[0][0]],  # S
            [1, 1, 1, shapes[0][0]],  # B
        ],
        patch_data=lambda data, layer:
        # fuse [gamma, beta, mean, var, epsilon] => [scale, bias]
        barracuda.fuse_batchnorm_weights(
            data[0], data[1], data[2], data[3], get_epsilon(layer)
        ),
    ),
    "BatchNormalizationRuntime": Struct(
        id=52,
        out_shapes=lambda shapes: [
            [1, 1, 1, shapes[0][0]],  # G
            [1, 1, 1, shapes[0][0]],  # B
        ],
        patch_data=lambda data: [data[0], data[1]]
        if len(data) == 4
        else [np.ones(np.shape(data[0])), data[0]],
    ),
    "InstanceNormalization": Struct(  # TODO: epsilon
        id=52,
        out_shapes=lambda shapes: [
            [1, 1, 1, shapes[0][0]],  # G
            [1, 1, 1, shapes[0][0]],  # B
        ],
        patch_data=lambda data: [data[0], data[1]]
        if len(data) == 2
        else [np.ones(np.shape(data[0])), data[0]],
    ),
    "LRN": 53,
    "RandomStandardNormal": 64,
    "RandomUniform": 65,
    "Multinomial": Struct(id=66, rank=2),
    "OneHot": Struct(id=67, rank=lambda inputs: inputs[0] + 1),
    # Broadcast ops
    "Add": Struct(id=100, rank=lambda inputs: np.max(inputs)),
    "AddV2": Struct(id=100, rank=lambda inputs: np.max(inputs)),
    "Sub": Struct(id=101, rank=lambda inputs: np.max(inputs)),
    "Mul": Struct(id=102, rank=lambda inputs: np.max(inputs)),
    "RealDiv": Struct(id=103, rank=lambda inputs: np.max(inputs)),
    "Pow": Struct(id=104, rank=lambda inputs: np.max(inputs)),
    "Minimum": Struct(id=110, rank=lambda inputs: np.max(inputs)),
    "Maximum": Struct(id=111, rank=lambda inputs: np.max(inputs)),
    # Reduce ops
    "Max": Struct(id=124, rank=lambda inputs: inputs[0] - 1),
    "Mean": Struct(id=125, rank=lambda inputs: inputs[0] - 1),
    "Min": Struct(id=126, rank=lambda inputs: inputs[0] - 1),
    "Prod": Struct(id=127, rank=lambda inputs: inputs[0] - 1),
    "Sum": Struct(id=128, rank=lambda inputs: inputs[0] - 1),
    "Flatten": Struct(id=200, rank=2),
    "Reshape": 201,
    "Concat": 210,
    "StridedSlice": 211,
    "Nop": 0,
}

requires_runtime_flag = {
    "Dropout": "DropoutRuntime",
    "BatchNormalization": "BatchNormalizationRuntime",
}

known_activations = {
    "Linear": 0,
    "Relu": 1,
    "Softmax": 2,
    "Tanh": 3,
    "Sigmoid": 4,
    "Elu": 5,
    "Relu6": 6,
    "LeakyRelu": 7,
    "Selu": 8,
    "Swish": 9,
    "LogSoftmax": 10,
    "Softplus": 11,
    "Softsign": 12,
    "Abs": 100,
    "Neg": 101,
    "Ceil": 102,
    "Floor": 104,
    "Sqrt": 111,
    "Exp": 113,
    "Log": 114,
    "Acos": 200,
    "Acosh": 201,
    "Asin": 202,
    "Asinh": 203,
    "Atan": 204,
    "Atanh": 205,
    "Cos": 206,
    "Cosh": 207,
    "Sin": 208,
    "Sinh": 209,
    "Tan": 210,
}

known_paddings = {"VALID": [0, 0, 0, 0], "SAME": [-1]}  # SameUpper

supported_data_formats = {"NHWC"}

known_patterns = {
    # TODO: Flatten pattern using namespace regexp
    repr(["Shape", "StridedSlice", "Pack", "Reshape"]): "Flatten",
    repr(["Shape", "StridedSlice", "Prod", "Pack", "Reshape"]): "Flatten",
    repr(
        ["Shape", "Slice", "Slice", "Prod", "ExpandDims", "ConcatV2", "Reshape"]
    ): "Flatten",
    repr(["Add", "Rsqrt", "Mul", "Mul", "Sub", "Add"]): "BatchNormalization",
    repr(["Add", "Rsqrt", "Mul", "Mul", "Mul", "Sub", "Add"]): "BatchNormalization",
    repr(
        [
            "Mean",
            "StopGradient",
            "SquaredDifference",
            "Mean",
            "Sub",
            "Add",
            "Pow",
            "RealDiv",
            "Mul",
            "Add",
        ]
    ): "InstanceNormalization_ByTensorOrder",
    repr(
        [
            "Mean",
            "StopGradient",
            "SquaredDifference",
            "Mean",
            "Squeeze",
            "Squeeze",
            "Add",
            "Rsqrt",
            "Mul",
            "Mul",
            "Mul",
            "Sub",
            "Add",
        ]
    ): "InstanceNormalization_ByTensorName",
    repr(["MatMul", "BiasAdd"]): "Dense",
    repr(["Conv2D", "BiasAdd"]): "Conv2D",
    repr(["DepthwiseConv2dNative", "BiasAdd"]): "DepthwiseConv2dNative",
    repr(["Conv2DBackpropInput", "BiasAdd"]): "Conv2DBackpropInput",
    repr(["Conv2DBackpropInput"]): "Conv2DBackpropInput",
    repr(
        [
            "Shape",
            "StridedSlice",
            "StridedSlice",
            "StridedSlice",
            "Mul",
            "Mul",
            "Pack",
            "Conv2DBackpropInput",
            "BiasAdd",
        ]
    ): "Conv2DBackpropInput",
    repr(
        [
            "Shape",
            "StridedSlice",
            "StridedSlice",
            "StridedSlice",
            "Mul",
            "Mul",
            "Pack",
            "Conv2DBackpropInput",
        ]
    ): "Conv2DBackpropInput",
    repr(
        ["Shape", "StridedSlice", "Mul", "ResizeNearestNeighbor"]
    ): "ResizeNearestNeighbor",
    repr(
        ["Pack", "Reshape"]
    ): "Flatten$",  # for now we assume that this combination is trivial Flatten
    # for exmaple it is used in ML-agents LSTM nets with sequence_length==1
    repr(
        [
            "StridedSlice",
            "Reshape",
            re.compile("^lstm/"),
            "Reshape",
            "ConcatV2",
            "Identity",
        ]
    ): "BasicLSTMReshapeOut",
    repr(
        [re.compile("^lstm/"), "Reshape", "ConcatV2", "Identity"]
    ): "BasicLSTMReshapeOut",
    repr(
        ["Reshape", re.compile("^lstm_[a-z]*/"), "Reshape", "ConcatV2"]
    ): "BasicLSTMReshapeOut",
    repr(["Reshape", re.compile("^lstm_[a-z]*/"), "ConcatV2"]): "BasicLSTMConcatOut",
    repr(["Sigmoid", "Mul"]): "Swish",
    repr(["Mul", "Abs", "Mul", "Add"]): "LeakyRelu",
    repr(
        ["Shape", "Reshape"]
    ): "ReshapeLikeInput0",  # shape comes from the 1st node as input[0]
    repr(["Reshape"]): "Reshape",
    repr(["ConcatV2"]): "ConcatV2",
    repr(["Mean"]): "Mean",
    repr(["Pad"]): "Pad",
    repr(["Multinomial"]): "Multinomial",
    repr(["OneHot"]): "OneHot",
    repr(["Square"]): "Square",
    repr(["SquaredDifference"]): "SquaredDifference",
    repr(["StridedSlice"]): "StridedSlice",
    repr(["Squeeze"]): "Squeeze",
    repr(["ExpandDims"]): "ExpandDims",
    # TODO: FusedResizeAndPadConv2D
}


def by_name(args, name):
    for a in args:
        if a.name.endswith(name):
            return a


def by_op(args, op):
    for a in args:
        if a.op == op:
            return a


def order_by(args, names):
    ordered = []
    arg_count = len(args)
    for name in names:
        ordered += [a for a in args if a.endswith(name)]
        args = [a for a in args if not a.endswith(name)]
    ordered += args  # append what is left
    assert len(ordered) == arg_count
    return ordered


transform_patterns = {
    "Flatten": lambda nodes, inputs, tensors, _: Struct(op="Flatten", input=inputs),
    "Flatten$": lambda nodes, inputs, tensors, _: Struct(
        op="Flatten",
        input=[
            inputs[-1]
        ],  # take only the last input, assume all other arguments are trivial (like sequence_length==1
        # always in ML-agents LSTM nets)
    ),
    "Reshape": lambda nodes, inputs, tensors, context: Struct(
        op="Reshape",
        rank=len(tensors[0].data)
        if len(tensors)
        > 0  # tensor data is treated as reshape coefficient, if not empty
        else context.layer_ranks[inputs[1]]
        if len(inputs) == 2  # otherwise shape of the 2nd input tensor is used
        else -1,
        input=inputs,
        shape=[
            tensors[0].data[0],
            tensors[0].data[1],
            tensors[0].data[2],
            tensors[0].data[3],
        ]
        if len(tensors) > 0 and len(tensors[0].data) == 4
        else [tensors[0].data[0], 1, tensors[0].data[1], tensors[0].data[2]]
        if len(tensors) > 0 and len(tensors[0].data) == 3
        else [tensors[0].data[0], 1, 1, tensors[0].data[1]]
        if len(tensors) > 0 and len(tensors[0].data) == 2
        else [1, 1, 1, tensors[0].data[0]]
        if len(tensors) > 0 and len(tensors[0].data) == 1
        else [],
    ),
    "ReshapeLikeInput0": lambda nodes, inputs, tensors, context: Struct(
        op="Reshape",
        rank=context.layer_ranks[inputs[0]]
        if len(inputs)
        == 2  # unlike standard 'Reshape' input[0] is used as shape & input[1] as data
        else -1,
        input=[inputs[1], inputs[0]]
        if len(inputs)
        == 2  # unlike standard 'Reshape' input[0] is used as shape & input[1] as data
        else inputs,
    ),
    "Pad": lambda nodes, inputs, tensors, _: Struct(
        op="Pad"
        if (
            len(tensors) > 0
            and np.shape(tensors[0]) == [4, 2]
            and get_attr(nodes[-1], "mode", default="constant").lower() == "constant"
        )
        else "BarracudaUnsupportedPad",
        input=inputs,
        pads=[
            tensors[0].data[1, 0],
            tensors[0].data[1, 1],
            tensors[0].data[2, 0],
            tensors[0].data[2, 1],
        ]
        if len(tensors) > 0 and np.shape(tensors[0]) == [4, 2]
        else [0, 0, 0, 0],
        beta=get_attr(nodes[-1], "constant_values") or 0,
    ),
    "Squeeze": lambda nodes, inputs, tensors, context: Struct(
        op="Nop",  # Squeeze is no-operation in Barracuda
        input=inputs,
        rank=context.layer_ranks[inputs[0]] - len(get_attr(nodes[-1], "squeeze_dims"))
        if len(get_attr(nodes[-1], "squeeze_dims")) > 0
        else -1,  # if list of squeeze axis is not specified, it is unknown what would be the rank of result
    ),
    "ExpandDims": lambda nodes, inputs, tensors, context: Struct(
        op="Nop",  # ExpandDims is no-operation in Barracuda
        input=[inputs[0]],
        rank=context.layer_ranks[inputs[0]] + 1,
    ),
    "Multinomial": lambda nodes, inputs, tensors, _: Struct(
        op="Multinomial",
        input=inputs,
        shape=[int(by_name(tensors, "/num_samples").data[0])],
        # seed = get_attr(nodes[0], 'seed'),
    ),
    "OneHot": lambda nodes, inputs, tensors, _: Struct(
        op="OneHot",
        input=inputs,
        shape=[int(by_name(tensors, "/depth").data[0])],
        alpha=by_name(tensors, "/on_value").data[0],
        beta=by_name(tensors, "/off_value").data[0],
    ),
    "Square": lambda nodes, inputs, tensors, _: Struct(
        op="Mul", input=[inputs[0], inputs[0]]  # input * input
    ),
    "ConcatV2": lambda nodes, inputs, tensors, context: Struct(
        op="Concat",
        input=inputs,
        axis=axis_to_barracuda(
            int(by_name(tensors, "/axis").data[0]), context.layer_ranks[inputs[0]]
        ),
    ),
    "StridedSlice": lambda nodes, inputs, tensors, context: strided_slice(
        nodes[-1].name,
        inputs[0],
        context.layer_ranks[inputs[0]],
        begin=tensors[0].data,
        end=tensors[1].data,
        strides=tensors[2].data,
        begin_mask=get_attr(nodes[-1], "begin_mask"),
        end_mask=get_attr(nodes[-1], "end_mask"),
        ellipsis_mask=get_attr(nodes[-1], "ellipsis_mask"),
        new_axis_mask=get_attr(nodes[-1], "new_axis_mask"),
        shrink_axis_mask=get_attr(nodes[-1], "shrink_axis_mask"),
    ),
    "BatchNormalization": lambda nodes, inputs, tensors, _: Struct(
        op="BatchNormalization",
        input=[i for i in inputs]
        + order_by([t.name for t in tensors], ["gamma", "beta", "mean", "variance"]),
    ),
    "InstanceNormalization_ByTensorName": lambda nodes, inputs, tensors, _: Struct(
        op="InstanceNormalization",
        input=[i for i in inputs]
        + order_by([t.name for t in tensors], ["scale", "offset"]),
    ),
    "InstanceNormalization_ByTensorOrder": lambda nodes, inputs, tensors, _: Struct(
        op="InstanceNormalization",
        input=[i for i in inputs] + [t.name for t in tensors][-2:],
    ),
    "Dense": lambda nodes, inputs, tensors, _: Struct(
        op="Dense",
        input=[i for i in inputs] + [t.name for t in tensors],
        data_frmt=get_attr(
            by_op(nodes, "Dense") or by_op(nodes, "MatMul"), "data_format"
        ),
    ),
    "Conv2D": lambda nodes, inputs, tensors, _: Struct(
        op="Conv2D",
        input=[i for i in inputs] + [t.name for t in tensors],
        padding=get_attr(by_op(nodes, "Conv2D"), "padding"),
        strides=get_attr(by_op(nodes, "Conv2D"), "strides"),
        dilations=get_attr(by_op(nodes, "Conv2D"), "dilations"),
        data_frmt=get_attr(by_op(nodes, "Conv2D"), "data_format"),
    ),
    "DepthwiseConv2dNative": lambda nodes, inputs, tensors, _: Struct(
        op="DepthwiseConv2dNative",
        input=[i for i in inputs] + [t.name for t in tensors],
        padding=get_attr(by_op(nodes, "DepthwiseConv2dNative"), "padding"),
        strides=get_attr(by_op(nodes, "DepthwiseConv2dNative"), "strides"),
        dilations=get_attr(by_op(nodes, "DepthwiseConv2dNative"), "dilations"),
        data_frmt=get_attr(by_op(nodes, "DepthwiseConv2dNative"), "data_format"),
    ),
    "Conv2DBackpropInput": lambda nodes, inputs, tensors, _: Struct(
        op="Conv2DBackpropInput",
        input=[i for i in inputs]
        + [t.name for t in tensors][1:][
            -2:
        ],  # [1:]  - skips the 0th tensor, since Conv2DBackpropInput 0th tensor is 'input_sizes'
        # (which differs from other Conv layers)
        # [-2:] - take only last 2 tensors, this allows to process large patterns with the same code
        padding=get_attr(by_op(nodes, "Conv2DBackpropInput"), "padding"),
        strides=get_attr(by_op(nodes, "Conv2DBackpropInput"), "strides"),
        dilations=get_attr(by_op(nodes, "Conv2DBackpropInput"), "dilations"),
        data_frmt=get_attr(by_op(nodes, "Conv2DBackpropInput"), "data_format"),
    ),
    "ResizeNearestNeighbor": lambda nodes, inputs, tensors, _: Struct(
        op="ResizeNearestNeighbor",
        input=[i for i in inputs],
        ksize=[int(tensors[0].data[0]), int(tensors[0].data[1])]
        if len(tensors) == 1 and len(tensors[0].data) == 2
        else [int(tensors[-1].data[0]), int(tensors[-1].data[1])]
        if len(tensors) >= 4 and len(tensors[-1].data) == 2
        else [1, 1],
    ),
    "Mean": lambda nodes, inputs, tensors, _:
    # take only the last input
    barracuda.mean(nodes[-1].name, inputs[-1], axis=tensors[0].data),
    "SquaredDifference": lambda nodes, inputs, tensors, _: sqr_diff(
        nodes[-1].name, inputs[0], inputs[1]
    ),
    "BasicLSTMReshapeOut": lambda nodes, inputs, tensors, context: basic_lstm(
        nodes, inputs, tensors, context, find_type="Reshape"
    ),
    "BasicLSTMConcatOut": lambda nodes, inputs, tensors, context: basic_lstm(
        nodes, inputs, tensors, context, find_type="ConcatV2"
    ),
    "Swish": lambda nodes, inputs, tensors, _: Struct(op="Swish", input=inputs),
    "LeakyRelu": lambda nodes, inputs, tensors, _: Struct(op="LeakyRelu", input=inputs),
    # TODO:'Round'
    # TODO:'Rsqrt'
}


# Debug
def debug(s):
    print(s)
    return s


# Helper
def embody(v, default=0):
    return default if v is None else v


# Parse
def get_attr(node, attr_name, default=None):
    if type(node) == Struct:
        if hasattr(node, attr_name):
            return getattr(node, attr_name)
        else:
            return default

    # See: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/attr_value.proto
    val = node.attr[attr_name]

    if val.HasField("list"):
        return val.list.i
        # NOTE: can't find way to identify type of list BUT it is almost always list(int)
        # except list(float) in FractionalAvg/MaxPool
    if val.HasField("b"):
        return val.b
    if val.HasField("i"):
        return val.i
    if val.HasField("f"):
        return val.f
    if val.HasField("s"):
        return val.s.decode("utf-8")
    if val.HasField("shape"):
        return val.shape
    if val.HasField("tensor"):
        return val.tensor
    return default


def get_epsilon(layer):
    return get_attr(
        layer, "epsilon", default=0.001
    )  # default epsilon taken from tf.layers.batch_normalization


def get_layer_rank(layer):
    shape = get_attr(layer, "shape")
    if not shape:
        return None
    if isinstance(shape, list):
        return 1
    shape = [dim.size for dim in shape.dim]
    return len(shape)


def get_layer_shape(layer):
    shape = get_attr(layer, "shape")
    if not shape:
        return [-1, -1, -1, -1]
    shape = [dim.size for dim in shape.dim]
    if len(shape) == 1:
        return [1, 1, 1, shape[0]]
    if len(shape) == 2:
        return [shape[0], 1, 1, shape[1]]
    if len(shape) == 3:
        return [shape[0], 1, shape[1], shape[2]]
    return shape


def get_tensor_dims(tensor):
    if isinstance(tensor, np.ndarray):
        return np.shape(tensor)

    dims = []
    if tensor.tensor_shape:
        dims = [v.size for v in tensor.tensor_shape.dim]
    if tensor.float_val:
        dims = np.shape(tensor.float_val)
    if tensor.int_val:
        dims = np.shape(tensor.int_val)
    if tensor.bool_val:
        dims = np.shape(tensor.bool_val)
    return dims


def get_tensor_dtype(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor.dtype

    dataType = ""
    fields = tensor.ListFields()

    for field, value in fields:
        if (
            field.name == "dtype"
            and field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_ENUM
        ):
            dataType = field.enum_type.values_by_number.get(value, None).name

    return dataType


def get_tensor_data(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor.astype(float)

    dims = get_tensor_dims(tensor)
    elems = np.product(dims)

    if tensor.tensor_content:
        # TODO: support other types
        dataType = get_tensor_dtype(tensor)
        if dataType == "DT_FLOAT":
            data = struct.unpack("<" + str(elems) + "f", tensor.tensor_content)
        elif dataType == "DT_INT32":
            data = struct.unpack("<" + str(elems) + "i", tensor.tensor_content)
        elif dataType == "DT_BOOL":
            data = struct.unpack("<" + str(elems) + "?", tensor.tensor_content)
        else:
            print("UNSUPPORTED: data type", dataType)
    if tensor.float_val:
        data = tensor.float_val
    if tensor.int_val:
        data = np.array(tensor.int_val, dtype=float)
    if tensor.bool_val:
        data = np.array(tensor.bool_val, dtype=float)
    return np.array(data).reshape(dims)


def flatten(items, enter=lambda x: isinstance(x, list)):
    # http://stackoverflow.com/a/40857703
    # https://github.com/ctmakro/canton/blob/master/canton/misc.py
    """Yield items from any nested iterable; see REF."""
    for x in items:
        if enter(x):
            yield from flatten(x)
        else:
            yield x


def replace_strings_in_list(array_of_strigs, replace_with_strings):
    "A value in replace_with_strings can be either single string or list of strings"
    potentially_nested_list = [
        replace_with_strings.get(s) or s for s in array_of_strigs
    ]
    return list(flatten(potentially_nested_list))


def remove_duplicates_from_list(array):
    "Preserves the order of elements in the list"
    output = []
    unique = set()
    for a in array:
        if a not in unique:
            unique.add(a)
            output.append(a)
    return output


#########################################################


def pool_to_HW(shape, data_frmt):
    """ Convert from NHWC|NCHW => HW
    """
    if len(shape) != 4:
        return shape  # Not NHWC|NCHW, return as is
    if data_frmt == "NCHW":
        return [shape[2], shape[3]]
    return [shape[1], shape[2]]


def strides_to_HW(shape, format):
    return pool_to_HW(shape, format)


def axis_to_barracuda(axis, input_rank):
    N = 0
    H = 1
    W = 2
    C = 3
    if axis < 0:
        axis = input_rank - axis
    assert axis >= 0
    assert axis < input_rank
    if input_rank == 4:
        # [NHWC]
        return [N, H, W, C][axis]
    if input_rank == 3:
        # [N_WC]
        return [N, W, C][axis]
    elif input_rank == 2:
        # [N__C]
        return [N, C][axis]
    elif input_rank == 1:
        # [___C]
        return [C][axis]
    return -1


#########################################################


def sqr_diff(name, a, b):
    nn = barracuda.Build(name)
    d = nn.sub(a, b)
    nn.mul(d, d, out=name)
    return nn.layers


def strided_slice(
    name,
    input,
    input_rank,
    begin,
    end,
    strides,
    begin_mask,
    end_mask,
    ellipsis_mask,
    new_axis_mask,
    shrink_axis_mask,
):
    assert input_rank != -1
    begin = begin.astype(np.int32).tolist()
    end = end.astype(np.int32).tolist()
    strides = strides.astype(np.int32).tolist()

    # StridedSlice range and mask descriptions:
    #   https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/strided-slice
    # TODO: I don't think elipsis and newaxis would work together well with current implementation

    assert len(begin) == len(end)
    assert len(begin) == len(strides)

    # prepare begin, end, stride arrays
    output_rank = input_rank
    insert_pos = 0
    while ellipsis_mask:
        ellipsis_mask >>= 1
        insert_pos += 1

    # NOTE: begin=0, end=0, stride=1  <=  full range from existing axis
    #       begin=0, end=0, stride=0  <=  new axis OR shrink axis to single 1st element
    #       begin=N, end=N, stride=0  <=              shrink axis to single Nth element
    while len(begin) < input_rank:
        if insert_pos:
            begin.insert(insert_pos, 0)
            end.insert(insert_pos, 0)
            strides.insert(insert_pos, 1)
        else:
            begin.append(0)
            end.append(0)
            strides.append(1)
    assert len(begin) <= input_rank

    descriptor_count = input_rank
    for i in range(len(begin)):
        if begin_mask & (1 << i):
            begin[i] = 0
        if end_mask & (1 << i):
            end[i] = 0
        if new_axis_mask & (1 << i):
            begin[i] = end[i] = strides[i] = 0
            output_rank += 1
        if shrink_axis_mask & (1 << i):
            end[i] = begin[i]
            strides[i] = 0
            output_rank -= 1

    # convert to Barracuda layout
    descriptor_count = len(begin)
    assert descriptor_count <= 4
    if descriptor_count == 3:
        begin = [begin[0], 0, begin[1], begin[2]]
        end = [end[0], 0, end[1], end[2]]
        strides = [strides[0], 1, strides[1], strides[2]]
    elif descriptor_count == 2:
        begin = [begin[0], 0, 0, begin[1]]
        end = [end[0], 0, 0, end[1]]
        strides = [strides[0], 1, 1, strides[1]]
    elif descriptor_count == 1:
        begin = [0, 0, 0, begin[0]]
        end = [0, 0, 0, end[0]]
        strides = [1, 1, 1, strides[0]]

    nn = barracuda.Build(name)
    nn.strided_slice(input, begin, end, strides, output_rank, out=name)
    return nn.layers


# search backwards starting from index_of_actual_output_node for non-const node
def locate_actual_output_node(
    nodes, index_of_actual_output_node=-1, find_type="Reshape"
):
    while (-index_of_actual_output_node - 1) < len(nodes) and nodes[
        index_of_actual_output_node
    ].op != find_type:
        index_of_actual_output_node -= 1
    actual_output_node = nodes[index_of_actual_output_node]
    assert -index_of_actual_output_node < len(nodes)
    return actual_output_node


def gru(
    nodes,
    inputs,
    tensors,
    context,
    index_of_actual_output_node,
    assert_output_node_op_type=None,
):
    assert len(inputs) == 2

    def find_tensor_by_name(name, default=None):
        nonlocal tensors
        candidates = [t for t in tensors if t.name.endswith(name)]
        return candidates[0].data if candidates else default

    input = inputs[-1]
    state = inputs[0]
    gates_kernel = find_tensor_by_name("/gates/kernel")
    gates_bias = find_tensor_by_name(
        "/gates/bias", default=np.zeros(np.shape(gates_kernel)[-1])
    )
    candidate_kernel = find_tensor_by_name("/candidate/kernel")
    candidate_bias = find_tensor_by_name(
        "/candidate/bias", default=np.zeros(np.shape(candidate_kernel)[-1])
    )
    new_state = nodes[-1].name + "_h"

    assert np.shape(gates_kernel)[-1] == np.shape(gates_bias)[-1]
    assert np.shape(candidate_kernel)[-1] == np.shape(candidate_bias)[-1]

    num_gates = 2
    seq_length = 1
    hidden_size = np.shape(gates_kernel)[-1] // num_gates

    gate_kernels = np.split(gates_kernel, num_gates, axis=-1)
    gate_biases = np.split(gates_bias, num_gates, axis=-1)

    context.model_tensors["kernel_r"] = gate_kernels[0]
    context.model_tensors["kernel_u"] = gate_kernels[1]
    context.model_tensors["kernel_c"] = candidate_kernel
    context.model_tensors["bias_r"] = gate_biases[0]
    context.model_tensors["bias_u"] = gate_biases[1]
    context.model_tensors["bias_c"] = candidate_bias

    context.layer_ranks[state] = 2

    new_layers = barracuda.gru(
        "gru",
        input,
        state,
        "kernel_r",
        "kernel_u",
        "kernel_c",
        "bias_r",
        "bias_u",
        "bias_c",
        new_state,
    )

    state_shape = [1, 1, seq_length, hidden_size]
    context.model_memories += [state_shape, state, new_state]

    # map exptected output of the replaced pattern to output from our GRU cell
    actual_output_node = locate_actual_output_node(
        nodes, index_of_actual_output_node, assert_output_node_op_type
    )
    context.map_ignored_layer_to_its_input[actual_output_node.name] = new_state

    return new_layers


def basic_lstm(nodes, inputs, tensors, context, find_type="Reshape"):
    assert len(inputs) == 2

    def find_tensor_by_name(name, default=None):
        nonlocal tensors
        candidates = [t for t in tensors if t.name.endswith(name)]
        return candidates[0].data if candidates else default

    def find_forget_bias():
        nonlocal nodes
        nonlocal tensors
        # TODO: make it more fault-tolerant
        # search for scalar float constant that is input to Add node
        # and hope it is not a constant for some complex activation function
        for t in tensors:
            if np.prod(t.shape) == 1 and get_tensor_dtype(t.obj) == "DT_FLOAT":
                for n in nodes:
                    if n.op == "Add" and t.name in n.input:
                        return t.data
        return np.zeros(1)

    input = inputs[-1]
    state_c = inputs[0] + "_c"
    state_h = inputs[0] + "_h"
    kernel = find_tensor_by_name("/kernel")
    bias = find_tensor_by_name("/bias", default=np.zeros(np.shape(kernel)[-1]))
    forget_bias = find_forget_bias()
    new_state_c = nodes[-1].name + "_c"
    new_state_h = nodes[-1].name + "_h"

    assert np.shape(kernel)[-1] == np.shape(bias)[-1]

    num_gates = 4
    seq_length = 1
    hidden_size = np.shape(kernel)[-1] // num_gates

    kernels = np.split(kernel, num_gates, axis=-1)
    biases = np.split(bias, num_gates, axis=-1)

    context.model_tensors["kernel_i"] = kernels[0]
    context.model_tensors["kernel_j"] = kernels[1]
    context.model_tensors["kernel_f"] = kernels[2]
    context.model_tensors["kernel_o"] = kernels[3]
    context.model_tensors["bias_i"] = biases[0]
    context.model_tensors["bias_j"] = biases[1]
    context.model_tensors["bias_f"] = biases[2] + forget_bias
    context.model_tensors["bias_o"] = biases[3]

    context.layer_ranks[state_c] = 2
    context.layer_ranks[state_h] = 2

    # lstm_value/strided_slice/stack => lstm_value
    lstm_name = next(i.name for i in nodes if i.name.startswith("lstm")).split("/")[0]

    new_layers = barracuda.lstm(
        lstm_name,
        input,
        state_c,
        state_h,
        "kernel_i",
        "kernel_j",
        "kernel_f",
        "kernel_o",
        "bias_i",
        "bias_j",
        "bias_f",
        "bias_o",
        new_state_c,
        new_state_h,
    )

    state_shape = [1, 1, seq_length, hidden_size]
    context.model_memories += [state_shape, state_c, new_state_c]
    context.model_memories += [state_shape, state_h, new_state_h]

    # map expected output of the replaced pattern to output from our LSTM cell
    actual_output_node = locate_actual_output_node(nodes, -1, find_type)
    concat_out_node = locate_actual_output_node(nodes, -1, "ConcatV2")
    context.map_ignored_layer_to_its_input[actual_output_node.name] = new_state_h
    context.map_ignored_layer_to_its_input[concat_out_node.name] = new_state_c

    return new_layers


#########################################################


def process_layer(layer, context, args):
    model_tensors = context.model_tensors
    input_shapes = context.input_shapes
    layer_ranks = context.layer_ranks
    map_ignored_layer_to_its_input = context.map_ignored_layer_to_its_input

    name = layer.name
    class_name = layer.op
    inputs = (
        layer.input
    )  # Tensorflow inputs are always explicit, but in case of Keras we had 'inputs = layer.input or [prev_layer_name]'
    inputs = replace_strings_in_list(inputs, map_ignored_layer_to_its_input)

    if class_name == "Nop":
        assert len(inputs) <= 1
        map_ignored_layer_to_its_input[name] = inputs
        return

    if class_name == "Const":
        model_tensors[name] = layer.attr["value"].tensor
        layer_ranks[name] = (
            get_layer_rank(layer) or 1
        )  # we treast constants without shape as rank=1 (scalar converted to tensor)
        return

    if class_name == "Placeholder":
        assert inputs == []
        map_ignored_layer_to_its_input[name] = inputs
        input_shapes[name] = get_layer_shape(layer)
        layer_ranks[name] = get_layer_rank(layer)
        return

    if class_name == "Identity":
        connected_to_const = len(inputs) == 1 and inputs[0] in model_tensors
        if connected_to_const:
            map_ignored_layer_to_its_input[name] = inputs
            return
        else:
            # treat Identity layer that are connected to processing nodes
            # as output from the network
            class_name = "Linear"

    if args.print_layers or args.verbose:
        var_tensors = [i for i in inputs if i not in model_tensors]
        const_tensors = [i for i in inputs if i in model_tensors]
        print(
            "'%s' %s Vars:%s Const:%s" % (name, class_name, var_tensors, const_tensors)
        )

    if class_name in known_activations:
        activation = class_name
        class_name = "Activation"
    else:
        activation = "Linear"

    if class_name not in known_classes:
        if class_name in requires_runtime_flag:
            print("SKIP:", class_name, "layer is used only for training")
        else:
            print("IGNORED:", class_name, "unknown layer")
        map_ignored_layer_to_its_input[name] = inputs
        return

    klass = known_classes[class_name]
    if type(klass) == int:
        klass = Struct(id=klass)

    o_l = Struct()
    o_l.type = klass.id
    o_l.class_name = class_name
    o_l.name = name

    auto_pad = get_attr(layer, "padding")  # layer.attr['padding'].s.decode("utf-8")
    pads = get_attr(layer, "pads")
    strides = get_attr(layer, "strides")  # layer.attr['strides'].list.i
    pool_size = get_attr(layer, "ksize")  # layer.attr['ksize'].list.i
    shape = get_attr(layer, "shape")
    starts = get_attr(layer, "starts")
    ends = get_attr(layer, "ends")
    slice_strides = get_attr(layer, "slice_strides")
    rank = get_attr(layer, "rank") or get_layer_rank(layer)
    data_frmt = get_attr(
        layer, "data_format"
    )  # layer.attr['data_format'].s.decode("utf-8")
    axis = get_attr(layer, "axis")
    alpha = get_attr(layer, "alpha", default=1)
    beta = get_attr(layer, "beta")

    if activation and activation not in known_activations:
        print("IGNORED: unknown activation", activation)
    if auto_pad and auto_pad not in known_paddings:
        print("IGNORED: unknown padding", auto_pad)
    if data_frmt and data_frmt not in supported_data_formats:
        print("UNSUPPORTED: data format", data_frmt)

    o_l.activation = known_activations.get(activation) or 0
    o_l.pads = (
        known_paddings.get(auto_pad) if auto_pad else pads or starts or [0, 0, 0, 0]
    )
    o_l.strides = strides_to_HW(strides, data_frmt) if strides else slice_strides or []
    o_l.pool_size = (
        pool_to_HW(pool_size, data_frmt) if pool_size else ends or shape or []
    )
    o_l.axis = embody(axis, default=-1)
    o_l.alpha = embody(alpha, default=1)
    o_l.beta = beta or 0
    o_l.rank = (
        -1
    )  # default initialization, actual value will be set later on in this function

    tensor_names = [i for i in inputs if i in model_tensors]
    o_l.tensors = [
        Struct(
            name=x,
            shape=get_tensor_dims(model_tensors[x]),
            data=get_tensor_data(model_tensors[x]),
        )
        for x in tensor_names
    ]
    # Patch shapes & data
    layer_has_model_tensors = len(o_l.tensors) > 0
    if hasattr(klass, "out_shapes") and layer_has_model_tensors:
        shapes = klass.out_shapes([x.shape for x in o_l.tensors])

        # if we have more shapes than actual tensors,
        # then create & fill missing tensors with zeros
        in_tensor_num = len(o_l.tensors)
        for index, new_shape in enumerate(shapes):
            if index >= in_tensor_num:
                new_tensor = Struct(
                    name=("%s/patch:%i") % (name, index - in_tensor_num),
                    shape=new_shape,
                    data=np.zeros(new_shape),
                )
                o_l.tensors.append(new_tensor)
        assert len(shapes) <= len(o_l.tensors)

        if hasattr(klass, "patch_data"):
            data = [x.data for x in o_l.tensors]

            patch_data_fn = klass.patch_data
            patch_data_expected_arg_count = patch_data_fn.__code__.co_argcount
            patch_data_args = (
                (data, layer) if patch_data_expected_arg_count > 1 else (data,)
            )
            tensor_data = patch_data_fn(*patch_data_args)
            o_l.tensors = o_l.tensors[
                : len(tensor_data)
            ]  # resize tensor array to match patched data - patching might reduce number of tensors
            for x, data in zip(o_l.tensors, tensor_data):
                x.data = data

        # after this point we should have equal amount of shapes and tensors
        assert len(o_l.tensors) == len(shapes)

        for x, shape in zip(o_l.tensors, shapes):
            assert x.data.size == np.prod(shape)
            x.shape = shape

        o_l.inputs = [i for i in inputs if i not in model_tensors]

    else:
        # no 'patch_data' lambda was specifiowned, op does not require tensor args
        o_l.tensors = []
        o_l.inputs = inputs

    # Force all tensors to float32
    for x in o_l.tensors:
        x.data = x.data.astype(np.float32)

    input_ranks = [layer_ranks.get(i, -1) for i in o_l.inputs]
    for i in o_l.inputs:
        if i not in layer_ranks and "lstm" not in i:
            print("WARNING: rank unknown for tensor", i, "while processing node", name)
    if hasattr(klass, "rank"):
        rank = klass.rank
        if hasattr(rank, "__call__"):
            assert (
                -1 not in input_ranks
            )  # for rank() lambda all input ranks have to be known (not -1)
            rank = rank(input_ranks)
    if rank is None:

        def all_elements_equal(arr):  # http://stackoverflow.com/q/3844948/
            return arr.count(arr[0]) == len(arr)

        assert len(input_ranks) > 0
        assert all_elements_equal(input_ranks)
        rank = input_ranks[0]
    layer_ranks[name] = rank
    o_l.rank = rank

    # Layer is ready
    context.layers.append(o_l)


class ModelBuilderContext:
    def __init__(self):
        self.layers = []
        self.input_shapes = {}
        self.model_tensors = {}
        self.model_memories = []
        self.layer_ranks = {}
        self.map_ignored_layer_to_its_input = {}


def process_model(model, args):
    o_context = ModelBuilderContext()

    # Find node patterns
    nodes_as_array = [node for node in model.node]
    nodes_as_array = slow_but_stable_topological_sort(nodes_as_array, verbose=True)

    node_index = 0
    while node_index < len(nodes_as_array):
        node = nodes_as_array[node_index]
        match = False
        for pattern_repr, pattern_name in known_patterns.items():
            pattern = eval(pattern_repr)
            if node_index + len(pattern) > len(nodes_as_array):
                continue  # pattern too long, skip

            require_exact_match = pattern[0] == "Const" or pattern[0] == "Identity"
            pattern_end = node_index

            def match_node(node, pattern):
                return node.op == pattern or (
                    hasattr(pattern, "match") and pattern.match(node.name)
                )

            for p in pattern:
                if not require_exact_match:
                    while (
                        pattern_end < len(nodes_as_array)
                        and nodes_as_array[pattern_end].op != p
                        and (
                            nodes_as_array[pattern_end].op == "Const"
                            or nodes_as_array[pattern_end].op == "Identity"
                        )
                    ):
                        pattern_end += 1
                if pattern_end >= len(nodes_as_array):
                    break

                match = False
                if hasattr(p, "match"):  # regexp
                    while pattern_end < len(nodes_as_array) and p.match(
                        nodes_as_array[pattern_end].name
                    ):
                        match = True
                        pattern_end += 1
                else:  # exact string
                    match = nodes_as_array[pattern_end].op == p
                    pattern_end += 1

                if not match:
                    break

            def get_tensors(pattern_nodes):
                nonlocal o_context
                map_ignored_layer_to_its_input = (
                    o_context.map_ignored_layer_to_its_input
                )
                model_tensors = o_context.model_tensors

                # tensors <= all Const nodes within this pattern
                const_nodes = [n for n in pattern_nodes if n.op == "Const"]

                # TODO: unify / reuse code from process_layer
                identity_nodes = [n for n in pattern_nodes if n.op == "Identity"]
                for i in identity_nodes:
                    inputs = replace_strings_in_list(
                        i.input, map_ignored_layer_to_its_input
                    )
                    map_ignored_layer_to_its_input[i.name] = inputs

                # gather inputs from Op nodes (not Const, not Identity)
                op_nodes = [
                    n
                    for n in pattern_nodes
                    if n not in const_nodes and n not in identity_nodes
                ]
                inputs_to_op_nodes = list(
                    flatten([list(flatten(n.input)) for n in op_nodes])
                )
                inputs_to_op_nodes = replace_strings_in_list(
                    inputs_to_op_nodes, map_ignored_layer_to_its_input
                )
                inputs_to_op_nodes = [i.split(":")[0] for i in inputs_to_op_nodes]

                const_nodes_by_name = {n.name: n for n in const_nodes}
                tensors = []
                for i in inputs_to_op_nodes:
                    if i in model_tensors:
                        src = model_tensors[i]
                        tensors += [
                            Struct(
                                name=i,
                                obj=src,
                                shape=get_tensor_dims(src),
                                data=get_tensor_data(src),
                            )
                        ]
                    elif i in const_nodes_by_name:
                        src = const_nodes_by_name[i].attr["value"].tensor
                        tensors += [
                            Struct(
                                name=i,
                                obj=src,
                                shape=get_tensor_dims(src),
                                data=get_tensor_data(src),
                            )
                        ]
                tensor_names = [n.name for n in tensors]

                # filter only inputs that are coming from nodes that are outside this pattern
                # preserve the order
                pattern_nodes = [n.name for n in pattern_nodes] + tensor_names
                # inputs_from_outside_pattern = remove_duplicates_from_list([i for i in inputs_to_op_nodes if
                #   nodes_by_name[i] not in pattern_nodes])
                inputs_from_outside_pattern = remove_duplicates_from_list(
                    [i for i in inputs_to_op_nodes if i not in pattern_nodes]
                )

                return inputs_from_outside_pattern, tensors

            if match:
                nodes = nodes_as_array[node_index:pattern_end]
                name = nodes[-1].name
                var_tensors, const_tensors = get_tensors(nodes)
                if args.print_patterns or args.verbose:
                    print(
                        "PATTERN:",
                        name,
                        "~~",
                        pattern_name,
                        "<-",
                        var_tensors,
                        "+",
                        [t.name for t in const_tensors],
                    )
                    print("        ", pattern)
                for n in nodes:
                    if n.op == "Const" or n.op == "Identity":
                        process_layer(n, o_context, args)

                new_layers = transform_patterns[pattern_name](
                    nodes, var_tensors, const_tensors, o_context
                )
                if not isinstance(new_layers, list):
                    if not hasattr(new_layers, name):
                        new_layers.name = name
                    new_layers = [new_layers]

                for l in new_layers:
                    # TODO: prefix new layer names with scope, patch inputs
                    # l.name = name + '/' + l.name
                    process_layer(l, o_context, args)

                node_index = pattern_end
                break  # pattern found & processed

        if not match:
            # TODO: gather tensors in the same way as patterns do
            process_layer(node, o_context, args)
            node_index += 1

    def find_unconnected_const_nodes(nodes):
        nodes_with_consts = {node.name: node for node in nodes if node.op == "Const"}
        for node in nodes:
            for i in node.input:
                nodes_with_consts.pop(i, None)
        return list(nodes_with_consts.keys())

    return (
        o_context.layers,
        o_context.input_shapes,
        o_context.model_tensors,
        o_context.model_memories,
        find_unconnected_const_nodes(nodes_as_array),
    )


# Sort nodes so that all input dependencies are satisfied beforehand
# while preserving original order of the nodes in the model whenever possible.
# NOITE: preservation of original order is important for pattern matching
def slow_but_stable_topological_sort(nodes, verbose):

    nodes_with_consts = [node for node in nodes if node.op == "Const"]
    nodes_for_sorting = [node for node in nodes if node.op != "Const"]

    # TODO: optimize for performance
    # based on http://blog.gapotchenko.com/stable-topological-sort

    def assign_ids(nodes):
        ids = []
        id_by_name = {}
        id = 0
        for node in nodes:
            id_by_name[node.name] = id
            ids.append(id)
            id += 1

        inputs_by_id = [None] * len(nodes)
        for node in nodes:
            id = id_by_name[node.name]
            inputs_by_id[id] = {id_by_name.get(i, -1) for i in node.input}

        return ids, inputs_by_id

    def sort(ids, inputs_by_id, verbose_lambda):
        sorted = False
        n = len(ids)
        while not sorted:
            sorted = True
            for i in range(n):
                for j in range(i):
                    if ids[i] in inputs_by_id[ids[j]]:
                        tmp = ids.pop(i)
                        ids.insert(j, tmp)
                        sorted = False
            verbose_lambda(sorted)
        return ids

    prefix_printed = False

    def print_status(sorted):
        nonlocal prefix_printed
        if not sorted:
            if not prefix_printed:
                print("Sorting model, may take a while...", end="", flush=True)
                prefix_printed = True
            else:
                print(".", end="", flush=True)
        else:
            if prefix_printed:
                print(" Done!")

    ids, inputs_by_id = assign_ids(nodes_for_sorting)
    ids = sort(
        ids, inputs_by_id, lambda sorted: print_status(sorted) if verbose else None
    )

    assert len(ids) == len(nodes_for_sorting)
    assert len(ids) + len(nodes_with_consts) == len(nodes)
    return nodes_with_consts + [nodes_for_sorting[id] for id in ids]


def very_slow_but_stable_topological_sort(nodes, verbose):
    # TODO: optimize for performance
    # based on http://blog.gapotchenko.com/stable-topological-sort
    n = len(nodes)
    sorted = False

    while not sorted:
        sorted = True
        for i in range(n):
            for j in range(i):
                if nodes[i].name in nodes[j].input:
                    tmp = nodes.pop(i)
                    nodes.insert(j, tmp)
                    sorted = False
    assert len(nodes) == n
    return nodes


#########################################################


def convert(
    source_file,
    target_file,
    trim_unused_by_output="",
    verbose=False,
    compress_f16=False,
):
    """
    Converts a TensorFlow model into a Barracuda model.
    :param source_file: The TensorFlow Model
    :param target_file: The name of the file the converted model will be saved to
    :param trim_unused_by_output: The regexp to match output nodes to remain in the model.
        All other unconnected nodes will be removed.
    :param verbose: If True, will display debug messages
    :param compress_f16: If true, the float values will be converted to f16
    :return:
    """
    if type(verbose) == bool:
        args = Struct()
        args.verbose = verbose
        args.print_layers = verbose
        args.print_source_json = verbose
        args.print_barracuda_json = verbose
        args.print_layer_links = verbose
        args.print_patterns = verbose
        args.print_tensors = verbose
        args.print_supported_ops = verbose
    else:
        args = verbose

    if args.print_supported_ops:
        barracuda.print_known_operations(known_classes, known_activations)

    # Load Tensorflow model
    print("Converting %s to %s" % (source_file, target_file))
    f = open(source_file, "rb")
    i_model = tf.GraphDef()
    i_model.ParseFromString(f.read())

    if args.verbose:
        print("OP_TYPES:", {layer.op for layer in i_model.node})

    if args.print_source_json or args.verbose:
        for layer in i_model.node:
            if not layer.op == "Const":
                print("MODEL:", MessageToJson(layer) + ",")

    # Convert
    o_model = barracuda.Model()
    o_model.layers, o_input_shapes, o_model.tensors, o_model.memories, o_model.globals = process_model(
        i_model, args
    )

    # Cleanup unconnected Identities (they might linger after processing complex node patterns like LSTM)
    def cleanup_layers(layers):
        all_layers = {l.name for l in layers}
        all_inputs = {i for l in layers for i in l.inputs}

        def is_unconnected_identity(layer):
            if layer.class_name == "Activation" and layer.activation == 0:  # Identity
                assert len(layer.inputs) == 1
                if layer.inputs[0] not in all_layers and layer.name not in all_inputs:
                    return True
            return False

        return [l for l in layers if not is_unconnected_identity(l)]

    o_model.layers = cleanup_layers(o_model.layers)

    all_inputs = {i for l in o_model.layers for i in l.inputs}

    # Trim
    if trim_unused_by_output:
        o_model.layers = barracuda.trim(
            o_model.layers, trim_unused_by_output, args.verbose
        )

    # Create load layer for constants
    def dims_to_barracuda_shape(dims):
        shape = list(dims)
        while len(shape) < 4:
            shape = [1] + shape
        return shape

    const_tensors = [i for i in all_inputs if i in o_model.tensors]
    const_tensors += o_model.globals
    for x in const_tensors:
        shape = dims_to_barracuda_shape(get_tensor_dims(o_model.tensors[x]))
        o_l = Struct(
            type=255,  # Load
            class_name="Const",
            name=x,
            pads=[0, 0, 0, 0],
            strides=[],
            pool_size=[],
            axis=-1,
            alpha=1,
            beta=0,
            activation=0,
            inputs=[],
            tensors=[
                Struct(
                    name=x,
                    shape=shape,
                    data=np.reshape(get_tensor_data(o_model.tensors[x]), shape).astype(
                        np.float32
                    ),
                )
            ],
        )
        o_model.layers.insert(0, o_l)

    # Find model inputs & outputs
    all_layers = {l.name for l in o_model.layers}
    # global inputs => are inputs that are NOT connected to any layer in the network
    # global outputs => are outputs that are NOT feeding any layer in the network OR are coming from Identity layers
    o_model.inputs = {
        i: o_input_shapes[i]
        for l in o_model.layers
        for i in l.inputs
        if i not in all_layers and i not in o_model.memories
    }

    def is_output_layer(layer):
        if (
            layer.class_name == "Const"
        ):  # Constants never count as global output even when unconnected
            return False
        if (
            layer.name not in all_inputs
        ):  # this layer is not inputing to any other layer
            return True
        if (
            layer.class_name == "Activation" and layer.activation == 0
        ):  # Identity marks global output
            return True
        return False

    o_model.outputs = [l.name for l in o_model.layers if is_output_layer(l)]

    # Compress
    if compress_f16:
        o_model = barracuda.compress(o_model)

    # Sort model so that layer inputs are always ready upfront
    o_model.layers = barracuda.sort(
        o_model.layers, o_model.inputs, o_model.memories, args.verbose
    )
    o_model.layers = barracuda.fuse(o_model.layers, args.verbose)

    # Summary
    barracuda.summary(
        o_model,
        print_layer_links=args.print_layer_links or args.verbose,
        print_barracuda_json=args.print_barracuda_json or args.verbose,
        print_tensors=args.print_tensors or args.verbose,
    )

    # Write to file
    barracuda.write(o_model, target_file)
    print("DONE: wrote", target_file, "file.")

from __future__ import print_function
import numpy as np
import onnx
import struct
import barracuda
from google.protobuf.json_format import MessageToJson
from collections import Iterable
from pprint import pprint

# ONNX format:  # See: https://github.com/onnx/onnx/blob/master/onnx/onnx.proto
# ONNX schema: https://github.com/onnx/onnx/blob/master/docs/Operators.md
# ONNX conventions: https://github.com/onnx/onnx/blob/master/docs/OpConventions.md

# Command line argumengts
args = barracuda.parse_args(
    description = 'Convert ONNX model to Barracuda binary',
    source_extension = '.onnx',
    help = 'input ONNX serialized  file')

# Definition of Barracuda supported layers
class Struct:
    "A structure that can have any fields defined."
    def __init__(self, **entries): self.__dict__.update(entries)

def fuse_batchnorm_weights(gamma, beta, mean, var):
    # https://github.com/Tencent/ncnn/blob/master/src/layer/batchnorm.cpp
    """ float sqrt_var = sqrt(var_data[i]);
        a_data[i] = bias_data[i] - slope_data[i] * mean_data[i] / sqrt_var;
        b_data[i] = slope_data[i] / sqrt_var;
        ...
        ptr[i] = b * ptr[i] + a;
    """
    scale = gamma / np.sqrt(var)
    bias = beta - gamma * mean / np.sqrt(var)
    return [scale, bias]

known_classes = {
    'Gemm': Struct(id = 1,
                    patch_shapes = lambda shapes: [
                        shape_to_HW(shapes[0]),                 # W
                        bias(shape_to_HW(shapes[-1])),          # B
                    ],
                    patch_data = lambda data: [
                        data_to_HW(data[0]),
                        data[1]
                    ]),
    'MatMul': Struct(id = 1,
                    patch_shapes = lambda shapes: [
                        shape_to_HW(shapes[0]),                 # W
                        bias(shape_to_HW(shapes[-1])),          # ZERO
                    ],
                    patch_data = lambda data: [
                        data_to_HW(data[0]),
                        np.zeros(np.shape(data[1]))
                    ]),

    'Conv': Struct(
                    id = 20,
                    patch_shapes = lambda shapes: [
                        shape_to_HWCK(shapes[0]),               # K 
                        bias(shape_to_HWCK(shapes[-1]))         # B
                    ],
                    patch_data = lambda data: [
                        data_to_HWCK(data[0]),
                        data[1]
                    ]),
    'SeparableConv':  21, # doesn't exist in ONNX standard yet
    'ConvTranspose': Struct(
                    id = 22,
                    patch_shapes = lambda shapes: [
                        shape_to_HWKC(shapes[0]),               # K 
                        bias(shape_to_HWKC(shapes[-1]))         # B
                    ],
                    patch_data = lambda data: [
                        data_to_HWKC(data[0]),
                        data[1]
                    ]),
    'Upsample':         23, 'ResizeNearest':    23,

    'MaxPool':          25,
    'AveragePool':      26,
    'GlobalMaxPool':    27,
    'GlobalAveragePool':28,

    # TODO: 1D
    # TODO: 3D

    'Activation':       50,
    'BatchNormalization': Struct(
                    id = 51, # after fusion implemented as ScaleBias
                    patch_shapes = lambda shapes: [
                        [1, 1, 1, shapes[0][0]],             # S
                        [1, 1, 1, shapes[0][0]],             # B
                    ],
                    patch_data = lambda data:
                        # fuse [gamma, beta, mean, var] => [scale, bias]
                        fuse_batchnorm_weights(data[0], data[1], data[2], data[3]) if len(data) == 4 else
                        # fuse [ONE, beta, mean, var] => [scale, bias]
                        fuse_batchnorm_weights(np.ones(np.shape(data[0])), data[0], data[1], data[2])
                    ),
    'BatchNormalizationRuntime': Struct(
                    id = 52,
                    patch_shapes = lambda shapes: [
                        [1, 1, 1, shapes[0][0]],             # G
                        [1, 1, 1, shapes[0][0]],             # B
                    ],
                    patch_data = lambda data:
                        [data[0], data[1]] if len(data) == 4 else
                        [np.ones(np.shape(data[0])), data[0]]
                    ),
    'InstanceNormalization': Struct(
                    id = 52,
                    patch_shapes = lambda shapes: [
                        [1, 1, 1, shapes[0][0]],             # G
                        [1, 1, 1, shapes[0][0]],             # B
                    ],
                    patch_data = lambda data:
                        [data[0], data[1]] if len(data) == 2 else
                        [np.ones(np.shape(data[0])), data[0]]
                    ),
    'LRN':              53, # TODO: args(beta, bias)

    'DropoutRuntime':   60,
    'RandomNormalLike': 64,
    'RandomUniformLike':65,
    'Multinomial':      66,

    'Add':              100,
    'Sum':              100,

    'Sub':              101,
    'Mul':              102,
    'Div':              103,
    'Pow':              104,

    'Min':              110,
    'Max':              111,

    'ReduceL1':         120, # TODO: args(axes)
    'ReduceL2':         121, # TODO: args(axes)
    'ReduceLogSum':     122, # TODO: args(axes)
    'ReduceLogSumExp':  123, # TODO: args(axes)
    'ReduceMax':        124, # TODO: args(axes)
    'ReduceMean':       125, # TODO: args(axes)
    'ReduceMin':        126, # TODO: args(axes)
    'ReduceProd':       127, # TODO: args(axes)
    'ReduceSum':        128, # TODO: args(axes)
    'ReduceSumSquare':  129, # TODO: args(axes)

    'Flatten':          200,
    'Reshape':          201,
    'Transpose':        202,
    'Squeeze':          203, # TODO: args(axes)
    'Unsqueeze':        204, # TODO: args(axes)

    'Concat':           210,
    'Slice':            211, # TODO: args(axes, ends, starts)
    'Pad':              212, # TODO: args(pads, value, mode)
    'Crop':             213, # TODO: args(border, scale)
    'Tile':             214  # TODO: args(repeats)
}

requires_runtime_flag = {
    'Dropout' : 'DropoutRuntime',
    'BatchNormalization' : 'BatchNormalizationRuntime',
}

known_activations = {
    'Linear' : 0,       # dummy, doesn't exist in ONNX standard, all nodes have no activation by default
    'Identity' : 0,
    'Relu' : 1,
    'Softmax' : 2,      # TODO: args(axis)
    'Tanh' : 3,
    'Sigmoid' : 4,
    'Elu' : 5,
    'Relu6' : 6,        # doesn't exist in ONNX standard yet
    'LeakyRelu' : 7,
    'Selu' : 8,         # TODO: args(gamma)
    'Swish' : 9,        # doesn't exist in ONNX standard yet

    'LogSoftmax': 10,   # TODO: args(axis)
    'Softplus': 11,
    'Softsign': 12,

    'Hardmax': 20,      # TODO: args(axis)
    'HardSigmoid': 21,

    'Abs': 100,
    'Neg': 101,
    'Ceil': 102,
    'Clip': 103,
    'Floor': 104,

    'Reciprocal': 110,
    'Sqrt': 111,
    'Pow': 112,
    'Exp': 113,
    'Log': 114,

    'Acos': 200,
    'Acosh': 201,
    'Asin': 202,
    'Asinh': 203,
    'Atan': 204,
    'Atanh': 205,
    'Cos': 206,
    'Cosh': 207,
    'Sin': 208,
    'Sinh': 209,
    'Tan': 210
}

known_paddings = {
    'VALID' : lambda kernel: [0,0,0,0],
    # @TODO: support UPPER/LOWER properly, OTOH they are legacy
    'SAME_UPPER'  : lambda kernel: np.array([np.array(kernel)//2, np.array(kernel)//2]).flatten().tolist(),
    'SAME_LOWER'  : lambda kernel: np.array([np.array(kernel)//2, np.array(kernel)//2]).flatten().tolist(),
}

# Load
print("Converting %s to %s" % (args.source_file, args.target_file))
model = onnx.load(args.source_file)

if args.print_source_json or args.verbose:
    for layer in model.graph.node:
        print(MessageToJson(layer) + ",")

# Parse
def get_attr(node, attr_name, default=None):
    for attr in node.attribute:
        if attr.name == attr_name:
            if attr.AttributeType.Name(attr.type) == 'INTS':
                return attr.ints
            elif attr.AttributeType.Name(attr.type) == 'FLOATS':
                return attr.floats
            elif attr.AttributeType.Name(attr.type) == 'STRINGS':
                return attr.strings
            elif attr.AttributeType.Name(attr.type) == 'INT':
                return attr.i
            elif attr.AttributeType.Name(attr.type) == 'FLOAT':
                return attr.f
            elif attr.AttributeType.Name(attr.type) == 'STRING':
                return attr.s.decode("utf-8")
            elif attr.AttributeType.Name(attr.type) == 'TENSOR':
                return attr.t
    return default

def axis_to_NHWC(axis):
    """ Convert from NCHW => NHWC
    """
    assert(axis >= -1 and axis < 4)
    return [-1, 0, 3, 1, 2][axis+1];

def shape_to_NHWC(shape):
    """ Convert from NCHW => NHWC
    """
    if len(shape) != 4:
        return shape # Not NCHW, return as is
    return [shape[0], shape[2], shape[3], shape[1]]

def shape_to_HW(shape):
    """ Convert from WH => HW
    """
    if len(shape) != 2:
        return shape # Not WH, return as is
    return [shape[1], 1, 1, shape[0]]

def shape_to_HWCK(shape):
    """ Convert from KCHW => HWCK
    """
    if len(shape) != 4:
        return shape # Not KCHW, return as is
    return [shape[2], shape[3], shape[1], shape[0]]

def shape_to_HWKC(shape):
    """ Convert from KCHW => HWKC
    """
    if len(shape) != 4:
        return shape # Not KCHW, return as is
    return [shape[2], shape[3], shape[0], shape[1]]

def bias(shape):
    return [1, 1, 1, shape[-1]]

def adapt_input_shape(shape):
    if hasattr(shape, 'dim'):
        shape = [dim.dim_value for dim in shape.dim]
    if len(shape) == 1:
        return bias(shape)
    if len(shape) == 2:
        return shape_to_HW(shape)
    if len(shape) == 4:
        return shape_to_NHWC(shape)
    return shape

def data_to_HW(tensor):
    """ Convert from WH => HW
    """
    return np.transpose(tensor)

def data_to_HWCK(tensor):
    """ Convert from KCHW => HWCK
    """
    return np.transpose(tensor, (2,3,1,0))
    
def data_to_HWKC(tensor):
    """ Convert from KCHW => HWKC
    """
    return np.transpose(tensor, (2,3,0,1))

def get_tensor_data(tensor):
    elems = np.product(tensor.dims)
    if tensor.raw_data:
        floats = struct.unpack('<'+str(elems)+'f', tensor.raw_data)
    elif tensor.float_data:
        floats = tensor.float_data
    elif tensor.int32_data:
        floats = np.array(tensor.int32_data, dtype=float)
    elif tensor.int64_data:
        floats = np.array(tensor.int64_data, dtype=float)
    return np.array(floats).reshape(tensor.dims)

def flatten(items,enter=lambda x:isinstance(x, list)):
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
    potentially_nested_list = [replace_with_strings.get(s) or s for s in array_of_strigs]
    return list(flatten(potentially_nested_list))


o_model = []
prev_layer_name = ''
map_ignored_layer_to_its_input = {}

input_shapes = { i.name:adapt_input_shape(i.type.tensor_type.shape) for i in model.graph.input }
model_tensors = { tensor.name:tensor for tensor in model.graph.initializer }

for layer in model.graph.node:

    name = layer.output[0] if len(layer.output) > 0 else layer.name # prefer node.output over the node.name
    class_name = layer.op_type
    inputs = layer.input or [prev_layer_name] # @TODO: Most likely should be only 'inputs = layer.input' like in Tensorflow exporter
    inputs = replace_strings_in_list(inputs, map_ignored_layer_to_its_input)

    if class_name == 'Constant':
        model_tensors[name] = get_attr(layer, 'value')
        model_tensors[name].name = name
        #print('CONST:', name, model_tensors[name].dims, struct.unpack('<'+str(np.prod(model_tensors[name].dims))+'f', model_tensors[name].raw_data))
        continue

    if args.print_layers or args.verbose:
        print("'%s' %s %s" % (name, class_name, inputs))

    if args.include_gan_layers and class_name in requires_runtime_flag:
        class_name = requires_runtime_flag[class_name]

    if class_name in known_activations:
        activation = class_name
        class_name = 'Activation'
    else:
        activation = 'Linear'
   
    if not class_name in known_classes:
        if class_name in requires_runtime_flag:
            print('SKIP:', class_name, 'layer is used only for training (use -gan flag to force inclusion in the model)')
        else:
            print('IGNORED:', class_name, 'unknown layer')
        map_ignored_layer_to_its_input[name] = inputs
        continue

    klass = known_classes[class_name]
    if type(klass) == int:
        klass = Struct(id = klass)

    o_l = Struct()
    o_l.type = klass.id
    o_l.class_name = class_name
    o_l.name = name

    axis        = axis_to_NHWC(get_attr(layer, 'axis', -1))
    auto_pad    = get_attr(layer, 'auto_pad')
    pads        = get_attr(layer, 'pads')
    strides     = get_attr(layer, 'strides')
    pool_size   = get_attr(layer, 'kernel_shape')
    shape       = get_attr(layer, 'shape')
    size        = [get_attr(layer, 'height_scale'), get_attr(layer, 'width_scale')]
    alpha       = get_attr(layer, 'alpha') or get_attr(layer, 'ratio')
    # TODO: decide what to do with 'is_test' attribute

    if auto_pad and not auto_pad in known_paddings:
        print('IGNORED: unknown padding', auto_pad)
    auto_pad = known_paddings.get(auto_pad)

    if size == [None, None]:
        size = None
    if size: size = np.array(size).astype(int).tolist()

    o_l.activation  = known_activations.get(activation) or 0
    o_l.pads        = auto_pad(pool_size) if auto_pad else pads or [0,0,0,0]
    o_l.strides     = strides or []
    o_l.pool_size   = pool_size or size or shape or []
    o_l.axis        = axis or -1
    o_l.alpha       = alpha or 1
    o_l.beta        = 0

    tensor_names = [i for i in inputs if i in model_tensors]
    inputs = [i for i in inputs if i not in model_tensors]
    o_l.tensors = [Struct(name = model_tensors[x].name, shape = model_tensors[x].dims, data = get_tensor_data(model_tensors[x]))
        for x in tensor_names]
    o_l.inputs = inputs

    # Patch shapes & data
    try:
        shapes = klass.patch_shapes([x.shape for x in o_l.tensors])

        # if we have more shapes than actual tensors,
        # then create & fill missing tensors with zeros
        in_tensor_num = len(o_l.tensors)
        for index, new_shape in enumerate(shapes):
            if index >= in_tensor_num:
                new_tensor = Struct(name = ('/model_weights/%s/%s/patch:%i') % (name, name, index-in_tensor_num),
                    shape = new_shape,
                    data = np.zeros(new_shape))
                o_l.tensors.append(new_tensor)
        assert(len(shapes) <= len(o_l.tensors))

        try:
            tensor_data = klass.patch_data([x.data for x in o_l.tensors])
            o_l.tensors = o_l.tensors[:len(tensor_data)] # resize tensor array to match patched data - patching might reduce number of tensors
            for x, data in zip(o_l.tensors, tensor_data):
                x.data = data
        except AttributeError:
            pass # no 'patch_data' lambda was specified

        # after this point we should have equal amount of shapes and tensors
        assert(len(o_l.tensors) == len(shapes))

        for x, shape in zip(o_l.tensors, shapes):
            x.shape = shape

    except AttributeError:
        # no 'patch_data' lambda was specified, op does not require tensor args
        o_l.tensors = []

    # Force all tensors to float32
    for x in o_l.tensors:
        x.data = x.data.astype(np.float32)

    # Layer is ready
    o_model.append(o_l)
    prev_layer_name = o_l.name

# Trim
if args.trim_unused_by_output:
    o_model = barracuda.trim(o_model, args.trim_unused_by_output, args.verbose)

# Find model inputs & outputs
all_layers = {l.name for l in o_model}
all_inputs = {i for l in o_model for i in l.inputs}
# global inputs - are inputs that are NOT connected to any layer in the network
# global outputs - are outputs that are NOT feeding any layer in the network
o_inputs = {i:input_shapes[i] for l in o_model for i in l.inputs if i not in all_layers}
o_outputs = [l.name for l in o_model if l.name not in all_inputs]
o_memories = []

# Compress
if args.compress_f16:
    model = barracuda.compress(model)

# Summary
barracuda.summary(o_model, o_inputs, o_outputs, o_memories,
    globals = [],
    print_layer_links = args.print_layer_links or args.verbose,
    print_barracuda_json = args.print_barracuda_json or args.verbose,
    print_tensors = args.print_tensors or args.verbose)

# Write to file
barracuda.write(o_model, o_inputs, o_outputs, o_memories, args.target_file)
print('DONE: wrote', args.target_file, 'file.')

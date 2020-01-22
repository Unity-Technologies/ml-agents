from __future__ import print_function
import numpy as np
import onnx
import struct
import barracuda
from barracuda import Struct
from google.protobuf.json_format import MessageToJson
from collections import Iterable
from pprint import pprint

if __name__ == '__main__':
    # Handle command line argumengts
    args = barracuda.parse_args(
        description = 'Convert ONNX model to Barracuda binary',
        source_extension = '.onnx',
        help = 'input ONNX serialized  file')
    # The following code can be used as an example of API used from another module
    # convert() is the main entry point for converter
    import onnx_to_barracuda as onnx2bc
    onnx2bc.convert(args.source_file, args.target_file, args.trim_unused_by_output, args)


# ONNX format:  # See: https://github.com/onnx/onnx/blob/master/onnx/onnx.proto
# ONNX schema: https://github.com/onnx/onnx/blob/master/docs/Operators.md
# ONNX conventions: https://github.com/onnx/onnx/blob/master/docs/OpConventions.md

def get_epsilon(layer):
    # default epsilon taken from https://github.com/onnx/onnx/blob/master/docs/Operators.md#BatchNormalization
    return get_attr(layer, 'epsilon', default=1e-05)

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
                    id = lambda layer:
                        21 if is_depthwise_convolution(layer) else # DepthwiseConv2D
                        20, # Conv2D
                    patch_shapes = lambda shapes: [
                        shape_to_HWCK(shapes[0]),               # K 
                        bias(shape_to_HWCK(shapes[-1]))         # B
                    ],
                    patch_data = lambda data: [
                        data_to_HWCK(data[0]),
                        data[1]
                    ]), # TODO: args(group!=0|kernel_shape.channels)
    'ConvTranspose': Struct(
                    id = 22,
                    patch_shapes = lambda shapes: [
                        shape_to_HWKC(shapes[0]),               # K 
                        bias(shape_to_HWKC(shapes[-1]))         # B
                    ],
                    patch_data = lambda data: [
                        data_to_HWKC(data[0]),
                        data[1]
                    ]), # TODO: args(output_padding, output_shape)
    'Upsample':         23, 'ResizeNearest':    23, 'Resize':    23,

    'MaxPool':          25,
    'AveragePool':      26, # TODO: args(count_include_pad)
    'GlobalMaxPool':    27,
    'GlobalAveragePool':28,
    'Pad':              29, # TODO: args(mode), implemented as Border2D
    'Crop': Struct(id = 29, # implemented as Border2D
                    patch_attrs = lambda attrs, layer: {
                        'pads': border_to_inverse_padding(layer)
                    }),
    # TODO: 3D

    'Activation':       50,
    'BatchNormalization': Struct(
        # TODO: args(epsilon)
                    id = 51, # after fusion implemented as ScaleBias
                    patch_shapes = lambda shapes: [
                        [1, 1, 1, shapes[0][0]],             # S
                        [1, 1, 1, shapes[0][0]],             # B
                    ],
                    patch_data = lambda data, layer:
                        # fuse [gamma, beta, mean, var] => [scale, bias]
                        barracuda.fuse_batchnorm_weights(data[0], data[1], data[2], data[3], get_epsilon(layer)) if len(data) == 4 else
                        # fuse [ONE, beta, mean, var] => [scale, bias]
                        barracuda.fuse_batchnorm_weights(np.ones(np.shape(data[0])), data[0], data[1], data[2], get_epsilon(layer))
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

    'Greater':          140, # TODO: args(axes)
    'GreaterEqual':     141, # TODO: args(axes)
    'Less':             142, # TODO: args(axes)
    'LessEqual':        143, # TODO: args(axes)
    'Equal':            144, # TODO: args(axes)
    'Or':               145, # TODO: args(axes)
    'And':              146, # TODO: args(axes)
    'Not':              147, # TODO: args(axes)
    'Xor':              148, # TODO: args(axes)

    'Flatten':          200,
    'Reshape':          201,
    'Transpose':        202,
    'Squeeze':          203, # TODO: args(axes)
    'Unsqueeze':        204, # TODO: args(axes)

    'Concat':           210,
    'Slice':            211, # TODO: args(axes, ends, starts)
    'Tile':             212  # TODO: args(repeats)
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
    'VALID' : [0,0,0,0],
    'SAME_UPPER'  : [-1],
    'SAME_LOWER'  : [-2],
}

# Helper
def embody(v, default=0):
    return default if v is None else v

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

def is_depthwise_convolution(layer):
    # https://github.com/onnx/onnx/blob/master/docs/Operators.md#conv
    # depthwise convolutions have multiple channel groups (in_channels == group_count)
    return get_attr(layer, 'group', default=1) > 1

def border_to_inverse_padding(layer):
    border = get_attr(layer, 'border', np.ones(4)) # 'border' is (leftBorder, topBorder, rightBorder, bottomBorder).
    height, width = get_attr(layer, 'scale', np.ones(2)) # 'scale' is (height, width)
    return [-border[0] * width, -border[1] * height, -border[2] * width, -border[3] * height]

def axis_to_NHWC(axis):
    """ Convert from NCHW => NHWC
    """
    assert(axis >= -1 and axis < 4)
    return [-1, 0, 3, 1, 2][axis+1];

def shape_to_NHWC(shape):
    """ Convert from NCHW|NCW => NHWC
    """
    if len(shape) == 4: # NCHW => NHWC
        return [shape[0], shape[2], shape[3], shape[1]]
    if len(shape) == 3: # NCW => N1WC
        return [shape[0], 1, shape[2], shape[1]]
    return shape # Not NCHW, return as is

def shape_to_HW(shape):
    """ Convert from WH => HW
    """
    if len(shape) != 2:
        return shape # Not WH, return as is
    return [shape[1], 1, 1, shape[0]]

def shape_to_HWCK(shape):
    """ Convert from KCHW|KCW => HWCK
    """
    if len(shape) == 4: # KCHW => HWCK
        return [shape[2], shape[3], shape[1], shape[0]]
    if len(shape) == 3: # KCW => 1WCK
        return [1, shape[2], shape[1], shape[0]]
    return shape # Not KCHW, return as is

def shape_to_HWKC(shape):
    """ Convert from KCHW|KCW => HWKC
    """
    if len(shape) == 4: # KCHW => HWKC
        return [shape[2], shape[3], shape[0], shape[1]]
    if len(shape) == 3: # KCW => 1WKC
        return [1, shape[2], shape[0], shape[1]]
    return shape # Not KCHW, return as is

def bias(shape):
    return [1, 1, 1, shape[-1]]

def adapt_input_shape(shape):
    if hasattr(shape, 'dim'):
        shape = [dim.dim_value for dim in shape.dim]
    if len(shape) == 1:
        return bias(shape)
    if len(shape) == 2:
        return shape_to_HW(shape)
    if len(shape) == 3 or len(shape) == 4:
        return shape_to_NHWC(shape)
    return shape

def data_to_HW(tensor):
    """ Convert from WH => HW
    """
    return np.transpose(tensor)

def data_to_HWCK(tensor):
    """ Convert from KCHW|KCW => HWCK
    """
    if len(tensor.shape) == 3:
        tensor = np.expand_dims(tensor, axis=2)
    return np.transpose(tensor, (2,3,1,0))
    
def data_to_HWKC(tensor):
    """ Convert from KCHW|KCW => HWKC
    """
    if len(tensor.shape) == 3:
        tensor = np.expand_dims(tensor, axis=2)
    return np.transpose(tensor, (2,3,0,1))

def get_tensor_data(tensor):
    elems = np.product(tensor.dims)
    if tensor.raw_data:
        floats = struct.unpack('<'+str(int(elems))+'f', tensor.raw_data)
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

#########################################################

class ModelBuilderContext:
    def __init__(self):
        self.layers = []
        self.input_shapes = {}
        self.model_tensors = {}
        self.model_memories = []
        self.map_ignored_layer_to_its_input = {}

def process_model(model, args):
    o_context = ModelBuilderContext()

    o_context.input_shapes = { i.name:adapt_input_shape(i.type.tensor_type.shape) for i in model.graph.input }
    o_context.model_tensors = { tensor.name:tensor for tensor in model.graph.initializer }

    for layer in model.graph.node:
        process_layer(layer, o_context, args)
   
    def find_unconnected_model_tensors(layers, const_tensors):
        const_tensors = const_tensors.copy()
        for l in layers:
            for i in l.input:
                const_tensors.pop(i, None)
        return list(const_tensors.keys())

    return o_context.layers, o_context.input_shapes, o_context.model_tensors, o_context.model_memories, \
        find_unconnected_model_tensors(model.graph.node, o_context.model_tensors)

def process_layer(layer, context, args):
    model_tensors = context.model_tensors
    map_ignored_layer_to_its_input = context.map_ignored_layer_to_its_input

    name = layer.output[0] if len(layer.output) > 0 else layer.name # prefer node.output over the node.name
    class_name = layer.op_type
    inputs = layer.input # ONNX inputs are always explicit, but in case of Keras we had 'inputs = layer.input or [prev_layer_name]'
    inputs = replace_strings_in_list(inputs, map_ignored_layer_to_its_input)

    if class_name == 'Constant':
        model_tensors[name] = get_attr(layer, 'value')
        model_tensors[name].name = name
        #print('CONST:', name, model_tensors[name].dims, struct.unpack('<'+str(np.prod(model_tensors[name].dims))+'f', model_tensors[name].raw_data))
        return

    if args.print_layers or args.verbose:
        print("'%s' %s %s" % (name, class_name, inputs))

    if class_name in known_activations:
        activation = class_name
        class_name = 'Activation'
    else:
        activation = 'Linear'
   
    if not class_name in known_classes:
        if class_name in requires_runtime_flag:
            print('SKIP:', class_name, 'layer is used only for training')
        else:
            print('IGNORED:', class_name, 'unknown layer')
        map_ignored_layer_to_its_input[name] = inputs
        return

    klass = known_classes[class_name]
    if type(klass) == int:
        klass = Struct(id = klass)

    o_l = Struct()
    o_l.type = klass.id(layer) if callable(klass.id) else klass.id
    o_l.class_name = class_name
    o_l.name = name

    axis        = axis_to_NHWC(get_attr(layer, 'axis', -1))
    auto_pad    = get_attr(layer, 'auto_pad')
    pads        = get_attr(layer, 'pads')
    strides     = get_attr(layer, 'strides')
    pool_size   = get_attr(layer, 'kernel_shape')
    shape       = get_attr(layer, 'shape')
    starts      = get_attr(layer, 'starts')
    ends        = get_attr(layer, 'ends')
    slice_strides = [1,1,1,1] if starts and ends else []
    #TODO properly extract scale from const Tensor for Upsample layers
    size        = [get_attr(layer, 'height_scale'), get_attr(layer, 'width_scale')] if get_attr(layer, 'width_scale') and class_name == 'Upsample' else [2,2]
    alpha       = get_attr(layer, 'alpha') or get_attr(layer, 'ratio') or get_attr(layer, 'value')
    beta        = get_attr(layer, 'beta') or get_attr(layer, 'epsilon')
    # TODO: decide what to do with 'is_test' attribute

    if auto_pad and not auto_pad in known_paddings:
        print('IGNORED: unknown padding', auto_pad)

    if size == [None, None]:
        size = None
    if size: size = np.array(size).astype(int).tolist()

    o_l.activation  = known_activations.get(activation) or 0
    o_l.pads        = known_paddings.get(auto_pad) if auto_pad else pads or starts or [0,0,0,0]
    o_l.strides     = strides or slice_strides or []
    o_l.pool_size   = pool_size or size or shape or ends or []
    o_l.axis        = embody(axis, default=-1)
    o_l.alpha       = embody(alpha, default=1)
    o_l.beta        = beta or 0


    # Patch shapes & data
    try:
        tensor_names = [i for i in inputs if i in model_tensors]
        o_l.tensors = [
            Struct(name=model_tensors[x].name, shape=model_tensors[x].dims, data=get_tensor_data(model_tensors[x]))
            for x in tensor_names]
        o_l.inputs = [i for i in inputs if i not in model_tensors]

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

        if hasattr(klass, 'patch_data'):
            data = [x.data for x in o_l.tensors]

            patch_data_fn = klass.patch_data
            patch_data_expected_arg_count = patch_data_fn.__code__.co_argcount
            patch_data_args = (data, layer) if patch_data_expected_arg_count > 1 else (data,)
            tensor_data = patch_data_fn(*patch_data_args)
            o_l.tensors = o_l.tensors[:len(tensor_data)] # resize tensor array to match patched data - patching might reduce number of tensors
            for x, data in zip(o_l.tensors, tensor_data):
                x.data = data

        # after this point we should have equal amount of shapes and tensors
        assert(len(o_l.tensors) == len(shapes))

        for x, shape in zip(o_l.tensors, shapes):
            x.shape = shape

    except AttributeError:
        # no 'patch_data' lambda was specified, op does not require tensor args
        o_l.tensors = []
        o_l.inputs = inputs

    try:
        attrs = klass.patch_attrs(o_l, layer)
        for k, v in attrs.items():
            o_l.__dict__[k] = v
    except AttributeError:
        pass # no 'patch_attrs' lambda was specified

    # Force all tensors to float32
    for x in o_l.tensors:
        x.data = x.data.astype(np.float32)

    # Layer is ready
    context.layers.append(o_l)

#########################################################

def convert(source_file, target_file, trim_unused_by_output="", verbose=False, compress_f16=False):
    """
    Converts a ONNX model into a Barracuda model.
    :param source_file: The ONNX Model
    :param target_file: The name of the file the converted model will be saved to
    :param trim_unused_by_output: The regexp to match output nodes to remain in the model. All other uconnected nodes will be removed.
    :param verbose: If True, will display debug messages
    :param compress_f16: If true, the float values will be converted to f16
    :return:
    """
    if (type(verbose)==bool):
        args = Struct()
        args.verbose = verbose
        args.print_layers = verbose
        args.print_source_json = verbose
        args.print_barracuda_json = verbose
        args.print_layer_links = verbose
        args.print_patterns = verbose
        args.print_tensors = verbose
    else:
        args = verbose

    if args.print_supported_ops:
        barracuda.print_known_operations(known_classes, known_activations)

    # Load ONNX model
    print("Converting %s to %s" % (source_file, target_file))
    i_model = onnx.load(source_file)

    if args.print_source_json or args.verbose:
        for layer in i_model.graph.node:
            print(MessageToJson(layer) + ",")

    # Convert
    o_model = barracuda.Model()
    o_model.layers, o_input_shapes, o_model.tensors, o_model.memories, o_model.globals = \
        process_model(i_model, args)

    # Trim
    if trim_unused_by_output:
        o_model.layers = barracuda.trim(o_model.layers, trim_unused_by_output, args.verbose)

    # Create load layers for constants
    def dims_to_barracuda_shape(tensor):
        if hasattr(tensor, 'dims') and len(tensor.dims) > 0:
            return adapt_input_shape(tensor.dims)
        return [1,1,1,1]

    barracuda.setup_constants(o_model,
        lambda tensor: dims_to_barracuda_shape(tensor),
        lambda tensor: get_tensor_data(tensor))

    # Find model inputs & outputs
    all_inputs = {i for l in o_model.layers for i in l.inputs}
    all_layers = {l.name for l in o_model.layers}

    # global inputs - are inputs that are NOT connected to any layer in the network
    # global outputs - are outputs that are NOT feeding any layer in the network
    o_model.inputs = {i:o_input_shapes[i] for l in o_model.layers for i in l.inputs if i not in all_layers}

    def is_output_layer(layer):
        if layer.name in all_inputs:  # Only layers that do not input to other layers can count as global output
            return False
        if layer.name in o_model.globals:
            return False
        return True
    o_model.outputs = [l.name for l in o_model.layers if is_output_layer(l)]

    # Compress
    if compress_f16:
        o_model = barracuda.compress(o_model)

    # Sort model so that layer inputs are always ready upfront
    o_model.layers = barracuda.sort(o_model.layers, o_model.inputs, o_model.memories, args.verbose)
    o_model.layers = barracuda.fuse(o_model.layers, args.verbose)

    # Summary
    barracuda.summary(o_model,
        print_layer_links = args.print_layer_links or args.verbose,
        print_barracuda_json = args.print_barracuda_json or args.verbose,
        print_tensors = args.print_tensors or args.verbose)

    # Write to file
    barracuda.write(o_model, target_file)
    print('DONE: wrote', target_file, 'file.')

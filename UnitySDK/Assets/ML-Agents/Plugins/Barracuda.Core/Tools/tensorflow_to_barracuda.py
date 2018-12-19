from __future__ import print_function
import numpy as np
import struct # convert from Python values and C structs
import tensorflow as tf
import re
import barracuda
from google.protobuf import descriptor
from google.protobuf.json_format import MessageToJson


'''
[LSTM Full]
- it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
- ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
- ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
- Ct = ft (.) Ct-1 + it (.) ct
- ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
- Ht = ot (.) h(Ct)

[LSTM No peephole]
- it = f(Xt * Wi.T + Ht_ * Ri.T + Wbi + Rbi)
- ft = f(Xt * Wf.T + Ht_ * Rf.T + Wbf + Rbf)
- ct = g(Xt * Wc.T + Ht_ * Rc.T + Wbc + Rbc)
- Ct =   ft . Ct_  + it . ct
- ot = f(Xt * Wo.T + Ht_ * Ro.T + Wbo + Rbo)
- Ht =   ot . h(Ct)
'''

# TODO: support more than 1 LSTM layer per model - prepend scope to names and inputs
# TODO: support different activation functions in LSTM
# TODO: implement defaults for Dense, Conv, ScaleBias nodes
# TODO: strip output Identity node, instead patch upstream layer names
# TODO: use ScaleBias and Pow with alpha when input is constant Tensor
# TODO: support all data format types (curretly only NHWC)
# TODO: support all data types (currently only FLOAT, INT32, BOOL)
# TODO: strip output Identity node, instead patch upstream layer names
# TODO: implement layers DepthwiseConv2dNative, FusedResizeAndPadConv2D

# Command line argumengts
args = barracuda.parse_args(
    description = 'Convert Tensorflow model to Barracuda binary',
    source_extension = '.pb',
    help = 'input Tensorflow serialized .pb file')

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

def basic_lstm(nodes, inputs, tensors):
    assert(len(inputs) == 2)

    def find_tensor_by_name(name):
        nonlocal tensors
        candidates = [t for t in tensors if t.name.endswith(name)]
        return candidates[0] if candidates else None

    def find_forget_bias(nodes, inputs, tensors):
        # TODO: make it more fault-tolerant
        # search for scalar float constant that is input to Add node
        # and hope it is not a constant for some complex activation function
        for t in tensors:
            if np.prod(t.shape) == 1 and get_tensor_dtype(t.obj) == "DT_FLOAT":
                for n in nodes:
                    if n.op == 'Add' and t.name in n.input:
                        return t
        return np.zeros(1)

    input = inputs[-1]
    state_c = inputs[0] + '_c'
    state_h = inputs[0] + '_h'
    kernel = find_tensor_by_name('/kernel')
    bias = find_tensor_by_name('/bias') or np.zeros(np.shape(kernel)[-1])
    forget_bias = find_forget_bias(nodes, inputs, tensors)
    new_state_c = nodes[-1].name + '_c'
    new_state_h = nodes[-1].name + '_h'

    assert(np.shape(kernel)[-1] == np.shape(bias)[-1])
    num_gates = 4
    seq_length = 1
    hidden_size = np.shape(kernel)[-1] // num_gates

    new_layers = basic_lstm_impl('lstm', input, state_c, state_h,
        kernel.data, bias.data, forget_bias.data,
        new_state_c, new_state_h, num_gates)

    global o_memories
    state_shape = [1, 1, seq_length, hidden_size]
    o_memories += [state_shape, state_c, new_state_c]
    o_memories += [state_shape, state_h, new_state_h]

    # map exptected output of the replaced pattern to output from our LSTM cell
    global map_ignored_layer_to_its_input
    actual_output_node = nodes[-4]
    assert(actual_output_node.op == 'Reshape')
    map_ignored_layer_to_its_input[actual_output_node.name] = new_state_h

    return new_layers


def basic_lstm_impl(name, input, state_c, state_h, kernel, bias, forget_bias, new_state_c, new_state_h, number_of_gates = 4):
    def concat(a, b, out):
        return Struct(name=out, op='Concat', input=[a, b])
    def mul(a, b, out):
        return Struct(name=out, op='Mul', input=[a,b])
    def add(a, b, out):
        return Struct(name=out, op='Add', input=[a,b])
    def mad(x, kernel, bias, out):
        return Struct(name=out, op='Dense', input=[x, kernel, bias])
    def sigmoid(x, out):
        return Struct(name=out, op='Sigmoid', input=[x])
    def tanh(x, out):
        return Struct(name=out, op='Tanh', input=[x])

    kernels = np.split(kernel, number_of_gates, axis=-1)
    biases = np.split(bias, number_of_gates, axis=-1)

    model_tensors['kernel_i'] = kernels[0]
    model_tensors['kernel_j'] = kernels[1]
    model_tensors['kernel_f'] = kernels[2]
    model_tensors['kernel_o'] = kernels[3]
    model_tensors['bias_i'] = biases[0]
    model_tensors['bias_j'] = biases[1]
    model_tensors['bias_f'] = biases[2] + forget_bias
    model_tensors['bias_o'] = biases[3]

    # TODO: prepend scope to names and inputs

    return [
        concat(input, state_h, 'inputs'),
        mad('inputs', 'kernel_i', 'bias_i', 'i'),
        mad('inputs', 'kernel_j', 'bias_j', 'j'),
        mad('inputs', 'kernel_f', 'bias_f', 'f'),
        mad('inputs', 'kernel_o', 'bias_o', 'o'),
        sigmoid('i', 'i_act'),
        sigmoid('f', 'f_act'),
        sigmoid('o', 'o_act'),
        tanh('j', 'j_act'),
        mul(state_c, 'f_act', 'temp_1'),
        mul('i_act', 'j_act', 'temp_2'),
        add('temp_1', 'temp_2', new_state_c),
        tanh(new_state_c, 'new_c_act'),
        mul('o_act', 'new_c_act', new_state_h),
    ]

# Important ProtoBuf definitions:
#    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/types.proto
#    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto
#    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/node_def.proto
#
# Node descriptions:
#    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/nn_ops.cc
#    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/math_ops.cc
#
# Class doc:
#    https://www.tensorflow.org/api_docs/cc/
#
known_classes = {

    'Dense': Struct(
                    id = 1,
                    out_shapes = lambda shapes: [
                        [shapes[0][0], 1, 1, shapes[0][1]],  # W
                        [1, 1, 1, shapes[-1][-1]]            # B
                    ],
                    patch_data = lambda data: [
                        data[0],
                        data[1]
                    ]),
    'MatMul': Struct(
                    id = 1,
                    out_shapes = lambda shapes: [
                        [shapes[0][0], 1, 1, shapes[0][1]],  # W
                        [1, 1, 1, shapes[0][1]]              # B
                    ],
                    patch_data = lambda data: [
                        data[0],
                        np.zeros(np.shape(data[1]))
                    ]),
    'BiasAdd': Struct(
                    id = 51, # implemented as ScaleBias
                    out_shapes = lambda shapes: [
                        [1, 1, 1, shapes[0][0]],             # ONE
                        [1, 1, 1, shapes[0][0]],             # B
                    ],
                    patch_data = lambda data: [
                        np.ones(np.shape(data[0])),
                        data[0]
                    ]),

    # NHWC
    # TODO: NCHW
    'Conv2D': Struct( # TODO: implement defaults = add_zero_bias_as_2nd_arg from Keras exporter and get rid of separate Conv2D & ^Conv2D
                    id = 20,
                    out_shapes = lambda shapes: [
                        shapes[0],                           # K
                        [1, 1, 1, shapes[0][-1]]             # B
                    ],
                    patch_data = lambda data: [
                        data[0],
                        np.zeros(np.shape(data[1]))
                    ]),
    '^Conv2D': Struct(
                    id = 20,
                    out_shapes = lambda shapes: [
                        shapes[0],                           # K
                        [1, 1, 1, shapes[-1][-1]]            # B
                    ],
                    patch_data = lambda data: [
                        data[0],
                        data[1]
                    ]),
    'Conv2DBackpropInput': Struct( # Conv2DTranspose
                    id = 22,
                    out_shapes = lambda shapes: [
                        shapes[0],                           # K
                        [1, 1, 1, shapes[0][-1]]             # B
                    ],
                    patch_data = lambda data: [
                        data[0],
                        np.zeros(np.shape(data[1]))
                    ]),
    '^Conv2DBackpropInput': Struct( # Conv2DTranspose
                    id = 22,
                    out_shapes = lambda shapes: [
                        shapes[0],                           # K
                        [1, 1, 1, shapes[-1][-1]]            # B
                    ],
                    patch_data = lambda data: [
                        data[0],
                        data[1]
                    ]),

    # TODO: 3D

    'ResizeNearestNeighbor':
                        23, # implemented as Upsample2D
    'ResizeBilinear':   23, # implemented as Upsample2D
    'ResizeBicubic':    23, # implemented as Upsample2D
    'MaxPool':          25,
    'AvgPool':          26,

    'GlobalAveragePool':28,

    'Activation':       50,

    'BatchNormalization': Struct(
                    id = 51, # after fusion implemented as ScaleBias
                    out_shapes = lambda shapes: [
                        [1, 1, 1, shapes[0][0]],             # S
                        [1, 1, 1, shapes[0][0]],             # B
                    ],
                    patch_data = lambda data:
                        # fuse [gamma, beta, mean, var] => [scale, bias]
                        fuse_batchnorm_weights(data[0], data[1], data[2], data[3]) if len(data) == 5 else
                        # fuse [ONE, beta, mean, var] => [scale, bias]
                        fuse_batchnorm_weights(np.ones(np.shape(data[0])), data[0], data[1], data[2] + data[3])
                    ),
    'FusedBatchNorm': Struct(
                    id = 51, # after fusion implemented as ScaleBias
                    out_shapes = lambda shapes: [
                        [1, 1, 1, shapes[0][0]],             # S
                        [1, 1, 1, shapes[0][0]],             # B
                    ],
                    patch_data = lambda data:
                        # fuse [gamma, beta, mean, var] => [scale, bias]
                        fuse_batchnorm_weights(data[0], data[1], data[2], data[3])
                    ),
    'LRN':              53,

    'RandomStandardNormal':
                        64,
    'RandomUniform':    65,
    'Multinomial':      66,
    'OneHot':           67,

    # Broadcast ops
    'Add':              100,
    'Sub':              101,
    'Mul':              102,
    'RealDiv':          103,
    'Pow':              104,
    'Minimum':          110,
    'Maximum':          111,

    # Reduce ops
    'Max':              124,
    'Mean':             125,
    'Min':              126,
    'Prod':             127,
    'Sum':              128,

    'Flatten':          200,
    'Reshape':          201,
    'Concat':           210,
    'StridedSlice':     211,
}

requires_runtime_flag = {
    'Dropout' : 'DropoutRuntime',
    'BatchNormalization' : 'BatchNormalizationRuntime',
}

known_activations = {
    'Linear' : 0,
    'Relu' : 1,
    'Softmax' : 2,
    'Tanh' : 3,
    'Sigmoid' : 4,
    'Elu' : 5,
    'Relu6' : 6,
    'LeakyRelu' : 7,
    'Selu' : 8,
    'Swish' : 9,

    'LogSoftmax' : 10,
    'Softplus' : 11,
    'Softsign' : 12,

    'Abs' : 100,
    'Neg' : 101,
    'Ceil' : 102,
    'Floor' : 104,

    'Sqrt' : 111,
    'Exp' : 113,
    'Log' : 114,

    'Acos' : 200,
    'Acosh' : 201,
    'Asin' : 202,
    'Asinh' : 203,
    'Atan' : 204,
    'Atanh' : 205,
    'Cos' : 206,
    'Cosh' : 207,
    'Sin' : 208,
    'Sinh' : 209,
    'Tan' : 210
}

known_paddings = {
	'VALID' : lambda kernel: [0,0,0,0],
	'SAME'  : lambda kernel: np.array([np.array(kernel)//2, np.array(kernel)//2]).flatten().tolist()
}

supported_data_formats = {
    'NHWC'
}

known_patterns = {
    # TODO: Flatten pattern using namespace regexp
    repr(['Shape', 'StridedSlice', 'Pack', 'Reshape'])          : "Flatten",
    repr(['Shape', 'StridedSlice', 'Prod', 'Pack', 'Reshape'])  : "Flatten",
    repr(['Shape', 'Slice', 'Slice', 'Prod',
        'ExpandDims', 'ConcatV2', 'Reshape'])                   : "Flatten",
    repr(['Const', 'Reshape'])                                  : 'Reshape',

    repr(['Add', 'Rsqrt', 'Mul', 'Mul', 'Sub', 'Add'])          : 'BatchNormalization',
    repr(['Add', 'Rsqrt', 'Mul', 'Mul', 'Mul', 'Sub', 'Add'])   : 'BatchNormalization',

    repr(['ConcatV2'])                                          : 'ConcatV2',
    repr(['Mean'])                                              : 'Mean',
    repr(['Multinomial'])                                       : 'Multinomial',
    repr(['OneHot'])                                            : 'OneHot',
    repr(['Square'])                                            : 'Square',

    repr(['MatMul', 'BiasAdd'])                                 : 'Dense',
    repr(['Conv2D', 'BiasAdd'])                                 : '^Conv2D',
    repr(['Conv2DBackpropInput', 'BiasAdd'])                    : '^Conv2DBackpropInput',



    repr(['Pack', 'Reshape'])                                   : 'Flatten$',    # for now we assume that this combination is trivial Flatten
                                                                                 # for exmaple it is used in ML-agents LSTM nets with sequence_length==1

    repr(['StridedSlice', 'Reshape',
        re.compile('^lstm/'),
        'Reshape', 'ConcatV2', 'Identity'])                     : 'BasicLSTM',

    repr([re.compile('^lstm/'),
        'Reshape', 'ConcatV2', 'Identity'])                     : 'BasicLSTM',

    repr(['Sigmoid', 'Mul'])  : "Swish",

    # TODO: FusedResizeAndPadConv2D
}

def by_name(args, name):
    for a in args:
        if a.name.endswith(name):
            return a

transform_patterns = {
    'Flatten' : lambda nodes, inputs, tensors:
        Struct(
            op    = 'Flatten',
            input = inputs
        ),
    'Flatten$' : lambda nodes, inputs, tensors:
        Struct(
            op    = 'Flatten',
            input = [inputs[-1]] # take only the last input, assume all other arguments are trivial (like sequence_length==1 always in ML-agents LSTM nets)
        ),
    'Reshape' : lambda nodes, inputs, tensors:
        Struct(
            op    = 'Reshape',
            input = inputs,
            ksize = [tensors[0].data[0], tensors[0].data[1], tensors[0].data[2], tensors[0].data[3]]
        ),
    'Multinomial' : lambda nodes, inputs, tensors:
        Struct(
            op    = 'Multinomial',
            input = inputs,
            ksize = [int(tensors[0].data[0])],
        ),
    'OneHot' : lambda nodes, inputs, tensors:
        Struct(
            op    = 'OneHot',
            input = inputs,
            ksize = [int(by_name(tensors, '/depth').data[0])],
            alpha = by_name(tensors, '/on_value').data[0],
            beta  = by_name(tensors, '/off_value').data[0],
        ),
    'Square' : lambda nodes, inputs, tensors:
        Struct(
            op    = 'Mul',
            input = [i for i in inputs] + [i for i in inputs], # input * input
        ),
    'ConcatV2' : lambda nodes, inputs, tensors:
        Struct(
            op    = 'Concat',
            input = inputs,

            # TEMPORARY: until we implemented rank detection and axis remapping (hopefully in exporter)
            # HACK: assume Concat is always for last channel
            axis = int(-1)
            #axis = int(tensors[-1].data[0]) # expect last tensor to hold Axis index
        ),
    'BatchNormalization' : lambda nodes, inputs, tensors:
        Struct(
            op    = 'BatchNormalization',
            input = [i for i in inputs] + [t.name for t in tensors],
            data_frmt = get_attr(nodes[-2], 'data_format'),
        ),
    'Mean' : lambda nodes, inputs, tensors:
        Struct(
            # TODO: use data_frmt of the input instead of hardcoded [1,2] for HW
            op    = 'GlobalAveragePool' if np.array_equal(tensors[0].data, [1,2]) else 'MeanWithUnsupportedReductionTensor',
            input = [i for i in inputs],
        ),
    'Dense' : lambda nodes, inputs, tensors:
        Struct(
            op    = 'Dense',
            input = [i for i in inputs] + [t.name for t in tensors],
            data_frmt = get_attr(nodes[-2], 'data_format'),
        ),
    '^Conv2D' : lambda nodes, inputs, tensors:
        Struct(
            op    = '^Conv2D',
            input = [i for i in inputs] + [t.name for t in tensors],
            padding = get_attr(nodes[-2], 'padding'),
            strides = get_attr(nodes[-2], 'strides'),
            dilations = get_attr(nodes[-2], 'dilations'),
            data_frmt = get_attr(nodes[-2], 'data_format'),
        ),
    '^Conv2DBackpropInput' : lambda nodes, inputs, tensors:
        Struct(
            op    = '^Conv2DBackpropInput',
            input = [i for i in inputs] + [t.name for t in tensors],
            padding = get_attr(nodes[-2], 'padding'),
            strides = get_attr(nodes[-2], 'strides'),
            dilations = get_attr(nodes[-2], 'dilations'),
            data_frmt = get_attr(nodes[-2], 'data_format'),
        ),
    'BasicLSTM' : lambda nodes, inputs, tensors:
        basic_lstm(nodes, inputs, tensors),

    'Swish' : lambda nodes, inputs, tensors:
            Struct(
                op    = 'Swish',
                input = inputs
            ),

    # TODO:'Round'
    # TODO:'Rsqrt'
}

# Load
print("Converting %s to %s" % (args.source_file, args.target_file))
f = open(args.source_file, 'rb')
model = tf.GraphDef()
model.ParseFromString(f.read())

if args.verbose:
    print('OP_TYPES:', {layer.op for layer in model.node})

if args.print_source_json or args.verbose:
    for layer in model.node:
        if not layer.op == 'Const':
            print('MODEL:', MessageToJson(layer) + ",")

# Parse
def get_attr(node, attr_name, default=None):
    if type(node) == Struct:
        if hasattr(node, attr_name):
            return getattr(node, attr_name)
        else:
            return None

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


def get_layer_shape(layer):
    shape = get_attr(layer, 'shape')
    if not shape:
        return [-1, -1, -1, -1]
    shape = [dim.size for dim in shape.dim]
    if len(shape) == 1:
        return [1, 1, 1, shape[0]]
    if len(shape) == 2:
        return [shape[0], 1, 1, shape[1]]
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

    dataType = ''
    fields = tensor.ListFields()

    for field, value in fields:
        if field.name == 'dtype' and field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_ENUM:
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
            data = struct.unpack('<'+str(elems)+'f', tensor.tensor_content)
        elif dataType == "DT_INT32":
            data = struct.unpack('<'+str(elems)+'i', tensor.tensor_content)
        elif dataType == "DT_BOOL":
            data = struct.unpack('<'+str(elems)+'?', tensor.tensor_content)
        else:
            print('UNSUPPORTED: data type', dataType)
    if tensor.float_val:
        data = tensor.float_val
    if tensor.int_val:
        data = np.array(tensor.int_val, dtype=float)
    if tensor.bool_val:
        data = np.array(tensor.bool_val, dtype=float)
    return np.array(data).reshape(dims)

def pool_to_HW(shape, data_frmt):
    """ Convert from NHWC|NCHW => HW
    """
    if len(shape) != 4:
        return shape # Not NHWC|NCHW, return as is
    if data_frmt == 'NCHW':
        return [shape[2], shape[3]]
    return [shape[1], shape[2]]

def strides_to_HW(shape, format):
    return pool_to_HW(shape, format)

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

def remove_duplicates_from_list(array):
    "Preserves the order of elements in the list"
    output = []
    unique = set()
    for a in array:
        if a not in unique:
            unique.add(a)
            output.append(a)
    return output

o_model = []
o_memories = []
o_globals = []
map_ignored_layer_to_its_input = {}
input_shapes = {}
model_tensors = {}

def process_layer(layer):
    global o_model
    global map_ignored_layer_to_its_input
    global model_tensors

    name = layer.name
    class_name = layer.op
    inputs = layer.input # Tensorflow inputs are always explicit, but in case of Keras we had 'inputs = layer.input or [prev_layer_name]'
    inputs = replace_strings_in_list(inputs, map_ignored_layer_to_its_input)

    if class_name == 'Const':
        model_tensors[name] = layer.attr["value"].tensor
        return

    if class_name == 'Placeholder':
        assert(inputs == [])
        map_ignored_layer_to_its_input[name] = inputs
        input_shapes[name] = get_layer_shape(layer)
        return

    if class_name == 'Identity':
        connected_to_const = len(inputs) == 1 and inputs[0] in model_tensors
        if connected_to_const:
            map_ignored_layer_to_its_input[name] = inputs
            return
        else:
            # treat Identity layer that are connected to processing nodes
            # as output from the network
            class_name = 'Linear'

    # TEMPORARY: until we implemented rank detection and StidedSlice at runtime
    # HACK: skips trivial StridedSlices for rank=2 tensors
    if class_name == 'StridedSlice' and get_attr(layer, 'begin_mask') == 1 and get_attr(layer, 'end_mask') == 1:
        map_ignored_layer_to_its_input[name] = inputs[0]
        return

    if args.print_layers or args.verbose:
        var_tensors = [i for i in inputs if i not in model_tensors]
        const_tensors = [i for i in inputs if i in model_tensors]
        print("'%s' %s Vars:%s Const:%s" % (name, class_name, var_tensors, const_tensors))

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
        return

    klass = known_classes[class_name]
    if type(klass) == int:
        klass = Struct(id = klass)

    o_l = Struct()
    o_l.type = klass.id
    o_l.class_name = class_name
    o_l.name = name

    padding     = get_attr(layer, 'padding') # layer.attr['padding'].s.decode("utf-8")
    strides     = get_attr(layer, 'strides') # layer.attr['strides'].list.i
    dilations   = get_attr(layer, 'dilations') # layer.attr['dilations'].list.i
    pool_size   = get_attr(layer, 'ksize') # layer.attr['ksize'].list.i
    data_frmt   = get_attr(layer, 'data_format') # layer.attr['data_format'].s.decode("utf-8")
    axis        = get_attr(layer, 'axis')
    alpha       = get_attr(layer, 'alpha')
    beta        = get_attr(layer, 'beta')

    if activation and not activation in known_activations:
        print('IGNORED: unknown activation', activation)
    if padding and not padding in known_paddings:
        print('IGNORED: unknown padding', padding)
    if data_frmt and not data_frmt in supported_data_formats:
        print('UNSUPPORTED: data format', data_frmt)

    o_l.activation  = known_activations.get(activation) or 0
    o_l.pads        = [] # initialized later once kernel shape is known, keep it here for stable order in json output
    o_l.strides     = strides_to_HW(strides, data_frmt) if strides else []
    o_l.pool_size   = pool_to_HW(pool_size, data_frmt) if pool_size else []
    o_l.axis        = axis or -1
    o_l.alpha       = alpha or 1
    o_l.beta        = beta or 0

    tensor_names = [i for i in inputs if i in model_tensors]
    o_l.tensors = [Struct(name = x, shape = get_tensor_dims(model_tensors[x]), data = get_tensor_data(model_tensors[x]))
        for x in tensor_names]
    # Patch shapes & data
    layer_has_model_tensors = len(o_l.tensors) > 0
    if hasattr(klass, 'out_shapes') and layer_has_model_tensors:
        shapes = klass.out_shapes([x.shape for x in o_l.tensors])

        # if we have more shapes than actual tensors,
        # then create & fill missing tensors with zeros
        in_tensor_num = len(o_l.tensors)
        for index, new_shape in enumerate(shapes):
            if index >= in_tensor_num:
                new_tensor = Struct(name = ('%s/patch:%i') % (name, index-in_tensor_num),
                    shape = new_shape,
                    data = np.zeros(new_shape))
                o_l.tensors.append(new_tensor)
        assert(len(shapes) <= len(o_l.tensors))

        if hasattr(klass, 'patch_data'):
            tensor_data = klass.patch_data([x.data for x in o_l.tensors])
            o_l.tensors = o_l.tensors[:len(tensor_data)] # resize tensor array to match patched data - patching might reduce number of tensors
            for x, data in zip(o_l.tensors, tensor_data):
                x.data = data

        # after this point we should have equal amount of shapes and tensors
        assert(len(o_l.tensors) == len(shapes))

        for x, shape in zip(o_l.tensors, shapes):
            x.shape = shape

        o_l.inputs = [i for i in inputs if i not in model_tensors]

    else:
        # no 'patch_data' lambda was specified, op does not require tensor args
        o_l.tensors = []
        o_l.inputs = inputs

    # Force all tensors to float32
    for x in o_l.tensors:
        x.data = x.data.astype(np.float32)

    # Calculate padding in pixels - kernel shape is known by now
    padding = known_paddings.get(padding)
    o_l.pads = padding(o_l.pool_size or o_l.tensors[0].shape[1::-1]) if padding else [0,0,0,0]

    # Layer is ready
    o_model.append(o_l)


# Find node patterns
nodes_as_array = [node for node in model.node]

node_index = 0
while node_index < len(nodes_as_array):
    node = nodes_as_array[node_index]
    match = False
    for pattern_repr, pattern_name in known_patterns.items():
        pattern = eval(pattern_repr)
        if node_index + len(pattern) > len(nodes_as_array):
            continue # pattern too long, skip

        require_exact_match = (pattern[0] == 'Const' or pattern[0] == 'Identity')
        pattern_end = node_index

        def match_node(node, pattern):
            return node.op == pattern or (hasattr(pattern, 'match') and pattern.match(node.name))

        for p in pattern:
            if not require_exact_match:
                while pattern_end < len(nodes_as_array) and nodes_as_array[pattern_end].op != p and (
                    nodes_as_array[pattern_end].op == 'Const' or
                    nodes_as_array[pattern_end].op == 'Identity'):
                    pattern_end += 1
            if pattern_end >= len(nodes_as_array):
                break

            match = False
            if (hasattr(p, 'match')): # regexp
                while pattern_end < len(nodes_as_array) and p.match(nodes_as_array[pattern_end].name):
                    match = True
                    pattern_end += 1
            else: # exact string
                match = nodes_as_array[pattern_end].op == p
                pattern_end += 1

            if not match:
                break

        def get_tensors(pattern_nodes):
            # tensors <= all Const nodes within this pattern
            tensor_nodes = [n for n in pattern_nodes if n.op == 'Const']
            tensors = [Struct(name = n.name, obj = n.attr["value"].tensor, shape = get_tensor_dims(n.attr["value"].tensor), data = get_tensor_data(n.attr["value"].tensor))
                for n in tensor_nodes]

            # TODO: unify / reuse code from process_layer
            identity_nodes = [n for n in pattern_nodes if n.op == 'Identity']
            for i in identity_nodes:
                inputs = replace_strings_in_list(i.input, map_ignored_layer_to_its_input)
                map_ignored_layer_to_its_input[i.name] = inputs

            # gather inputs from Op nodes (not Const, not Identity)
            op_nodes = [n for n in pattern_nodes if n not in tensor_nodes and n not in identity_nodes]
            inputs_to_op_nodes = list(flatten([list(flatten(n.input)) for n in op_nodes]))
            inputs_to_op_nodes = replace_strings_in_list(inputs_to_op_nodes, map_ignored_layer_to_its_input)
            inputs_to_op_nodes = [i.split(':')[0] for i in inputs_to_op_nodes]

            # filter only inputs that are coming from nodes that are outside this pattern
            # preserve the order
            pattern_nodes = [n.name for n in pattern_nodes]
            #inputs_from_outside_pattern = remove_duplicates_from_list([i for i in inputs_to_op_nodes if nodes_by_name[i] not in pattern_nodes])
            inputs_from_outside_pattern = remove_duplicates_from_list([i for i in inputs_to_op_nodes if i not in pattern_nodes])

            return inputs_from_outside_pattern, tensors

        if match:
            nodes = nodes_as_array[node_index:pattern_end]
            name = nodes[-1].name
            if args.print_patterns or args.verbose:
                print('PATTERN:', name, ' ~~ ', pattern_name, pattern)
            var_tensors, const_tensors = get_tensors(nodes)
            for n in nodes:
                if n.op == 'Const' or n.op == 'Identity':
                    process_layer(n)

            new_layers = transform_patterns[pattern_name](nodes, var_tensors, const_tensors)
            if not isinstance(new_layers, list):
                if not hasattr(new_layers, name): new_layers.name = name
                new_layers = [new_layers]

            for l in new_layers:
                # TODO: prefix new layer names with scope, patch inputs
                #l.name = name + '/' + l.name
                process_layer(l)

            node_index = pattern_end
            break # pattern found & processed

    if not match:
        # TODO: gather tensors in the same way as patterns do
        process_layer(node)
        node_index += 1

# Cleanup unconnected Identities (they might linger after processing complex node patterns like LSTM)
def cleanup_model(model):
    all_layers = {l.name for l in o_model}
    all_inputs = {i for l in o_model for i in l.inputs}

    def is_unconnected_identity(layer):
        if layer.class_name == 'Activation' and layer.activation == 0: # Identity
            assert(len(layer.inputs) == 1)
            if layer.inputs[0] not in all_layers and layer.name not in all_inputs:
                return True;
        return False;

    return [l for l in model if not is_unconnected_identity(l)]
o_model = cleanup_model(o_model)

# Find global tensors
def dims_to_barracuda_shape(dims):
    shape = list(dims)
    while len(shape) < 4:
        shape = [1] + shape
    return shape




all_inputs = {i for l in o_model for i in l.inputs}
embedded_tensors = {t.name for l in o_model for t in l.tensors}
global_tensors = [t for t in model_tensors if t not in all_inputs and t not in embedded_tensors]
o_globals = []
#for x in global_tensors:
#    shape = dims_to_barracuda_shape(get_tensor_dims(model_tensors[x]))    
#    o_globals += [Struct(
#        name = x,
#        shape = shape,
#        data = np.reshape(get_tensor_data(model_tensors[x]), shape).astype(np.float32))]

# Trim
if args.trim_unused_by_output:
    o_model = barracuda.trim(o_model, args.trim_unused_by_output, args.verbose)

# Find model inputs & outputs
const_tensors = [i for i in all_inputs if i in model_tensors]
const_tensors += global_tensors
for x in const_tensors:
    shape = dims_to_barracuda_shape(get_tensor_dims(model_tensors[x]))

    o_l = Struct(
        type        = 255,  # Load
        class_name  = "Const",
        name        = x,
        pads        = [0,0,0,0],
        strides     = [],
        pool_size   = [],
        axis        = -1,
        alpha       = 1,
        beta        = 0,
        activation  = 0,
        inputs      = [],
        tensors     = [Struct(
            name = x,
            shape = shape,
            data = np.reshape(get_tensor_data(model_tensors[x]), shape).astype(np.float32))]
    )
    o_model.insert(0, o_l)

# Inputs, outputs
all_layers = {l.name for l in o_model}
# global inputs => are inputs that are NOT connected to any layer in the network
# global outputs => are outputs that are NOT feeding any layer in the network OR are coming from Identity layers
o_inputs = {i:input_shapes[i] for l in o_model for i in l.inputs if i not in all_layers and i not in o_memories}

def is_output_layer(layer):
    if layer.class_name == 'Const': # Constants never count as global output even when unconnected
        return False;
    if layer.name not in all_inputs: # this layer is not inputing to any other layer
        return True
    if layer.class_name == 'Activation' and layer.activation == 0: # Identity marks global output
        return True
    return False
o_outputs = [l.name for l in o_model if is_output_layer(l)]

# Compress
if args.compress_f16:
    model = barracuda.compress(model)

# Sort model so that layer inputs are always ready upfront
o_model = barracuda.sort(o_model, o_inputs, o_memories, args.verbose)

# Summary
barracuda.summary(o_model, o_inputs, o_outputs, o_memories,
    global_tensors, #o_globals
    print_layer_links = args.print_layer_links or args.verbose,
    print_barracuda_json = args.print_barracuda_json or args.verbose,
    print_tensors = args.print_tensors or args.verbose)

# Write to file
barracuda.write(o_model, o_inputs, o_outputs, o_memories, args.target_file)
print('DONE: wrote', args.target_file, 'file.')

from __future__ import print_function
import os.path
import numpy as np
import h5py
import json
import barracuda
from barracuda import Struct
from collections import Iterable
from pprint import pprint


if __name__ == '__main__':
    # Handle command line argumengts
    args = barracuda.parse_args(
        description = 'Convert Keras model to Barracuda binary',
        source_extension = '.h5',
        help = 'input Keras serialized .h5 file')
    # The following code can be used as an example of API used from another module
    # convert() is the main entry point for converter
    import keras_to_barracuda as keras2bc
    keras2bc.convert(args.source_file, args.target_file, args.trim_unused_by_output, args)


def get_epsilon(layer):
    # default epsilon taken from https://keras.io/layers/normalization/
    return layer['config'].get('epsilon') or 0.001

add_zero_bias_as_2nd_arg = [
    None,
    lambda default_shape: np.zeros(default_shape[-1])
]

known_classes = {
    'Dense': Struct(
                    id = 1,
                    in_args = ['kernel:0', 'bias:0'],
                    defaults = add_zero_bias_as_2nd_arg,
                    out_shapes = lambda shapes: [
                        [shapes[0][0], 1, 1, shapes[0][1]],  # W
                        [1, 1, 1, shapes[1][0]]              # B
                    ]),
    'Conv2D': Struct(
                    id = 20,
                    in_args = ['kernel:0', 'bias:0'],
                    defaults = add_zero_bias_as_2nd_arg,
                    out_shapes = lambda shapes: [
                        shapes[0],                           # K 
                        [1, 1, 1, shapes[1][0]]              # B
                    ]),
    'SeparableConv2D': Struct(
                    id = 21,
                    in_args = ['kernel:0', 'bias:0'],
                    defaults = add_zero_bias_as_2nd_arg,
                    out_shapes = lambda shapes: [
                        shapes[0],                           # K 
                        [1, 1, 1, shapes[1][0]]              # B
                    ]),
    'Conv2DTranspose': Struct(
                    id = 22,
                    in_args = ['kernel:0', 'bias:0'],
                    defaults = add_zero_bias_as_2nd_arg,
                    out_shapes = lambda shapes: [
                        shapes[0],                           # K 
                        [1, 1, 1, shapes[1][0]]              # B
                    ]),
    'UpSampling2D':     23,
    'MaxPooling2D':     25,
    'AveragePooling2D': 26,
    'GlobalMaxPooling2D': 27,
    'GlobalAveragePooling2D': 28,
    'ZeroPadding2D': 29, # TODO: test

    'Activation':       50,
    'BatchNormalization': Struct(
                    id = 51, # after fusion implemented as ScaleBias
                    in_args = ['gamma:0', 'beta:0', 'moving_mean:0', 'moving_variance:0'],
                    defaults = [
                        lambda default_shape: np.ones(default_shape),
                        lambda default_shape: np.zeros(default_shape),
                        lambda default_shape: np.zeros(default_shape),
                        lambda default_shape: np.ones(default_shape),
                    ],
                    patch_data = lambda data, layer:
                        # fuse [gamma, beta, mean, var, epsilon] => [scale, bias]
                        barracuda.fuse_batchnorm_weights(data[0], data[1], data[2], data[3], get_epsilon(layer))
                    ,
                    out_shapes = lambda shapes: [
                        [1, 1, 1, shapes[0][0]],             # S
                        [1, 1, 1, shapes[1][0]],             # B
                    ]),

    'BatchNormalizationRuntime' : 52,
    'DropoutRuntime' :  60,
    
    'Add':              100,
    'Subtract':         101,
    'Multiply':         102,
    'Div':              103, # TODO: test
    'Pow':              104, # TODO: test
    'Minimum':          110, # TODO: test
    'Maximum':          111, # TODO: test

    'Flatten':          200,
    'Reshape':          201,
    'Concatenate':      210,
}

requires_runtime_flag = {
    'Dropout' : 'DropoutRuntime',
    'BatchNormalization' : 'BatchNormalizationRuntime',
}

known_activations = {
    'linear' : 0,
    'relu' : 1,
    'softmax' : 2,
    'tanh' : 3,
    'sigmoid' : 4,
    'elu' : 5,
    'relu6' : 6,
    'leakyrelu' : 7,
    'selu' : 8,

    'softplus' : 11,
    'softsign' : 12,

    'hard_sigmoid' : 21,

    'exponential' : 113,
}

known_paddings = {
	'valid' : [0,0,0,0],
	'same'  : [-1] # SameUpper, just like TensorFlow
    # TODO: 'causal'
}

supported_data_formats = {
    'channels_last'
}

# Keras layers that define model structure, but are not directly serialized in Barracuda file
def nested_model(layers, name, inputs, parent_model_tensors, args, context):
    # find Model inputs and map them to defaults - nested layers will can access them as ('')
    inputs = list(flatten([context.map_model_to_its_outputs[i] for i in inputs]))
    model_default_inputs_and_missing_layers = {'' : inputs}

    process_model(layers, parent_model_tensors[name], args, model_default_inputs_and_missing_layers, context)

    outputs = extract_strings(layers.get('output_layers'))
    outputs = replace_strings_in_list(outputs, model_default_inputs_and_missing_layers)
    context.map_model_to_its_outputs[name] = outputs

def input_layer(layer, name, input_shapes):
    shape = get_input_layer_shape(layer)
    input_shapes[name] = shape

transient_classes = {
    'Model':
        lambda layers, name, inputs, model_tensors, args, context:
            nested_model(layers, name, inputs, model_tensors, args, context)
    ,
    'InputLayer':
        lambda layers, name, inputs, model_tensors, args, context:
            input_layer(layers, name, context.input_shapes)
    ,
}

# Helper
def embody(v, default=0):
    return default if v is None else v

# Parse
def flatten(items,enter=lambda x:isinstance(x, list)):
    # http://stackoverflow.com/a/40857703
    # https://github.com/ctmakro/canton/blob/master/canton/misc.py
    """Yield items from any nested iterable; see REF."""
    for x in items:
        if enter(x):
            yield from flatten(x)
        else:
            yield x

def extract_strings(array_of_values):
    if not array_of_values:
        return array_of_values
    return [x for x in flatten(array_of_values) if type(x) == str]

def replace_strings_in_list(array_of_strigs, replace_with_strings):
    "A value in replace_with_strings can be either single string or list of strings"
    potentially_nested_list = [replace_with_strings.get(s) or s for s in array_of_strigs]
    return list(flatten(potentially_nested_list))

def get_input_layer_shape(layer):
    shape = layer.get('batch_input_shape')
    if not shape:
        return [-1, -1, -1, -1]
    if len(shape) == 1:
        return [1, 1, 1, shape[0]]
    if len(shape) == 2:
        return [shape[0], 1, 1, shape[1]]
    return shape    

#########################################################

class ModelBuilderContext:
    def __init__(self):
        self.layers = []
        self.input_shapes = {}
        self.model_memories = []
        self.map_model_to_its_outputs = {'':''}

def process_model(layers, model_tensors, args, map_ignored_layer_to_its_input = {}, o_context = ModelBuilderContext()): #model_tensors, input_shapes, map_ignored_layer_to_its_input = {}):
    prev_layer_name = ''

    if 'get' in dir(layers):
        layers = layers.get('layers')

    # special handling for Sequential model case in Keras
    # when the 1st layer can define the shape of the input
    if layers:
        o_context.input_shapes[prev_layer_name] = get_input_layer_shape(layers[0]['config'])

    for layer in layers:
        
        name = layer['config']['name']
        class_name = layer['class_name']
        inputs = extract_strings(layer.get('inbound_nodes')) or [prev_layer_name]
        inputs = replace_strings_in_list(inputs, map_ignored_layer_to_its_input)

        if args.print_layers or args.verbose:
            print("'%s' %s %s" % (name, class_name, inputs))

        if class_name in transient_classes:
            transient_classes[class_name](layer['config'], name, inputs, model_tensors, args, o_context)
            continue
       
        if not class_name in known_classes:
            if class_name in requires_runtime_flag:
                print('SKIP:', class_name, 'layer is used only for training')
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
        o_l.inputs = inputs

        activation  = layer['config'].get('activation')
        axis        = layer['config'].get('axis')
        padding     = layer['config'].get('padding')
        strides     = layer['config'].get('strides')
        pool_size   = layer['config'].get('pool_size')
        size        = layer['config'].get('size')
        use_bias    = layer['config'].get('use_bias')
        data_frmt   = layer['config'].get('data_format')
        alpha       = layer['config'].get('alpha')
        beta        = layer['config'].get('beta')

        if activation and not activation in known_activations:
            print('IGNORED: unknown activation', activation)
        if padding and not padding in known_paddings:
            print('IGNORED: unknown padding', padding)
        if data_frmt and not data_frmt in supported_data_formats:
            print('UNSUPPORTED: data format', data_frmt)

        o_l.activation  = known_activations.get(activation) or 0
        o_l.pads        = known_paddings.get(padding) if padding else [0,0,0,0] or [0,0,0,0]
        o_l.strides     = strides or []
        o_l.pool_size   = pool_size or size or []
        o_l.use_bias    = embody(use_bias, default=True)
        o_l.axis        = embody(axis, default=-1)
        o_l.alpha       = embody(alpha, default=1)
        o_l.beta        = beta or 0

        tensors = {}
        # Process input arguments
        if hasattr(klass, 'in_args'):
            if isinstance(klass.in_args, list):
                klass.in_args = {name:idx for idx, name in enumerate(klass.in_args)}

            def convert_datasets_to_tensor(name, obj):
                if type(obj) == h5py.Dataset:
                    name = os.path.basename(name) # leave only last chunk of the tensor name, such as 'kernel:0'
                    try:
                        index = klass.in_args[name]
                        tensors[index] = Struct(name = obj.name, shape = obj.shape, data = obj[:])
                        if index == 0 or -1 not in tensors:
                            tensors[-1] = tensors[index] # use '-1' as 'default' tensor
                    except KeyError:
                        print('SKIP: unknown tensor', name)
            try:
                layer_tensors = model_tensors[o_l.name]
                layer_tensors.visititems(convert_datasets_to_tensor)
            except KeyError:
                # no tensors with specified name, op does not require tensor args
                pass

        # Set defaults for missing argument tensors
        if hasattr(klass, 'defaults'):
            assert(hasattr(klass, 'in_args'))
            index_to_arg_name = {v: k for k, v in klass.in_args.items()}

            default_shape = tensors[-1].shape
            for index, default in enumerate(klass.defaults):
                if index not in tensors and klass.defaults[index] != None:
                    data = klass.defaults[index](default_shape)
                    if args.verbose:
                        print(name + ':' +index_to_arg_name[index], 'default to', data[0])
                    tensors[index] = Struct(name = ('/model_weights/default/%s/%s') % (name, index_to_arg_name[index]),
                        shape = np.shape(data), data = data)

        # Patch tensor data
        if hasattr(klass, 'patch_data'):
            data = {i:x.data for i, x in tensors.items()}

            patch_data_fn = klass.patch_data
            patch_data_expected_arg_count = patch_data_fn.__code__.co_argcount
            patch_data_args = (data, layer) if patch_data_expected_arg_count > 1 else (data,)
            tensor_data = patch_data_fn(*patch_data_args)
            for i, data in enumerate(tensor_data):
                tensors[i].data = data

        # Force all tensors to float32
        for x in tensors.values():
            x.data = x.data.astype(np.float32)

        # Patch shapes and write out tensors
        o_l.tensors = []
        if hasattr(klass, 'out_shapes'):
            shapes = klass.out_shapes({i:x.shape for i, x in tensors.items()})
            for i, shape in enumerate(shapes):
                tensors[i].shape = shape
                o_l.tensors.append(tensors[i])
        else:
            # no 'out_shapes' lambda was specified, op does not require tensor args
            pass
        
        # Layer is ready
        o_context.layers.append(o_l)
        prev_layer_name = o_l.name

    return o_context.layers, o_context.input_shapes, o_context.model_memories

def convert(source_file, target_file, trim_unused_by_output="", verbose=False, compress_f16=False):
    """
    Converts a Keras model into a Barracuda model.
    :param source_file: The Keras Model
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

    # Load Keras model
    print("Converting %s to %s" % (source_file, target_file))
    i_model = h5py.File(source_file, 'r')

    configJSON = json.loads(i_model.attrs['model_config'].decode('utf-8'))
    layers = configJSON['config']
    model_tensors = i_model['model_weights']

    if args.print_source_json or args.verbose:
        pprint(configJSON)

    # Convert
    o_model = barracuda.Model()
    o_model.layers, o_input_shapes, o_model.memories = \
        process_model(layers, model_tensors, args)
    
    # Gather patched model tensors
    for l in o_model.layers:
        for x in l.tensors:
            o_model.tensors[x.name] = x

    # Trim
    if trim_unused_by_output:
        o_model.layers = barracuda.trim(o_model.layers, trim_unused_by_output, args.verbose)

    # Find model inputs & outputs
    all_layers = {l.name for l in o_model.layers}
    all_inputs = {i for l in o_model.layers for i in l.inputs}
    # global inputs - are inputs that are NOT connected to any layer in the network
    # global outputs - are outputs that are NOT feeding any layer in the network
    o_model.inputs = {i:o_input_shapes[i] for l in o_model.layers for i in l.inputs if i not in all_layers}
    o_model.outputs = [l.name for l in o_model.layers if l.name not in all_inputs]

    # Compress
    if compress_f16:
        o_model = barracuda.compress(o_model)

    # Summary
    barracuda.summary(o_model,
        print_layer_links = args.print_layer_links or args.verbose,
        print_barracuda_json = args.print_barracuda_json or args.verbose,
        print_tensors = args.print_tensors or args.verbose)

    # Write to file
    barracuda.write(o_model, target_file)
    print('DONE: wrote', target_file, 'file.')

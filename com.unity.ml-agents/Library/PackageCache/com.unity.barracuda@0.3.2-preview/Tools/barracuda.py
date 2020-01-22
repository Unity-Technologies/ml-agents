from __future__ import print_function
from collections import defaultdict
import numpy as np
import json
import struct # convert from Python values and C structs
import re
import argparse
import os.path

BARRACUDA_VERSION = 16

# Definition of Barracuda model
class Model:
    def __init__(self):
        self.layers = []
        self.tensors = {}
        self.inputs = {}
        self.outputs = []
        self.globals = []
        self.memories = []

class Struct:
    "A structure that can have any fields defined."
    def __init__(self, **entries): self.__dict__.update(entries)

# Parse command line argumengts
def parse_args(description, source_extension, help):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('source_file', help=help)
    parser.add_argument('target_file', help='output Barracuda binary file')
    parser.add_argument('-trim', '--trim-unused-by-output')
    parser.add_argument('--print-layers', action='store_true')
    parser.add_argument('--print-source-json', action='store_true')
    parser.add_argument('-json', '--print-barracuda-json', action='store_true')
    parser.add_argument('--print-layer-links', action='store_true')
    parser.add_argument('--print-patterns', action='store_true')
    parser.add_argument('--print-tensors', action='store_true')
    parser.add_argument('--print-supported-ops', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    args.compress_f16 = False # TEMP: disabled, until properly implemented parser.add_argument('-f16', '--compress-f16', action='store_true')
    
    output_extension = '.bc' if not args.compress_f16 else '.f16.bc'

    if not os.path.exists(args.source_file):
        args.source_file = args.source_file + source_extension

    if not os.path.exists(args.source_file):
        print('File', args.source_file, 'does not exist.')
        exit(-1)

    def replaceFilenameExtension(filename, newExtenstion):
        return os.path.splitext(os.path.basename(filename))[0] + newExtenstion;

    if os.path.isdir(args.target_file):
        args.target_file = os.path.join(args.target_file, replaceFilenameExtension(args.source_file, output_extension))

    if args.verbose:
        print(args)

    return args

# Fuse training time BatchNorm tensors into Scale & Bias
def fuse_batchnorm_weights(gamma, beta, mean, var, epsilon):
    # https://github.com/Tencent/ncnn/blob/master/src/layer/batchnorm.cpp
    """ float sqrt_var = sqrt(var_data[i]);
        a_data[i] = bias_data[i] - slope_data[i] * mean_data[i] / sqrt_var;
        b_data[i] = slope_data[i] / sqrt_var;
        ...
        ptr[i] = b * ptr[i] + a;
    """
    scale = gamma / np.sqrt(var + epsilon)
    bias = beta - gamma * mean / np.sqrt(var + epsilon)
    return [scale, bias]

# Resort layers so that all inputs are satisfied for every layer beforehand
def sort(model, inputs, memories, verbose):
    if hasattr(model, 'layers'):
        model = model.layers
    inputs_and_memories = set(list(inputs) + list(memories[1::3]))

    def find_missing_inputs(model, inputs):
        missing = set()
        ready = set(inputs)
        for l in model:
            for i in l.inputs:
                if i not in ready:
                    missing.add(i)
            ready.add(l.name)
        return missing

    # Class to represent a graph 
    # Taken from: https://www.geeksforgeeks.org/python-program-for-topological-sorting/
    class Graph: 
        def __init__(self,vertices): 
            self.graph = defaultdict(list) #dictionary containing adjacency List 
            self.V = vertices #No. of vertices 
      
        # function to add an edge to graph 
        def addEdge(self,u,v): 
            self.graph[u].append(v) 
      
        # A recursive function used by topologicalSort 
        def topologicalSortUtil(self,v,visited,stack): 
      
            # Mark the current node as visited. 
            visited[v] = True
      
            # Recur for all the vertices adjacent to this vertex 
            for i in self.graph[v]: 
                if visited[i] == False: 
                    self.topologicalSortUtil(i,visited,stack) 
      
            # Push current vertex to stack which stores result 
            stack.insert(0,v) 

        # The function to do Topological Sort. It uses recursive  
        # topologicalSortUtil() 
        def topologicalSort(self): 
            # Mark all the vertices as not visited 
            visited = [False]*self.V 
            stack =[] 
      
            # Call the recursive helper function to store Topological 
            # Sort starting from all vertices one by one 
            for i in range(self.V): 
                if visited[i] == False: 
                    self.topologicalSortUtil(i,visited,stack) 
      
            #print(stack)
            return stack

    if (len(find_missing_inputs(model, inputs_and_memories)) == 0):
        return model

    g = Graph(len(model))

    layers = {}
    id = 0
    for l in model:
        layers[l.name] = id;
        id += 1

    for layer in model:
        for i in layer.inputs:
            if i not in inputs_and_memories:
                g.addEdge(layers[i], layers[layer.name])

    sorted_layer_indices = g.topologicalSort()
    print("SORTED:", sorted_layer_indices)
    new_model = [model[idx] for idx in sorted_layer_indices]

    assert(len(find_missing_inputs(new_model, inputs_and_memories)) == 0)
    return new_model

# Trim
def trim(model, criteria_regexp_string, verbose):
    if hasattr(model, 'layers'):
        model = model.layers

    def flatten(items,enter=lambda x:isinstance(x, list)):
        # http://stackoverflow.com/a/40857703
        # https://github.com/ctmakro/canton/blob/master/canton/misc.py
        """Yield items from any nested iterable; see REF."""
        for x in items:
            if enter(x):
                yield from flatten(x)
            else:
                yield x

    def trim_model(model, outputs):
        layers = {l.name:l for l in model}
        connected = {o for o in outputs}
        while len(outputs) > 0:
            outputs = set(flatten([layers[o].inputs for o in outputs if o in layers]))
            if verbose and len(outputs) > 0:
                print(outputs)
            for o in outputs:
                connected.add(o)

        trimmed = [l.name for l in model if l.name not in connected]
        def array_without_brackets(arr):
            return str(arr)[1:-1] # array to string without brackets
        print("TRIMMED:", array_without_brackets(trimmed))

        return [l for l in model if l.name in connected]

    layer_names = {l.name for l in model}
    criteria = re.compile(criteria_regexp_string)
    preserve_outputs = list(filter(criteria.match, layer_names))
    if preserve_outputs:
        print("Trimming model given outputs to preserve:", preserve_outputs)
        model = trim_model(model, preserve_outputs)
    else:
        print("WARNING: Trim couldn't find any layers to match:", criteria_regexp_string)
    return model

# Setup load ops for constants
def setup_constants(model, get_tensor_shape_lambda, get_tensor_data_lambda):
    all_inputs = {i for l in model.layers for i in l.inputs}
    const_tensor_names = [i for i in all_inputs if i in model.tensors]
    const_tensor_names += model.globals
    for name in const_tensor_names:
        tensor = model.tensors[name]
        shape = get_tensor_shape_lambda(tensor)
        o_l = Struct(
            type        = 255,  # Load
            class_name  = "Const",
            name        = name,
            pads        = [0,0,0,0],
            strides     = [],
            pool_size   = [],
            axis        = -1,
            alpha       = 1,
            beta        = 0,
            activation  = 0,
            inputs      = [],
            tensors     = [Struct(
                name = name,
                shape = shape,
                data = np.reshape(get_tensor_data_lambda(tensor), shape).astype(np.float32))]
        )
        model.layers.insert(0, o_l)
    return model

# Fuse
def fuse(model, verbose):
    i = 0
    while i < len(model) - 1:
        if model[i].type == model[i+1].type and model[i].type == 255: # Load
            model[i].tensors += model[i+1].tensors
            del model[i+1]
        else:
            i += 1
    return model

def compress(model):
    compress_classes = {
        'Dense'
    }
    for l in model.layers:
        if (l.class_name in compress_classes):
            print("Compressing %s layer '%s' weights to float16" % (l.class_name, l.name))
            for x in l.tensors:
                x.data = np.float16(x.data)
    return model


def simplify_names(model):
    def strip_tensforlow_postfix(name):
        #return re.sub(r":\d+", '',  name) # completely remove Tensorflow complex tensor indexing
        return name.replace(':0', '')

    for l in model.layers:
        l.name = strip_tensforlow_postfix(l.name)
        l.inputs = [strip_tensforlow_postfix(i) for i in l.inputs]
        for x in l.tensors:
            x.name = strip_tensforlow_postfix(x.name)

    model.tensors = [strip_tensforlow_postfix(x) for x in model.tensors]

    if isinstance(model.inputs, dict):
        model.inputs = {strip_tensforlow_postfix(name):shape for name, shape in model.inputs.items()}
    else:
        model.inputs = [strip_tensforlow_postfix(i) for i in model.inputs]

    model.outputs = [strip_tensforlow_postfix(o) for o in model.outputs]
    model.globals = [strip_tensforlow_postfix(g) for g in model.globals]
    model.memories = [strip_tensforlow_postfix(m) if isinstance(str) else m for m in model.memories]

    return model

# Verbose
def to_json(model):
    class StructEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, np.ndarray): # skip binary data packed inside ndarray
                return ""
            if getattr(o, '__dict__', None):
                return o.__dict__
            return str(o)

    s = json.dumps(model.layers, cls=StructEncoder, separators=(', ',':'))
    # custom formatting
    s = s.replace(']}, {', ']},\n{')
    s = s.replace(':[{', ':[\n\t{')
    s = s.replace('}, {', '},\n\t{')
    s = s.replace('}], ', '}],\n\t')
    s = s.replace('"', "'")
    # skip defaults
    s = s.replace("'activation':0, ", '')
    s = s.replace("'strides':[], ", '')
    s = s.replace("'pads':[0, 0, 0, 0], ", '')
    s = s.replace("'axis':-1, ", '')
    s = s.replace("'alpha':1, ", '')
    s = s.replace("'beta':0, ", '')
    s = s.replace("'tensors':[], ", '')
    return s

def summary(model, print_layer_links, print_barracuda_json, print_tensors):
    def array_without_brackets(arr):
        return str(arr)[1:-1] # array to string without brackets

    if print_layer_links:
        for l in model.layers:
            print(l.name, " <= ", l.inputs)

    if print_barracuda_json:
        print(to_json(model))

    if model.globals:
        if isinstance(model.globals, dict):
            model.globals = {x.name:x.shape for x in model.globals}
        print("GLOBALS:", array_without_brackets(model.globals))

    for l in model.layers:
        if isinstance(model.inputs, dict):
            ins = {i:model.inputs[i] for i in l.inputs if i in model.inputs}
        else:
            ins = [i for i in l.inputs if i in model.inputs]
        if ins:
            print("IN: %s => '%s'" % (array_without_brackets(ins), l.name))
    for mem_in, mem_out in zip(model.memories[1::3], model.memories[2::3]):
        print("MEM: '%s' => '%s'" % (mem_in, mem_out))
    print("OUT:", array_without_brackets(model.outputs))

    if (print_tensors):
        for l in model.layers:
            for x in l.tensors:
                print(x.name, x.shape, x.data.dtype, x.data)

class Build:
    def __init__(self, scope=''):
        self.scope = scope
        self.layers = []
        self.names_taken = set()

    def __getattr__(self, attr):
        if attr == '_':
            return self.layers[-1].name if len(self.layer) > 0 else self.scope
        raise AttributeError(attr)

    def _patch_last_layer_name_and_return(self):
        if self.layers[-1].name:
            return self.layers[-1].name

        # generate unique name based on op and increasing id
        name = self.layers[-1].op

        i = 1
        while name in self.names_taken:
            name = self.layers[-1].op + '_' + str(i)
            i += 1
        self.names_taken.add(name)

        self.layers[-1].name = self.scope + ('/' if self.scope else '') + name
        return self.layers[-1].name

    def concat(self, a, b, axis=-1, out=''):
        self.layers += [Struct(name=out, op='Concat', axis=axis, input=[a, b])]
        return self._patch_last_layer_name_and_return()
    def mad(self, x, kernel, bias, out=''):
        self.layers += [Struct(name=out, op='Dense', input=[x, kernel, bias])]
        return self._patch_last_layer_name_and_return()
    def mul(self, a, b, out=''):
        self.layers += [Struct(name=out, op='Mul', input=[a, b])]
        return self._patch_last_layer_name_and_return()
    def add(self, a, b, out=''):
        self.layers += [Struct(name=out, op='Add', input=[a, b])]
        return self._patch_last_layer_name_and_return()
    def sub(self, a, b, out=''):
        self.layers += [Struct(name=out, op='Sub', input=[a, b])]
        return self._patch_last_layer_name_and_return()
    def sigmoid(self, x, out=''):
        self.layers += [Struct(name=out, op='Sigmoid', input=[x])]
        return self._patch_last_layer_name_and_return()
    def tanh(self, x, out=''):
        self.layers += [Struct(name=out, op='Tanh', input=[x])]
        return self._patch_last_layer_name_and_return()
    def reduce(self, op, x, axis=-1, out=''):
        self.layers += [Struct(name=out, op='Reduce'+op, axis=axis, input=[x])]
        return self._patch_last_layer_name_and_return()
    def pool(self, op, x, out=''):
        self.layers += [Struct(name=out, op=op+'Pool', input=[x])]
        return self._patch_last_layer_name_and_return()
    def strided_slice(self, x, begin, end, strides, rank, out=''):
        self.layers += [Struct(name=out, op='StridedSlice', rank=rank, starts=begin, ends=end, slice_strides=strides, input=[x])]
        return self._patch_last_layer_name_and_return()

def mean(name, input, axis=-1):
    ''' combines mean operation out of several simpler ops
    '''
    nn = Build(name)
    if np.array_equal(axis, [1,2]):
        nn.pool('GlobalAvg', input, out=name)
    elif np.array_equal(axis, [1,2,3]):
        nn.reduce('Mean',                # over channels
            nn.pool('GlobalAvg', input), # over height & width
        out=name) 
    elif np.array_equal(axis, [3]) or np.array_equal(axis, [-1]) or np.array_equal(axis, 3) or np.array_equal(axis, -1):
        nn.reduce('Mean', input, out=name)
    return nn.layers

def rnn(name, input, state, kernel, bias, new_state, number_of_gates = 2):
    ''' - Ht = f(Xt*Wi + Ht_1*Ri + Wbi + Rbi)
    '''

    nn = Build(name)
    nn.tanh(
        nn.mad(kernel=kernel, bias=bias,
            x=nn.concat(input, state)),
        out=new_state);
    return nn.layers

def gru(name, input, state, kernel_r, kernel_u, kernel_c, bias_r, bias_u, bias_c, new_state, number_of_gates = 2):
    ''' - zt = f(Xt*Wz + Ht_1*Rz        + Wbz + Rbz)
        - rt = f(Xt*Wr + Ht_1*Rr        + Wbr + Rbr)
        - ht = g(Xt*Wh + (rt . Ht_1)*Rh + Rbh + Wbh)
        - Ht = (1-zt).ht + zt.Ht_1
    '''
    nn = Build(name)
    inputs = nn.concat(input, state)

    u = nn.sigmoid(nn.mad(inputs, kernel_u, bias_u))
    r = nn.sigmoid(nn.mad(inputs, kernel_r, bias_r))
    r_state = nn.mul(r, state)

    c = nn.tanh(nn.mad(kernel=kernel_c, bias=bias_c,
        x=nn.concat(input, r_state)))

    # new_h = u' * state + (1 - u') * c'
    #       = u' * state + c' - u' * c'

    # u' * state + c'
    nn.add(nn.mul(u, state), c)
    # - u' * c'
    nn.sub(nn._, nn.mul(u, c),
        out=new_state)

    return nn.layers;

def lstm(name, input, state_c, state_h, kernel_i, kernel_j, kernel_f, kernel_o, bias_i, bias_j, bias_f, bias_o, new_state_c, new_state_h):
    ''' Full:
    - it = f(Xt*Wi + Ht_1*Ri + Pi . Ct_1 + Wbi + Rbi)
    - ft = f(Xt*Wf + Ht_1*Rf + Pf . Ct_1 + Wbf + Rbf)
    - ct = g(Xt*Wc + Ht_1*Rc + Wbc + Rbc)
    - Ct =  ft . Ct_1  + it . ct
    - ot = f(Xt*Wo + Ht_1*Ro + Po . Ct + Wbo + Rbo)
    - Ht =  ot . h(Ct)
    '''

    ''' No peephole:
    - it = f(Xt*Wi + Ht_1*Ri + Wbi + Rbi)
    - ft = f(Xt*Wf + Ht_1*Rf + Wbf + Rbf)
    - ct = g(Xt*Wc + Ht_1*Rc + Wbc + Rbc)
    - Ct =   ft . Ct_  + it . ct
    - ot = f(Xt*Wo + Ht_1*Ro + Wbo + Rbo)
    - Ht =   ot . h(Ct)
    '''

    nn = Build(name)
    inputs = nn.concat(input, state_h)

    i = nn.sigmoid(nn.mad(x=inputs, kernel=kernel_i, bias=bias_i))
    j =    nn.tanh(nn.mad(inputs, kernel_j, bias_j))
    f = nn.sigmoid(nn.mad(inputs, kernel_f, bias_f))
    o = nn.sigmoid(nn.mad(inputs, kernel_o, bias_o))

    # new_c = state_c * f' + i' * j'
    nn.add(
        nn.mul(state_c, f), nn.mul(i, j),
        out=new_state_c)

    # new_h = 
    nn.mul(o, nn.tanh(new_state_c),
        out=new_state_h)

    return nn.layers

# Serialize
class BarracudaWriter:
    f = None

    def __init__(self, filename):
        self.f = open(filename, 'wb+')

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.f.close()

    def write_array(self, arr):
        arr.tofile(self.f)

    def write_str_array(self, array_of_strigs):
        self.write_int32(len(array_of_strigs))
        for s in array_of_strigs:
            self.write_str(s)

    def write_str(self, s):
        self.write_int32(len(s))
        self.f.write(s.encode('ascii'))

    def write_float(self, d):
        self.f.write(struct.pack('<f', d))

    def write_int32(self, d):
        self.f.write(struct.pack('<i', d))

    def write_int64(self, d):
        self.f.write(struct.pack('<q', d))

    def write_shape(self, s):
        self.write_int32(len(s))
        for el in s:
            self.write_int32(el if el != None else -1)

    def close(self):
        self.f.close()

def write(model, filename):

    with BarracudaWriter(filename) as w:

        #VERSION = 0xBA22AC0DA000 + BARRACUDA_VERSION
        w.write_int64(BARRACUDA_VERSION)

        # inputs
        w.write_int32(len(model.inputs))
        for name, shape in model.inputs.items():
            w.write_str(name)
            w.write_shape(shape)
        # outputs
        w.write_str_array(model.outputs)

        # memories
        w.write_int32(len(model.memories)//3)
        for mem_shape, mem_in, mem_out in zip(model.memories[0::3], model.memories[1::3], model.memories[2::3]):
            w.write_shape(mem_shape)
            w.write_str(mem_in)
            w.write_str(mem_out)

        # layers
        offset = 0
        all_tensors = []

        w.write_int32(len(model.layers))
        for l in model.layers:

            assert(not l.name in l.inputs)

            w.write_str(l.name)
            w.write_int32(l.type)
            w.write_int32(l.activation)
            w.write_int32(0) #dummy
            w.write_int32(0) #dummy
            w.write_shape(l.pads)
            w.write_shape(l.strides)
            w.write_shape(l.pool_size)
            w.write_int32(l.axis)
            w.write_float(l.alpha)
            w.write_float(l.beta)
            w.write_int32(0) #dummy
            w.write_str_array(l.inputs)

            w.write_int32(len(l.tensors))
            for x in l.tensors:
                assert(len(x.shape) == 4)
                assert(x.data.nbytes % 4 == 0)
                length = x.data.nbytes >> 2 # length is measured in float32s (at least for now)

                w.write_str(x.name)
                w.write_shape(x.shape)
                w.write_int64(offset)
                w.write_int32(x.data.itemsize)
                w.write_int32(length)

                offset += length
                all_tensors.append(x)

        for x in all_tensors:
            w.write_array(x.data)


def print_known_operations(known_classes, known_activations):
    print('OPS supported by the converter:')
    for key in sorted(known_classes.keys()):
        print(key)
    print('ACTIVATIONS supported by the converter:')
    for key in sorted(known_activations.keys()):
        print(key)
from __future__ import print_function
from collections import defaultdict
import numpy as np
import json
import struct # convert from Python values and C structs
import re
import argparse
import os.path

BARRACUDA_VERSION = 15

# Parse command line argumengts
def parse_args(description, source_extension, help):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('source_file', help=help)
    parser.add_argument('target_file', help='output Barracuda binary file')
    parser.add_argument('-f16', '--compress-f16', action='store_true')
    parser.add_argument('-gan', '--include-gan-layers', action='store_true')
    parser.add_argument('-trim', '--trim-unused-by-output')
    parser.add_argument('--print-layers', action='store_true')
    parser.add_argument('--print-source-json', action='store_true')
    parser.add_argument('-json', '--print-barracuda-json', action='store_true')
    parser.add_argument('--print-layer-links', action='store_true')
    parser.add_argument('--print-patterns', action='store_true')
    parser.add_argument('--print-tensors', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

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

# Resort layers so that all inputs are satisfied for every layer beforehand
def sort(model, inputs, memories, verbose):
    inputs_and_memories = set(list(inputs) + list(memories[1::3]))

    def find_missing_inputs():
        nonlocal model
        nonlocal inputs_and_memories
        missing = set()
        ready = set(inputs_and_memories) # copy
        print(ready)
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

    if (len(find_missing_inputs()) == 0):
        return model

    g = Graph(len(model))
    #layers = {l.name:l for l in model}

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

    assert(len(find_missing_inputs()) == 0)
    return new_model



# Trim
def trim(model, criteria_regexp_string, verbose):
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

def compress(model):
    compress_classes = {
        'Dense'
    }
    for l in model:
        if (l.class_name in compress_classes):
            print("Compressing %s layer '%s' weights to float16" % (l.class_name, l.name))
            for x in l.tensors:
                x.data = np.float16(x.data)
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

    s = json.dumps(model, cls=StructEncoder, separators=(', ',':'))
    # custom formatting
    s = s.replace(']}, {', ']},\n{')
    s = s.replace(':[{', ':[\n\t{')
    s = s.replace('}, {', '},\n\t{')
    s = s.replace('"', "'")
    return s

def summary(model, inputs, outputs, memories, globals, print_layer_links, print_barracuda_json, print_tensors):
    def array_without_brackets(arr):
        return str(arr)[1:-1] # array to string without brackets

    if print_layer_links:
        for l in model:
            print(l.name, " <= ", l.inputs)

    if print_barracuda_json:
        print(to_json(model))

    if globals:
        if isinstance(globals, dict):
            globals = {x.name:x.shape for x in globals}
        print("GLOBALS:", array_without_brackets(globals))

    for l in model:
        if isinstance(inputs, dict):
            ins = {i:inputs[i] for i in l.inputs if i in inputs}
        else:
            ins = [i for i in l.inputs if i in inputs]
        if ins:
            print("IN: %s => '%s'" % (array_without_brackets(ins), l.name))
    for mem_in, mem_out in zip(memories[1::3], memories[2::3]):
        print("MEM: '%s' => '%s'" % (mem_in, mem_out))
    print("OUT:", array_without_brackets(outputs))

    if (print_tensors):
        for l in model:
            for x in l.tensors:
                print(x.name, x.shape, x.data.dtype, x.data)

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

def write(model, inputs, outputs, memories, filename):

    with BarracudaWriter(filename) as w:

        #VERSION = 0xBA22AC0DA000 + BARRACUDA_VERSION
        w.write_int64(BARRACUDA_VERSION)

        # inputs
        w.write_int32(len(inputs))
        for name, shape in inputs.items():
            w.write_str(name)
            w.write_shape(shape)
        # outputs
        w.write_str_array(outputs)

        # memories
        w.write_int32(len(memories)//3)
        for mem_shape, mem_in, mem_out in zip(memories[0::3], memories[1::3], memories[2::3]):
            w.write_shape(mem_shape)
            w.write_str(mem_in)
            w.write_str(mem_out)

        # layers
        offset = 0
        all_tensors = []

        w.write_int32(len(model))
        for l in model:

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




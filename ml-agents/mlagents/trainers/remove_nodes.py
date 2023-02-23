import onnx

def remove_node(model, node_name):
    node_to_remove = None
    for node in model.graph.node:
        #print(node.name)
        if node.name == node_name:
            print(node.name)
            print(node.input)
            print(f"out: {node.output}")
            node_to_remove = node
            break
    if node_to_remove != None:
        if len(node_to_remove.output) > 1:
            print(f"ERROR: This function currently only handles removal of nodes with a single output")
            return
        print(f"Removing {node_to_remove.name}")
        print(f"Looking for {node_to_remove.output[0]}")
        model.graph.node.remove(node_to_remove)
        for connectNode in model.graph.node:
            #print(connectNode.name)
            if connectNode.input == []: 
                continue
            
            for i in range(0, len(connectNode.input)):
                output = connectNode.input[i]
                #print(f"if {output} == {node.output[0]}")
                if output == node.output[0]:
                    print(f"Replace {connectNode.name} {i}th input '{output}' with '{node.input[0]}'")
                    connectNode.input[i] = node.input[0]
                    

def remove_nodes(model, node_name_list):
    for node_name in node_name_list:
        remove_node(model, node_name)
        
        
def main():
    model_path = ".\Project\Assets\ML-Agents\Examples\Sorter\TFModels\Sorter 1.onnx"
    model = onnx.load(model_path)
    remove_nodes(model, ["Reshape_5", "Reshape_17", "Transpose_15", "Transpose_3"]) 
    
    
# For python debugger to directly run this script
if __name__ == "__main__":
    main()   
    
    
import onnx
import onnx.numpy_helper
import sys
import os

def main():    
    metadata_fields = ["memory_size", "version_number", "continuous_action_output_shape", "discrete_action_output_shape", "is_continuous_control", "action", "action_output_shape"]
    for model_file in os.listdir():
        if model_file.endswith(".onnx"):

            print(f"--------Processing {model_file}")
            # Load the ONNX model
            model = onnx.load(model_file)
            
            # Print the model information
            print("Model name: ", model.graph.name)
            print("Inputs: ", [input.name for input in model.graph.input])
            print("Outputs: ", [output.name for output in model.graph.output])
            
            initializer_names = [x.name for x in model.graph.initializer]
            output_names = [x.name for x in model.graph.output]
            # Extract keys and values from output constants we want to embed as metadata, add as new metadata if they don't exist, otherwise update them.
            for item in model.graph.initializer:
                for name in metadata_fields:
                    if item.name == name:
                        array = onnx.numpy_helper.to_array(item)
                        try:
                            index = [x.key for x in model.metadata_props].index(name)
                            metadata = model.metadata_props[index]
                            print(f"Updating from {metadata.value}")
                        except ValueError:
                            metadata = model.metadata_props.add()
                            print("Adding")

                        metadata.key = name
                        metadata.value = f"{array.shape} {array.flatten()}"
                            
                        print(f"{metadata.key} = {metadata.value}")

            for name in [x for x in output_names if x not in initializer_names and x in metadata_fields]:
                try:
                    index = [x.key for x in model.metadata_props].index(name)
                    metadata = model.metadata_props[index]
                    print(f"{name} existed already and has no initializer")
                except ValueError:
                    print(f"{name} not in initializers, adding with default value")
                    print(f"{name} (1,) [0.]")
                    metadata = model.metadata_props.add()
                    metadata.key = name
                    metadata.value = f"(1,) [0.]"
                print(f"{metadata.key} = {metadata.value}")
    
            onnx.save(model, model_file)
            
            
            

# For python debugger to directly run this script
if __name__ == "__main__":
    main()
    
    

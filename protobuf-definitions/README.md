# Unity ML-Agents Protobuf Definitions

Contains relevant definitions needed to generate probobuf files used in [ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents).

## Requirements

* grpc 1.14.1
* protobuf 3.6.0

## Set-up & Installation

`pip install protobuf==3.6.0 --force`

`pip install grpcio-tools`

`nuget install Grpc.Tools` into known directory.

### Installing Protobuf Compiler

On Mac: `brew install protobuf`

On Windows & Linux: [See here](https://github.com/google/protobuf/blob/master/src/README.md).

## Running

1. Install pre-requisites.
2. Un-comment line 4 in `make.bat`, and set to correct Grpc.Tools sub-directory.
3. Run `make.bat`
4. In the generated `UnityToExternalGrpc.cs` file in the `UnitySDK/Assets/ML-Agents/Scripts/CommunicatorObjects` folder, you will need to add the following to the beginning of the file

```csharp
# if UNITY_EDITOR || UNITY_STANDALONE_WIN || UNITY_STANDALONE_OSX || UNITY_STANDALONE_LINUX
```
 and the following line to the end
 
 ```csharp
 #endif
 ```
This is to make sure the generated code does not try to access the Grpc library
on platforms that are not supported by Grpc.

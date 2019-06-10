# Unity ML-Agents Protobuf Definitions

Contains relevant definitions needed to generate probobuf files used in [ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents).

## Requirements

* grpc 1.14.1
* protobuf 3.6.0

## Set-up & Installation

Assume the ml-agents repository is checked out to a folder named $MLAGENTS_ROOT.
First we will install protobuf and grpcio-tools via your terminal.
**Note:** If you're using Anaconda, don't forget to activate the ml-agents environment first.

`pip install protobuf==3.6.0 --force`

`pip install grpcio-tools==1.11.1`

If you don't have it already, download the latest version of [nuget](https://www.nuget.org/downloads).
Navigate to your installation of nuget and run the following: 

`nuget install Grpc.Tools -OutputDirectory $MLAGENTS_ROOT\protobuf-definitions`

### Installing Protobuf Compiler

On Mac: `brew install protobuf`

## Running

1. Open a terminal. **Note:** If you're using Anaconda, don't forget to activate the ml-agents environment first.
2. Un-comment line 6 in `make.bat` (for Windows, use `make_for_win.bat`), and set to correct Grpc.Tools sub-directory.
3. Run the `.bat` from the terminal by navigating to `$MLAGENTS_ROOT\protobuf-definitions` and entering `make.bat` (for Windows, use `make_for_win.bat`)
4. Note any errors generated that may result from setting the wrong directory in step 2.
5. In the generated `UnityToExternalGrpc.cs` file in the `$MLAGENTS_ROOT/UnitySDK/Assets/ML-Agents/Scripts/CommunicatorObjects` folder, you will need to add the following to the beginning of the file:

```csharp
# if UNITY_EDITOR || UNITY_STANDALONE_WIN || UNITY_STANDALONE_OSX || UNITY_STANDALONE_LINUX
```
 and the following line to the end
 
 ```csharp
 #endif
 ```
This is to make sure the generated code does not try to access the Grpc library
on platforms that are not supported by Grpc.

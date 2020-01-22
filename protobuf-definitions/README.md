# Unity ML-Agents Protobuf Definitions

Contains relevant definitions needed to generate probobuf files used in [ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents).

## Requirements

* protobuf 3.6.0
* grpcio-tools 1.11.1
* Grpc.Tools 1.14.1

## Set-up & Installation

First we will follow these steps once install protobuf and grpcio-tools via your terminal.
Assume the ml-agents repository is checked out to a folder named $MLAGENTS_ROOT.
**Note:** If you're using Anaconda, don't forget to activate the ml-agents environment first.

`pip install protobuf==3.6.0 --force`

`pip install grpcio-tools==1.11.1`

`pip install mypy-protobuf`


#### On Windows

Download and install the latest version of [nuget](https://www.nuget.org/downloads).

#### On Mac

`brew install nuget`

#### On Linux

`sudo apt-get install nuget`


Navigate to your installation of nuget and run the following:

`nuget install Grpc.Tools -Version 1.14.1 -OutputDirectory $MLAGENTS_ROOT\protobuf-definitions`

## Running

Whenever you change the fields of a message, you must follow the steps below to create C# and Python files corresponding to the new message.

1. Open a terminal. **Note:** If you're using Anaconda, don't forget to activate the ml-agents environment first.
2. Un-comment line 7 in `make.sh` (for Windows, use `make_for_win.bat`), and set to correct Grpc.Tools sub-directory.
3. Run the protobuf generation script from the terminal by navigating to `$MLAGENTS_ROOT\protobuf-definitions` and entering `make.sh` (for Windows, use `make_for_win.bat`)
4. Note any errors generated that may result from setting the wrong directory in step 2.
5. In the generated `UnityToExternalGrpc.cs` file in the `$MLAGENTS_ROOT/com.unity.ml-agents/Runtime/Grpc/CommunicatorObjects` folder, check to see if you need to add the following to the beginning of the file:

```csharp
# if UNITY_EDITOR || UNITY_STANDALONE_WIN || UNITY_STANDALONE_OSX || UNITY_STANDALONE_LINUX
```
 and the following line to the end

 ```csharp
 #endif
 ```
This is to make sure the generated code does not try to access the Grpc library
on platforms that are not supported by Grpc.

Finally, re-install the mlagents packages by running the following commands from the same `$MLAGENTS_ROOT\protobuf-definitions` directory.

```
cd ..
cd ml-agents-envs
pip install -e .
cd ..
cd ml-agents
pip install -e .
mlagents-learn
```

The final line will test if everything was generated and installed correctly. If it worked, you should see the Unity logo.

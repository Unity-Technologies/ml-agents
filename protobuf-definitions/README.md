# Unity ML-Agents Protobuf Definitions

Contains relevant definitions needed to generate probobuf files used in [ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents).

## Requirements

* grpc 1.10.1
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

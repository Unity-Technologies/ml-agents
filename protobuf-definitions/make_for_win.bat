rem variables

rem GRPC-TOOLS required. Install with `nuget install Grpc.Tools`. 
rem Then un-comment and replace [DIRECTORY] with location of files.
rem For example, on Windows, you might have something like:
rem set COMPILER=Grpc.Tools.1.14.1/tools/windows_x64

set SRC_DIR=proto\mlagents\envs\communicator_objects
set DST_DIR_C=..\UnitySDK\Assets\ML-Agents\Scripts\CommunicatorObjects
set DST_DIR_P=..\ml-agents-envs
set PROTO_PATH=proto

set PYTHON_PACKAGE=mlagents\envs\communicator_objects

rem clean
rd /s /q %DST_DIR_C%
rd /s /q %DST_DIR_P%\%PYTHON_PACKAGE%
mkdir %DST_DIR_C%
mkdir %DST_DIR_P%\%PYTHON_PACKAGE%

rem generate proto objects in python and C#

for %%i in (%SRC_DIR%\*.proto) do (
    protoc --proto_path=proto --csharp_out=%DST_DIR_C% %%i
    protoc --proto_path=proto --python_out=%DST_DIR_P% %%i
)

rem grpc

set GRPC=unity_to_external.proto

%COMPILER%\protoc --proto_path=proto --csharp_out %DST_DIR_C% --grpc_out %DST_DIR_C% %SRC_DIR%\%GRPC% --plugin=protoc-gen-grpc=%COMPILER%\grpc_csharp_plugin.exe
python3 -m grpc_tools.protoc --proto_path=proto --python_out=%DST_DIR_P% --grpc_python_out=%DST_DIR_P% %SRC_DIR%\%GRPC%

rem Generate the init file for the python module
rem rm -f $DST_DIR_P/$PYTHON_PACKAGE/__init__.py
setlocal enabledelayedexpansion
for %%i in (%DST_DIR_P%\%PYTHON_PACKAGE%\*.py) do ( 
set FILE=%%~ni
rem echo from .$(basename $FILE) import * >> $DST_DIR_P/$PYTHON_PACKAGE/__init__.py
echo from .!FILE! import * >> %DST_DIR_P%\%PYTHON_PACKAGE%\__init__.py
)


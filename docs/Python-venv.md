# Installing and Running ML-Agents in a virtual environment

__Requirement - Python 3.6 must be installed on the server. Python 3.6 can be [here](https://www.python.org/downloads/)__ 

## Mac OS X Setup

1. Create a folder where the virtual environments will live ` $ mkdir ~/python-venvs `
1. To create a new environment named `test-env` execute `$ python3 -m venv ~/python-envs/test-env`  
1. To activate the environment execute `$ source ~/python-envs/test-env/bin/activate`
1. Install ML-Agents package using `$ pip3 install mlagents`
1. To deactivate the environment execute `$ deactivate `

## Ubuntu Setup 

1. Install the python3-venv package using `$ sudo apt-get install python3-venv`

Now follow the steps in the Mac OS X installation.

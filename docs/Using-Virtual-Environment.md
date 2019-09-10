# Using Virtual Environment

## What is a Virtual Environment?
A Virtual Environment is a walled garden for Python projects.  

## Why should I use a Virtual Environment?
A Virtual Environment keeps all dependencies for the project separate from dependencies 
of other projects. This has a few advantages:
1. Firstly, it makes dependency management for the project easy.
1. Secondly, it enables using and testing of different library versions by quickly 
spinning up a new environment and verifying the compatibility of the code with the
different version. 

Requirement - Python 3.6 must be installed on the machine you would like 
to run ML-Agents on (either local laptop/desktop or remote server). Python 3.6 can be 
installed from [here](https://www.python.org/downloads/). 

## Mac OS X Setup

1. Create a folder where the virtual environments will live ` $ mkdir ~/python-venvs `
1. To create a new environment named `test-env` execute `$ python3 -m venv ~/python-envs/test-env`  
1. To activate the environment execute `$ source ~/python-envs/test-env/bin/activate`
1. Install ML-Agents package using `$ pip3 install mlagents`
1. To deactivate the environment execute `$ deactivate `

## Ubuntu Setup 

1. Install the python3-venv package using `$ sudo apt-get install python3-venv`
1. Follow steps 2-5 in the Mac OS X installation.

## Windows Setup

Coming Soon.
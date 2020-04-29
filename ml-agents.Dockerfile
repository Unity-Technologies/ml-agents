FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
LABEL maintainer=ml-agents@unity3d.com

RUN apt update && apt -y upgrade

# install relevant tools
RUN apt install -y wget curl vim tmux git gdebi-core build-essential python3-pip unzip
RUN python3 -m pip install --upgrade pip
RUN pip install setuptools==45.1.0

# install libxfont1, xvfb older version (used for environments with graphics)
RUN wget http://security.ubuntu.com/ubuntu/pool/main/libx/libxfont/libxfont1_1.5.1-1ubuntu0.16.04.4_amd64.deb
RUN wget http://security.ubuntu.com/ubuntu/pool/universe/x/xorg-server/xvfb_1.18.4-0ubuntu0.7_amd64.deb
RUN yes | gdebi libxfont1_1.5.1-1ubuntu0.16.04.4_amd64.deb
RUN yes | gdebi xvfb_1.18.4-0ubuntu0.7_amd64.deb
RUN apt-get install -y --no-install-recommends htop mesa-utils xorg-dev xorg libglvnd-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# install the latest version of ml-agents (from the master branch )
RUN git clone https://github.com/Unity-Technologies/ml-agents.git /ml-agents
WORKDIR /ml-agents
RUN git pull origin master
WORKDIR /ml-agents/ml-agents-envs
RUN pip install -e .
WORKDIR /ml-agents/ml-agents
RUN pip install -e .

# to build the docker image execute the following command in the folder containing this file
# docker build -t ml-agents:latest -f ml-agents.Dockerfile .

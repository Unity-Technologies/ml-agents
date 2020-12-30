# From https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/supported-tags.md
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN yes | unminimize

RUN echo "deb http://packages.cloud.google.com/apt cloud-sdk-xenial main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN wget https://packages.cloud.google.com/apt/doc/apt-key.gpg && apt-key add apt-key.gpg
RUN apt-get update && \
  apt-get install -y --no-install-recommends wget curl tmux vim git gdebi-core \
  build-essential python3-pip unzip google-cloud-sdk htop mesa-utils xorg-dev xorg \
  libglvnd-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev xvfb && \
  wget http://security.ubuntu.com/ubuntu/pool/main/libx/libxfont/libxfont1_1.5.1-1ubuntu0.16.04.4_amd64.deb && \
  yes | gdebi libxfont1_1.5.1-1ubuntu0.16.04.4_amd64.deb
RUN python3 -m pip install --upgrade pip
RUN pip install setuptools==41.0.0

ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

#checkout ml-agents for SHA
RUN mkdir /ml-agents
WORKDIR /ml-agents
ARG SHA
RUN git init
RUN git remote add origin https://github.com/Unity-Technologies/ml-agents.git
RUN git fetch --depth 1 origin $SHA
RUN git checkout FETCH_HEAD
RUN pip install -e /ml-agents/ml-agents-envs
RUN pip install -e /ml-agents/ml-agents

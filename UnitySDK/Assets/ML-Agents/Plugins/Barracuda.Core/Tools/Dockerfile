FROM ubuntu:16.04

ENV DST=/Barracuda
ENV DST_TOOLS="$DST/Tools"
ENV DST_TMP="$DST/Temp"

RUN apt-get update
RUN apt-get install -y ca-certificates
RUN apt-get install -y protobuf-compiler libprotoc-dev build-essential
RUN apt-get install -y python3 python3-setuptools python3-dev
RUN easy_install3 pip

RUN mkdir -p "$DST_TOOLS"
RUN mkdir -p "$DST_TMP"

WORKDIR "$DST"

COPY requirements.txt "$DST/"

RUN pip install -r requirements.txt


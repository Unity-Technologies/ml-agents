# Use an official Python runtime as a parent image
FROM python:3.6-slim

RUN apt-get update && apt-get -y upgrade

ADD python/requirements.txt .
RUN pip install --trusted-host pypi.python.org -r requirements.txt

WORKDIR /execute
COPY python /execute/python

ENTRYPOINT ["python", "python/learn.py"]

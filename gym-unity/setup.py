#!/usr/bin/env python

from setuptools import setup, find_packages

VERSION = "0.10.1"

setup(
    name="gym_unity",
    version=VERSION,
    description="Unity Machine Learning Agents Gym Interface",
    license="Apache License 2.0",
    author="Unity Technologies",
    author_email="ML-Agents@unity3d.com",
    url="https://github.com/Unity-Technologies/ml-agents",
    packages=find_packages(),
    install_requires=["gym", "mlagents_envs=={}".format(VERSION)],
)

#!/usr/bin/env python

from setuptools import setup, find_packages

with open("../VERSION") as f:
    version_string = f.read().strip()

setup(
    name="gym_unity",
    version="0.4.7",
    description="Unity Machine Learning Agents Gym Interface",
    license="Apache License 2.0",
    author="Unity Technologies",
    author_email="ML-Agents@unity3d.com",
    url="https://github.com/Unity-Technologies/ml-agents",
    packages=find_packages(),
    install_requires=["gym", "mlagents_envs=={}".format(version_string)],
)

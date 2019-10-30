#!/usr/bin/env python

import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install

VERSION = "0.11.0"


class VerifyVersionCommand(install):
    """
    Custom command to verify that the git tag matches our version
    See https://circleci.com/blog/continuously-deploying-python-packages-to-pypi-with-circleci/
    """

    description = "verify that the git tag matches our version"

    def run(self):
        tag = os.getenv("CIRCLE_TAG")

        if tag != VERSION:
            info = "Git tag: {0} does not match the version of this app: {1}".format(
                tag, VERSION
            )
            sys.exit(info)


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
    cmdclass={"verify": VerifyVersionCommand},
)

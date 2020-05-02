#!/usr/bin/env python

import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install
import gym_unity

VERSION = gym_unity.__version__
EXPECTED_TAG = gym_unity.__release_tag__


class VerifyVersionCommand(install):
    """
    Custom command to verify that the git tag is the expected one for the release.
    Based on https://circleci.com/blog/continuously-deploying-python-packages-to-pypi-with-circleci/
    This differs slightly because our tags and versions are different.
    """

    description = "verify that the git tag matches our version"

    def run(self):
        tag = os.getenv("CIRCLE_TAG")

        if tag != EXPECTED_TAG:
            info = "Git tag: {0} does not match the expected tag of this app: {1}".format(
                tag, EXPECTED_TAG
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

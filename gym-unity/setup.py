#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='gym_unity',
      version='0.4.1',
      description='Unity Machine Learning Agents Gym Interface',
      license='Apache License 2.0',
      author='Unity Technologies',
      author_email='ML-Agents@unity3d.com',
      url='https://github.com/Unity-Technologies/ml-agents',
      packages=find_packages(),
      install_requires=['gym', 'mlagents_envs==0.8.1']
      )

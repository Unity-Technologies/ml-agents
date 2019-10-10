from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name="mlagents_envs",
    version="0.10.1",
    description="Unity Machine Learning Agents Interface",
    url="https://github.com/Unity-Technologies/ml-agents",
    author="Unity Technologies",
    author_email="ML-Agents@unity3d.com",
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["mlagents.envs", "mlagents.envs.communicator_objects"],  # Required
    zip_safe=False,
    install_requires=[
        "cloudpickle",
        "grpcio>=1.11.0",
        "numpy>=1.13.3,<2.0",
        "Pillow>=4.2.1",
        "protobuf>=3.6",
    ],
    python_requires=">=3.5",
)

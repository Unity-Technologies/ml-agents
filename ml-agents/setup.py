from io import open
import os
import sys

from setuptools import setup, find_packages
from setuptools.command.install import install
import mlagents.trainers

VERSION = mlagents.trainers.__version__

here = os.path.abspath(os.path.dirname(__file__))


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


# Get the long description from the README file
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="mlagents",
    version=VERSION,
    description="Unity Machine Learning Agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
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
    # find_namespace_packages will recurse through the directories and find all the packages
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    zip_safe=False,
    install_requires=[
        # Test-only dependencies should go in test_requirements.txt, not here.
        "grpcio>=1.11.0",
        "h5py>=2.9.0",
        "jupyter",
        "matplotlib",
        "mlagents_envs=={}".format(VERSION),
        "numpy>=1.13.3,<2.0",
        "Pillow>=4.2.1",
        "protobuf>=3.6",
        "pyyaml",
        "tensorflow>=1.7,<2.1",
        'pypiwin32==223;platform_system=="Windows"',
    ],
    python_requires=">=3.6.1",
    entry_points={
        "console_scripts": [
            "mlagents-learn=mlagents.trainers.learn:main",
            "mlagents-run-experiment=mlagents.trainers.run_experiment:main",
        ]
    },
    cmdclass={"verify": VerifyVersionCommand},
)

import os
import sys

from setuptools import setup, find_packages
from setuptools.command.install import install
import mlagents.trainers

VERSION = mlagents.trainers.__version__
EXPECTED_TAG = mlagents.trainers.__release_tag__

here = os.path.abspath(os.path.dirname(__file__))


class VerifyVersionCommand(install):
    """
    Custom command to verify that the git tag is the expected one for the release.
    Originally based on https://circleci.com/blog/continuously-deploying-python-packages-to-pypi-with-circleci/
    This differs slightly because our tags and versions are different.
    """

    description = "verify that the git tag matches our version"

    def run(self):
        tag = os.getenv("GITHUB_REF", "NO GITHUB TAG!").replace("refs/tags/", "")

        if tag != EXPECTED_TAG:
            info = "Git tag: {} does not match the expected tag of this app: {}".format(
                tag, EXPECTED_TAG
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
        "Programming Language :: Python :: 3.8",
    ],
    # find_namespace_packages will recurse through the directories and find all the packages
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    zip_safe=False,
    install_requires=[
        # Test-only dependencies should go in test_requirements.txt, not here.
        "grpcio>=1.11.0",
        "h5py>=2.9.0",
        f"mlagents_envs=={VERSION}",
        "numpy>=1.13.3,<2.0",
        "Pillow>=4.2.1",
        "protobuf>=3.6",
        "pyyaml>=3.1.0",
        # Windows ver. of PyTorch doesn't work from PyPi. Installation:
        # https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md#windows-installing-pytorch
        'torch>=1.6.0,<1.8.0;platform_system!="Windows"',
        "tensorboard>=1.15",
        # cattrs 1.1.0 dropped support for python 3.6.
        "cattrs>=1.0.0,<1.1.0",
        "attrs>=19.3.0",
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
    extras_require={"tensorflow": ["tensorflow>=1.14,<3.0", "six>=1.12.0"]},
)

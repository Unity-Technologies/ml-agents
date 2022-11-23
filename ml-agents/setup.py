import os
import sys

from setuptools import setup, find_packages
from setuptools.command.install import install
from mlagents.plugins import ML_AGENTS_STATS_WRITER, ML_AGENTS_TRAINER_TYPE
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
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
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
        # https://github.com/Unity-Technologies/ml-agents/blob/release_20_docs/docs/Installation.md#windows-installing-pytorch
        # Torch only working on python 3.9 for 1.8.0 and above. Details see:
        # https://github.com/pytorch/pytorch/issues/50014
        "torch>=1.8.0,<=1.11.0;(platform_system!='Windows' and python_version>='3.9')",
        "torch>=1.6.0,<1.9.0;(platform_system!='Windows' and python_version<'3.9')",
        "tensorboard>=1.15",
        # cattrs 1.1.0 dropped support for python 3.6, but 1.0.0 doesn't work for python 3.9
        # Since there's no version that supports both, we have to draw the line somwehere.
        "cattrs<1.1.0; python_version<'3.8'",
        "cattrs>=1.1.0,<1.7; python_version>='3.8'",
        "attrs>=19.3.0",
        'pypiwin32==223;platform_system=="Windows"',
        "importlib_metadata==4.4; python_version<'3.8'",
    ],
    python_requires=">=3.8.13,<=3.10.8",
    entry_points={
        "console_scripts": [
            "mlagents-learn=mlagents.trainers.learn:main",
            "mlagents-run-experiment=mlagents.trainers.run_experiment:main",
        ],
        # Plugins - each plugin type should have an entry here for the default behavior
        ML_AGENTS_STATS_WRITER: [
            "default=mlagents.plugins.stats_writer:get_default_stats_writers"
        ],
        ML_AGENTS_TRAINER_TYPE: [
            "default=mlagents.plugins.trainer_type:get_default_trainer_types"
        ],
    },
    # TODO: Remove this once mypy stops having spurious setuptools issues.
    cmdclass={"verify": VerifyVersionCommand},  # type: ignore
)

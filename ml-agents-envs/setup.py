import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install
import mlagents_envs

VERSION = mlagents_envs.__version__
EXPECTED_TAG = mlagents_envs.__release_tag__

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
    name="mlagents_envs",
    version=VERSION,
    description="Unity Machine Learning Agents Interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Unity-Technologies/ml-agents",
    author="Unity Technologies",
    author_email="ML-Agents@unity3d.com",
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
    ],
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests", "colabs", "*.ipynb"]
    ),
    zip_safe=False,
    install_requires=[
        "cloudpickle",
        "grpcio>=1.11.0,<=1.48.2",
        "Pillow>=4.2.1",
        "protobuf>=3.6,<3.21",
        "pyyaml>=3.1.0",
        "gym>=0.21.0",
        "pettingzoo==1.15.0",
        "numpy>=1.23.5,<1.24.0",
        "filelock>=3.4.0",
    ],
    python_requires=">=3.10.1,<=3.10.12",
    # TODO: Remove this once mypy stops having spurious setuptools issues.
    cmdclass={"verify": VerifyVersionCommand},  # type: ignore
)

from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='mlagents',
    version='0.6.0',
    description='Unity Machine Learning Agents',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Unity-Technologies/ml-agents',
    author='Unity Technologies',
    author_email='ML-Agents@unity3d.com',

    classifiers=[
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6'
    ],

    packages=find_packages(exclude=['tests', 'tests.*', '*.tests', '*.tests.*']),  # Required

    install_requires=[
        'tensorflow>=1.7,<1.8',
        'Pillow>=4.2.1',
        'matplotlib',
        'numpy>=1.13.3,<=1.14.5',
        'jupyter',
        'pytest>=3.2.2,<4.0.0',
        'docopt',
        'pyyaml',
        'protobuf>=3.6,<3.7',
        'grpcio>=1.11.0,<1.12.0'],

    python_requires=">=3.6,<3.7",

    entry_points={
        'console_scripts': [
            'mlagents-learn=mlagents.trainers.learn:main',
        ],
    },
)

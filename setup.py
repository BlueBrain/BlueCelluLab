#!/usr/bin/env python
""" BGLibPy setup """

import sys

import setuptools

import versioneer

if sys.version_info[:2] < (3, 7):
    raise RuntimeError("Python version >= 3.7 required.")

setuptools.setup(
    name="bglibpy",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=setuptools.find_packages(include=['bglibpy', 'bglibpy.*']),
    author="Blue Brain Project, EPFL",
    description="The Pythonic Blue Brain simulator access",
    license="Apache 2.0",
    install_requires=[
        "NEURON>=8.0.2",
        "numpy>=1.8.0",
        "matplotlib>=3.0.0",
        "pandas>=1.0.0",
        "bluepysnap>=1.0.5,<2.0.0",
        "pydantic>=1.10.2"
        ],
    keywords=[
        'computational neuroscience',
        'simulation',
        'analysis',
        'parameters',
        'Blue Brain Project'],
    url="http://bbpteam.epfl.ch/project/issues/projects/BGLPY",
    download_url="https://bbpgitlab.epfl.ch/cells/bglibpy.git",
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'License :: Proprietary',
        'Operating System :: POSIX',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3',
        'Topic :: Utilities',
    ],
    include_package_data=True,
)

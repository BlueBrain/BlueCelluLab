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
    packages=setuptools.find_packages(exclude=('examples', )),
    author="Werner Van Geit",
    author_email="werner.vangeit@epfl.ch",
    description="The Pythonic Blue Brain simulator access",
    license="BBP-internal-confidential",
    dependency_links=["https://bbpteam.epfl.ch/repository/devpi/bbprelman/"
                      "dev/+simple/bluepy/"],
    install_requires=[
        "bluepy[bbp]>=2.4.2",
        "cachetools",
        "numpy>=1.8.0",
        "matplotlib>=3.0.0",
        "bluepy-configfile>=0.1.18",
        "pandas>=1.0.0",
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
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'License :: Proprietary',
        'Operating System :: POSIX',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Utilities',
    ],
)

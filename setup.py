#!/usr/bin/env python
""" bglibpy setup """

import setuptools
import versioneer

setuptools.setup(
    name="bglibpy",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=['bglibpy'],
    author="Werner Van Geit",
    author_email="werner.vangeit@epfl.ch",
    description="The Pythonic Blue Brain simulator access",
    license="BBP-internal-confidential",
    dependency_links=["https://bbpteam.epfl.ch/repository/devpi/bbprelman/dev/+simple/bluepy/"],
    install_requires=["bluepy[bbp]"],
    keywords=(
        'computational neuroscience',
        'simulation',
        'analysis',
        'visualization',
        'parameters',
        'BlueBrainProject'),
    url="http://bbpteam.epfl.ch/project/issues/projects/BGLPY",
    download_url="https://bbpcode.epfl.ch/code/#/admin/projects/sim/BGLibPy,tags",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'License :: Proprietary',
        'Operating System :: POSIX',
        'Topic :: Scientific/Engineering',
        'Topic :: Utilities',
    ],
) 

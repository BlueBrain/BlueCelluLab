#!/usr/bin/env python
""" bglibpy setup """

import setuptools

setuptools.setup(
    name="bglibpy",
    version=3.0,
    packages=['bglibpy'],
    author="Werner Van Geit",
    author_email="werner.vangeit@epfl.ch",
    description="The Pythonic Blue Brain simulator access",
    license="BBP-internal-confidential",
    keywords=('computational neuroscience',
              'simulation',
              'analysis',
              'visualization',
              'parameters',
              'BlueBrainProject'),
    url="http://bluebrain.epfl.ch",
    classifiers=['Development Status :: 3 - Alpha',
                 'Environment :: Console',
                 'License :: Proprietary',
                 'Operating System :: POSIX',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Utilities',
                 ],
)

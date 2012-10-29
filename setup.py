#!/usr/bin/env python

'''Setup script of bglibpy'''

import distutils.core
from distutils.command import build as build_module
#from setuptools import find_packages
import subprocess

class build(build_module.build):
    '''Own definition of build function'''
    def run(self):
        subprocess.call(['nrnivmodl','modlib'])
        build_module.build.run(self)

distutils.core.setup(name='bglibpy',
    version='0.0.1',
    description='Python Wrapper of BgLib @ Blue Brain Project',
    author='Werner Van Geit',
    author_email='werner.vangeit@epfl.ch',
    url='http://bbpteam.epfl.ch/documentation/bbp_example',
    py_modules=["bglibpy"],
    ext_package = "modlib",
    package_data = { "modlib": ["libnrnmech.so", "libnrnmech.so.0.0.0"] },
    package_dir = { "modlib": "i686/.libs" },
    cmdclass = {'build': build}
)

# Copyright 2012-2023 Blue Brain Project / EPFL

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import setuptools


if sys.version_info[:2] < (3, 7):
    raise RuntimeError("Python version >= 3.7 required.")

# Read the README.rst file
with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bluecellulab",
    use_scm_version={
        'version_scheme': 'python-simplified-semver',
        'local_scheme': 'no-local-version'
    },
    setup_requires=['setuptools_scm'],
    packages=setuptools.find_packages(include=['bluecellulab', 'bluecellulab.*']),
    author="Blue Brain Project, EPFL",
    description="The Pythonic Blue Brain simulator access",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    license="Apache2.0",
    install_requires=[
        "NEURON>=8.0.2,<9.0.0",
        "numpy>=1.8.0,<2.0.0",
        "matplotlib>=3.0.0,<4.0.0",
        "pandas>=1.0.0,<3.0.0",
        "bluepysnap>=1.0.5,<2.0.0",
        "pydantic>=1.10.2,<2.0.0",
    ],
    keywords=[
        'computational neuroscience',
        'simulation',
        'analysis',
        'parameters',
        'Blue Brain Project'],
    url="https://github.com/BlueBrain/BlueCelluLab",
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3',
        'Topic :: Utilities',
    ],
    include_package_data=True,
)

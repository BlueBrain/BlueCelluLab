import sys
import setuptools


if sys.version_info[:2] < (3, 7):
    raise RuntimeError("Python version >= 3.7 required.")

setuptools.setup(
    name="bluecellulab",
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    packages=setuptools.find_packages(include=['bluecellulab', 'bluecellulab.*']),
    author="Blue Brain Project, EPFL",
    description="The Pythonic Blue Brain simulator access",
    license="Apache2.0",
    install_requires=[
        "NEURON>=8.0.2,<9.0.0",
        "numpy>=1.8.0,<2.0.0",
        "matplotlib>=3.0.0,<4.0.0",
        "pandas>=1.0.0,<2.0.0",
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
        'License :: Proprietary',
        'Operating System :: POSIX',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3',
        'Topic :: Utilities',
    ],
    include_package_data=True,
)

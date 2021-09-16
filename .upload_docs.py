#!/bin/env python

from __future__ import print_function

import sys
import os
import contextlib
import datetime
import pkg_resources

metadata_template = \
    """---
packageurl: https://bbpteam.epfl.ch/repository/devpi/bbprelman/dev/bglibpy
major: {major_version}
description: Simulate small amount of cells from large simulation
repository: https://bbpgitlab.epfl.ch/cells/bglibpy
externaldoc: https://bbpteam.epfl.ch/documentation/projects/BGLibPy/latest/index.html
updated: {date}
maintainers: Werner Van Geit
name: BGLibPy
license: BBP-internal-confidential
issuesurl: https://bbpteam.epfl.ch/project/issues/projects/BGLPY
version: {version}
contributors: Werner Van Geit, Eilif Muller, Benjamin Torben-Nielsen, Anil Tuncel, BBP
minor: {minor_version}
---
"""


@contextlib.contextmanager
def cd(dir_name):
    """Change directory"""
    old_cwd = os.getcwd()
    os.chdir(dir_name)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def main():
    """Main"""
    doc_dir = sys.argv[1]

    doc_dir = os.path.abspath(doc_dir)

    with cd(doc_dir):
        print('Reading BGLibPy version ...')
        bglibpy_version = pkg_resources.get_distribution('bglibpy').version
        bglibpy_major_version = bglibpy_version.split('.')[0]
        bglibpy_minor_version = bglibpy_version.split('.')[1]
        print('BGLibPy version is: %s' % bglibpy_version)

        finished_filename = '.doc_version'

        if os.path.exists(finished_filename):
            os.remove(finished_filename)

        metadata_filename = 'metadata.md'

        metadata_content = metadata_template.format(
            major_version=bglibpy_major_version,
            minor_version=bglibpy_minor_version,
            date=datetime.datetime.now().strftime("%d/%m/%y"),
            version=bglibpy_version)

        print('Created metadata: %s' % metadata_content)

        with open(metadata_filename, 'w') as metadata_file:
            metadata_file.write(metadata_content)

        print('Wrote metadata to: %s' % metadata_filename)

        with open(finished_filename, 'w') as finished_file:
            finished_file.write(bglibpy_version)

        print('Wrote doc version info to: %s' % finished_filename)


if __name__ == '__main__':
    main()

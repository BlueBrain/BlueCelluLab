#!/bin/env python

from __future__ import print_function

import sys
import os
import contextlib
import datetime
import shutil

import sh


metadata_template = \
    """---
packageurl: https://bbpteam.epfl.ch/repository/devpi/bbprelman/dev/bglibpy
major: {major_version}
description: Simulate small amount of cells from large simulation
repository: https://bbpcode.epfl.ch/code/#/admin/projects/sim/BGLibPy
externaldoc: https://bbpcode.epfl.ch/documentation/#BGLibpy
updated: {date}
maintainers: Werner Van Geit
name: BGLibPy
license: BBP-internal-confidential
issuesurl: https://bbpteam.epfl.ch/project/issues/projects/BGLPY
version: {version}
contributors: Werner Van Geit, Eilif Muller, Benjamin Torben-Nielsen, BBP
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
    doc_repo = sys.argv[1]
    doc_dir = sys.argv[2]

    print('Cloning jekylltest')
    if not os.path.exists('jekylltest'):
        sh.git('clone', '-b', 'master', '--depth=1', doc_repo)

    with cd('jekylltest'):

        print('Reading BGLibPy version ...')
        import bglibpy
        bglibpy_version = bglibpy.__version__
        bglibpy_major_version, bglibpy_minor_version, _ = \
            bglibpy.__version__.split('.')
        print('BGLibPy version is: %s' % bglibpy_version)

        doc_subdir = "BGLibPy-%d.%d" % (int(bglibpy_major_version),
                                        int(bglibpy_minor_version))

        print('Doc subdir: %s' % doc_subdir)

        finished_filename = os.path.join(doc_subdir, '.doc_version')

        if os.path.exists(finished_filename):
            os.remove(finished_filename)

        metadata_filename = os.path.join('_projects', doc_subdir)

        print('Pulling latest jekylltest ...')
        sh.git('reset', '--hard', 'origin/master')
        sh.git('pull')

        bglibpy_prev_version = None

        if os.path.exists(finished_filename):
            with open(finished_filename) as finished_file:
                bglibpy_prev_version = finished_file.read()

        print("Previous BGLibPy version: %s" % bglibpy_prev_version)

        if bglibpy_prev_version is None or \
                bglibpy_prev_version != bglibpy_version:

            print('Removing old docs ...')

            import glob
            for subdir in glob.glob('BGLibPy-*'):
                sh.git('rm', '-r', '--ignore-unmatch', subdir)
                sh.git(
                    'rm',
                    '--ignore-unmatch',
                    os.path.join(
                        '_projects',
                        subdir))

            sh.git('rm', '-r', '--ignore-unmatch', doc_subdir)
            sh.git('rm', '--ignore-unmatch', metadata_filename)

            print('Copying %s to %s ...' % (doc_dir, doc_subdir))
            shutil.copytree(doc_dir, doc_subdir)

            metadata_content = metadata_template.format(
                major_version=bglibpy_major_version,
                minor_version=bglibpy_minor_version,
                date=datetime.datetime.now().strftime("%d/%m/%y"),
                version='%d.%d' %
                (int(bglibpy_major_version),
                 int(bglibpy_minor_version)))

            print('Created metadata: %s' % metadata_content)

            with open(metadata_filename, 'w') as metadata_file:
                metadata_file.write(metadata_content)

            print('Wrote metadata to: %s' % metadata_filename)

            with open(finished_filename, 'w') as finished_file:
                finished_file.write(bglibpy_version)

            print('Wrote doc version info to: %s' % finished_filename)

            sh.git('add', metadata_filename)
            sh.git('add', doc_subdir)
            sh.git('add', finished_filename)

            print('Added doc to repo')

            untracked_status = sh.git(
                'status',
                '--porcelain',
                '--untracked-files=no')

            if len(untracked_status) > 0:
                print('Committing doc changes ...')
                sh.git('config', 'user.email', 'bbprelman@epfl.ch')
                sh.git(
                    'commit',
                    '-m',
                    'Added documentation for %s' %
                    doc_subdir)

                print('Pushing doc changes ...')
                sh.git('push', 'origin', 'master')
            else:
                print('No doc changes found, not committing')
        else:
            print('Doc dir of version already exists, not uploading anything')

if __name__ == '__main__':
    main()

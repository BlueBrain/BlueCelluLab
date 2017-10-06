#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=E1101,W0201

"""Testing the loading of the BGLibPy"""


from nose.plugins.attrib import attr


@attr('unit')
def test_moduleload():
    """BGLibPy: Test the loading of the bglibpy module"""
    import bglibpy  # NOQA

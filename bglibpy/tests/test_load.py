#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=E1101,W0201

"""Testing the loading of the BGLibPy"""


from nose.plugins.attrib import attr
import nose.tools as nt

@attr('unit')
def test_moduleload():
    """BGLibPy: Test the loading of the bglibpy module"""
    import bglibpy  # NOQA

@attr('unit')
def test_verbose_env():
    """BGLibPy: Test env verbose level var"""
    import os
    os.environ['BGLIBPY_VERBOSE_LEVEL'] = '10'

    import bglibpy  # NOQA

    bglibpy.set_verbose_from_env()

    nt.assert_equal(bglibpy.ENV_VERBOSE_LEVEL, 10)
    nt.assert_equal(bglibpy.VERBOSE_LEVEL, 10)

    del os.environ['BGLIBPY_VERBOSE_LEVEL']
    
    bglibpy.set_verbose_from_env()
    nt.assert_equal(bglibpy.ENV_VERBOSE_LEVEL, None)
    nt.assert_equal(bglibpy.VERBOSE_LEVEL, 10)

    bglibpy.set_verbose(0)
    nt.assert_equal(bglibpy.VERBOSE_LEVEL, 0)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=E1101,W0201

"""Testing the loading of the BGLibPy"""

import nose.tools as nt

def test_moduleload():
    """BGLibPy: Test the loading of the bglibpy module"""
    import bglibpy
    nt.assert_not_equal(bglibpy.__file__, "")

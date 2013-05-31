#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=E1101,W0201

"""Unit tests for BGLib functionality"""

import nose.tools as nt
#import math
import bglibpy

class TestBGLibBaseClass1(object):
    """Test BGLib base class"""
    def setup(self):
        """Setup"""
        self.taper_cell = bglibpy.Cell("examples/tapertest_cells/taper_cell.hoc", "examples/tapertest_cells")
        self.notaper_cell = bglibpy.Cell("examples/tapertest_cells/notaper_cell.hoc", "examples/tapertest_cells")

    def teardown(self):
        """Teardown"""
        del self.taper_cell
        del self.notaper_cell

    def test_tapering(self):
        """Test if tapering in Cell.hoc works correctly"""
        axon0_taper_diams = [segment.diam for segment in self.taper_cell.axonal[0]]
        nt.assert_equal(axon0_taper_diams, [1.9, 1.7, 1.5, 1.3, 1.1])

        axon1_taper_diams = [segment.diam for segment in self.taper_cell.axonal[1]]
        nt.assert_equal(axon1_taper_diams,
                [0.9642857142857143, 0.8928571428571429,
                    0.8214285714285714, 0.75,
                    0.6785714285714286, 0.6071428571428571,
                    0.5357142857142857])

        myelin_taper_diams = [segment.diam for segment in self.taper_cell.cell.getCell().myelin[0]]
        nt.assert_equal(myelin_taper_diams, [0.25]*21)


        axon0_notaper_diams = [segment.diam for segment in self.notaper_cell.axonal[0]]
        nt.assert_equal(axon0_notaper_diams, [0.7484678706370597])

        axon1_notaper_diams = [segment.diam for segment in self.notaper_cell.axonal[1]]
        nt.assert_equal(axon1_notaper_diams, [0.14187096846614167])


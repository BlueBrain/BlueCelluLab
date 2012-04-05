#!/usr/bin/env python

import sys
sys.path.append("..")
import bglibpy
import pylab

def main():
    #cell = bglibpy.Cell("cAC.hoc","/bgscratch/bbp/build/morphologies/ascii/tkb060509a2_ch3_mc_n_el_100x_1.asc")
    #cell = bglibpy.Cell("cAC.hoc","/bgscratch/bbp/build/morphologies/ascii/rp100426-2_idA.asc") #stuck
    #cell = bglibpy.Cell("bAC.hoc","/bgscratch/bbp/build/morphologies/ascii/C011098A-I3.asc")
    #cell = bglibpy.Cell("cADpyr2280_L23_PC_2_dend-C031000B-P3_axon-C170797A-P2.hoc","/bgscratch/bbp/build/morphologies/ascii/")
    cell = bglibpy.Cell("cADpyr2280_L5_TTPC2_5_dend-ch150801A1_axon-tr050310_5_c1.hoc","/bgscratch/bbp/build/morphologies/ascii/")
    #cell = bglibpy.Cell("cADpyr228_L5TTPC_5_C060110A3.hoc","/bgscratch/bbp/release/02.05.11/morphologies/ascii")
    hyp_level = -0.005
    #dep_level = 0.05
    dep_level = 0.9
    cell.addRamp(0, 9000, hyp_level, hyp_level, dt=1.0)
    cell.addRamp(100, 900, dep_level-hyp_level, dep_level-hyp_level, dt=1.0)
    cell.addAllSectionsVoltageRecordings()
    cell.addPlotWindow('soma(0.5)._ref_v',ylim=[-100,100])
    #bglibpy.calculateAllSynapticAttenuations(cell)
    #cell.addAllSynapses()
    cell.showDendrogram(variable="v")
    #cell.activateDendrogram()
    #cell.addPlotWindow('soma(0.5)._ref_ina_NaTs2_t',ylim=[-1,1])
    #cell.addPlotWindow('soma(0.5)._ref_ina_Nap_Et2',ylim=[-1,1])
    simulation = bglibpy.Simulation(verbose_level=0)
    simulation.addCell(cell)
    simulation.run(115)

    all_voltage = cell.getAllSectionsVoltageRecordings()
    pylab.figure()
    for section_name in all_voltage:
        pylab.plot(cell.getTime(), all_voltage[section_name])

    pylab.plot(cell.getTime(), cell.getSomaVoltage())
    pylab.draw()

    raw_input()

if __name__ == "__main__":
    main()

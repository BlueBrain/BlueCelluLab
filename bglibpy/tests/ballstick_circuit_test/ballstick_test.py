"""Test script 2"""

#import sys
#sys.path = ["/home/vangeit/local/bglibpy/lib64/python2.6/site-packages"]+ sys.path
import bglibpy


def main():
    """Main"""
    cell = bglibpy.Cell("ballstick.hoc", "./")
    print("Loaded", cell, "OK")
    print(
        "Soma length %f, diameter %f, area %f" %
        (cell.soma.L,
         cell.soma.diam,
         bglibpy.neuron.h.area(
             0.5,
             sec=cell.soma)))
    print(
        "Dend length %f, diameter %f, area %f" %
        (cell.basal[0].L,
         cell.basal[0].diam,
         bglibpy.neuron.h.area(
            0.5,
            sec=cell.basal[0])))
    #print "Axon length %f, diameter %f, area %f" % (cell.axonal[0].L, cell.axonal[0].diam, bglibpy.neuron.h.area(0.5, sec=cell.axonal[0]))
    #print "Axon length %f, diameter %f, area %f" % (cell.axonal[1].L, cell.axonal[1].diam, bglibpy.neuron.h.area(0.5, sec=cell.axonal[1]))
    bglibpy.neuron.h.topology()


if __name__ == "__main__":
    main()

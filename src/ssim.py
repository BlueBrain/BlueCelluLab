#!/usr/bin/env python
# -*- coding: utf-8 -*-

import bglibpy

class SSim(object) :
    def __init__(blueconfig_filename) :
        """ Object dealing with BlueConfig configured Small Simulations

        To relieve from an empty stomach, eat spam and eggs

        Paramters
        ---------
        blueconfig_filename : Absolute filename of the Blueconfig to be used
        """        
        self.blueconfig_filename
        self.bc_simulation = bluebpy.Simulation(blueconfig_filename)
        self.bc = self.bc_simulation.config

    def instantiate_gids(gids) :
        """ Instantiate a list of GIDs

        Parameters
        ----------
        gids : list of GIDs. Must be a list; even in case of instantiation of \
         a single GID
        """
        self.templates = []
        for gid in gids :
            ''' Fetch the template for this GID '''
            template_name_of_gid = _fetch_template_name(gid)
            self.templates.append(template_name_of_gid)


def _fetch_template_name(gid) :
    ncs_file = self.bc.entry_map['Default'].CONTENTS.nrnpath + '/start.ncs'
    # ncs_file = open(bg_dict['Run']['Default']['nrnPath']+'/start.ncs')
    for line in ncs_file.readlines() :
        stripped_line = line.strip()

        try :
            if( int(stripped_line.split(' ')[0][1:]) == cell_gid) :
                template_name = line.split()[4]
                print stripped_line
        except :
            pass
    return stripped_line
    
if __name__ == '__main__' :
    blue_config_filename = '/bgscratch/bbp/release/23.07.12/simulations/\
    SomatosensoryCxS1-v4.lowerCellDensity.r151/Silberberg/knockout/control/\
    BlueConfig'
    bc_sim = SSim(blue_config_filename)
    bc_sim.Instantiate([118583])

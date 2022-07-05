#!/usr/bin/env python
"""Create example cell info_dict"""


import pickle

import bglibpy


def main():
    """Main"""
    import os
    # prev_cwd = os.getcwd()
    os.chdir("../../examples/sim_twocell_synapseid")
    ssim_bglibpy = bglibpy.SSim("BlueConfig", record_dt=0.1)
    ssim_bglibpy.instantiate_gids(
        [1],
        synapse_detail=2)

    cell_info_dict = ssim_bglibpy.cells[1].info_dict
    print('Content:', cell_info_dict)

    cell_info_dict_filename = 'cell1_info_dict.pickle'
    print('Writing file %s ...' % cell_info_dict_filename, end=' ')
    with open(cell_info_dict_filename, 'w') as cell_info_dict_file:
        pickle.dump(cell_info_dict, cell_info_dict_file)
    print('Done')


if __name__ == '__main__':
    main()

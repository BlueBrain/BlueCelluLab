#!/usr/bin/env python

"""
Generate a few lists of GIDs associated with some M-type. These are later used \
to validate ssim.get_gids_of_mtypes
"""

import pickle
import re
import sys

import bglibpy

BLUE_CONFIG = "/bgscratch/bbp/l5/projects/proj1/2013.01.14/simulations/SomatosensoryCxS1-v4.lowerCellDensity.r151/Silberberg/Control_Mg0p5/BlueConfig"

sim = bglibpy.ssim.SSim(BLUE_CONFIG)
bc = sim.bc
path_to_ncs = bc.Run.nrnPath
print 'path_to_ncs: %s' % path_to_ncs
ncs_file_name = path_to_ncs + '/start.ncs'
ncs = open(ncs_file_name)

l23_btc_gids = []
l56_gids = []
l23_several_gids = []


counter = 0
for line in ncs.readlines():
    # print 'line -> %s' % line
    if not line.strip().startswith('a'):
        continue
    counter = counter + 1
    gid = line.strip().split(' ')[0][1:]
    mtype = -1
    match = re.search(r' [a-zA-Z0-9]*_(L[0-9]+_[a-zA-Z0-9-]+_L[14])_', line)
    if match:
        mtype = match.groups()[0]
        # print 'gid=%s, mtype=%s' % (gid,mtype)
    else:
        match = re.search(r' [a-zA-Z0-9]*_(L[0-9]+_[a-zA-Z0-9-]+)_', line)
        if match:
            mtype = match.groups()[0]
            # print 'gid=%s, mtype=%s' % (gid,mtype)
        else:
            raise Exception('Could not parse the M-type at line:\n%s' % line)

    if mtype != -1:
        if mtype == 'L23_BTC':
            l23_btc_gids.append(gid)
        if mtype == 'L5_TTPC1' or mtype == 'L6_TPC_L1':
            l56_gids.append(gid)
        if mtype == 'L23_BTC' or mtype == 'L23_LBC':
            l23_several_gids.append(gid)

print 'len l23_btc=%f' % len(l23_btc_gids)
print 'len l56=%f' % len(l56_gids)
print 'len l23_several=%f' % len(l23_several_gids)
pickle.dump(
    l23_btc_gids,
    open(
        '../../examples/mtype_lists/l23_btc_gids.pkl',
        'w'))
pickle.dump(l56_gids, open('../../examples/mtype_lists/l56_gids.pkl', 'w'))
pickle.dump(
    l23_several_gids,
    open(
        '../../examples/mtype_lists/l23_several_gids.pkl',
        'w'))

print 'encountered %i lines' % counter

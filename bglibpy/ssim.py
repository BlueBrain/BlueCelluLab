#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SSim Class

@remarks Copyright (c) BBP/EPFL 2012; All rights reserved.
         Do not distribute without further notice.

"""

# pylint: disable=C0103, R0914, R0912, F0401, R0101

import collections
import os
import re


from cachetools import cachedmethod, LRUCache

import numpy

from bglibpy import bluepy
import bglibpy
from bglibpy import printv
from bglibpy import tools

from bluepy.enums import Synapse as BLPSynapse
from bluepy.impl.connectome_sonata import SonataConnectome


class SSim(object):

    """Class that can load a BGLib BlueConfig,
               and instantiate the simulation"""

    # pylint: disable=R0913

    def __init__(self, blueconfig_filename, dt=0.025, record_dt=None,
                 base_seed=None, base_noise_seed=None, rng_mode=None,
                 ignore_populationid_error=False):
        """Object dealing with BlueConfig configured Small Simulations

        Parameters
        ----------
        blueconfig_filename : string
                              Absolute filename of the Blueconfig
                              Alternatively the blueconfig object can be passed
        dt : float
             Timestep of the simulation
        record_dt : float
                    Sampling interval of the recordings
        base_seed : int
                    Base seed used for this simulation. Setting this
                    will override the value set in the BlueConfig.
                    Has to positive integer.
                    When this is not set, and no seed is set in the
                    BlueConfig, the seed will be 0.
        base_noise_seed : int
                    Base seed used for the noise stimuli in the simulation.
                    Not setting this will result in the default Neurodamus
                    behavior (i.e. seed=0)
                    Has to positive integer.
        rng_mode : str
                    String with rng mode, if not specified mode is taken from
                    BlueConfig. Possible values are Compatibility, Random123
                    and UpdatedMCell.
        ignore_populationid_error: bool
                    Flag to ignore the missing population ids of projections.
        """
        self.dt = dt
        self.record_dt = record_dt
        self.blueconfig_filename = blueconfig_filename
        self.bc_simulation = bluepy.Simulation(blueconfig_filename)
        self.bc_circuit = self.bc_simulation.circuit
        self.bc = self.bc_simulation.config

        self._caches = {
            "is_group_target": LRUCache(maxsize=1000),
            "target_has_gid": LRUCache(maxsize=1000),
            "fetch_gid_cell_info": LRUCache(maxsize=100),
        }

        if self.node_properties_available:
            self.use_mecombotsv = False
            self.mecombo_emodels = None
            self.mecombo_thresholds, \
                self.mecombo_hypamps = self.get_sonata_mecombo_emodels()
        elif 'MEComboInfoFile' in self.bc.Run:
            self.use_mecombotsv = True
            self.mecombo_emodels, \
                self.mecombo_thresholds, \
                self.mecombo_hypamps = self.get_mecombo_emodels()
        else:
            self.use_mecombotsv = False
            self.mecombo_emodels = None
            self.mecombo_thresholds = None
            self.mecombo_hypamps = None

        self.rng_settings = bglibpy.RNGSettings(
            rng_mode,
            self.bc,
            base_seed=base_seed,
            base_noise_seed=base_noise_seed)

        self.ignore_populationid_error = ignore_populationid_error
        self.connection_entries = self.bc.typed_sections('Connection')
        self.all_targets = self.bc_circuit.cells.targets

        self.gids = []
        self.cells = {}

        self.neuronconfigure_entries = \
            self.bc.typed_sections("NeuronConfigure")
        self.neuronconfigure_expressions = {}
        for entry in self.neuronconfigure_entries:
            for gid in self.bc_circuit.cells.ids(entry.Target):
                conf = entry.Configure
                self.neuronconfigure_expressions.\
                    setdefault(gid, []).append(conf)

        self.gids_instantiated = False
        self.connections = \
            collections.defaultdict(
                lambda: collections.defaultdict(
                    lambda: None))

        self.emodels_dir = self.bc.Run['METypePath']
        self.morph_dir = self.bc.Run['MorphologyPath']

        # Make sure tstop is set correctly, because it is used by the
        # TStim noise stimulus
        if 'Duration' in self.bc.Run:
            bglibpy.neuron.h.tstop = float(self.bc.Run['Duration'])

        if 'MorphologyType' in self.bc.Run:
            self.morph_extension = self.bc.Run['MorphologyType']
        else:
            # backwards compatible
            if self.morph_dir[-3:] == "/h5":
                self.morph_dir = self.morph_dir[:-3]

            # latest circuits don't have asc dir
            asc_dir = os.path.join(self.morph_dir, 'ascii')
            if os.path.exists(asc_dir):
                self.morph_dir = asc_dir

            self.morph_extension = 'asc'

        self.extracellular_calcium = \
            float(self.bc.Run['ExtracellularCalcium']) \
            if 'ExtracellularCalcium' in self.bc.Run else None
        printv('Setting extracellular calcium to: %s' %
               str(self.extracellular_calcium), 50)

        self.spike_threshold = \
            float(self.bc.Run['SpikeThreshold']) \
            if 'SpikeThreshold' in self.bc.Run else -30

        if 'SpikeLocation' in self.bc.Run:
            self.spike_location = self.bc.Run['SpikeLocation']
            if self.spike_location not in ["soma", "AIS"]:
                raise bglibpy.ConfigError(
                    "Possible options for SpikeLocation are 'soma' and 'AIS'")
        else:
            self.spike_location = "soma"

        if "MinisSingleVesicle" in self.bc.Run:
            if not hasattr(
                bglibpy.neuron.h, "minis_single_vesicle_ProbAMPANMDA_EMS"
            ):
                raise bglibpy.OldNeurodamusVersionError(
                    "Synapses don't implement minis_single_vesicle."
                    "More recent neurodamus model required."
                )
            minis_single_vesicle = int(self.bc.Run["MinisSingleVesicle"])
            printv(
                "Setting synapses minis_single_vesicle to %d"
                % minis_single_vesicle,
                50,
            )
            bglibpy.neuron.h.minis_single_vesicle_ProbAMPANMDA_EMS = (
                minis_single_vesicle
            )
            bglibpy.neuron.h.minis_single_vesicle_ProbGABAAB_EMS = (
                minis_single_vesicle
            )
            bglibpy.neuron.h.minis_single_vesicle_GluSynapse = (
                minis_single_vesicle
            )

    @property
    def node_properties_available(self):
        """Checks if the node properties can be used."""
        node_props = {
            "@dynamics:holding_current",
            "@dynamics:threshold_current",
            "model_template",
        }

        return node_props.issubset(self.bc_circuit.cells.available_properties)

    @property
    def base_seed(self):
        """Baseseed of sim"""

        return self.rng_settings.base_seed

    @property
    def base_noise_seed(self):
        """Baseseed of noise stimuli in sim"""

        return self.rng_settings.base_noise_seed

    # pylint: disable=R0913
    def instantiate_gids(self, gids, synapse_detail=None,
                         add_replay=False,
                         add_stimuli=False,
                         add_synapses=None,
                         add_minis=None,
                         add_noise_stimuli=False,
                         add_hyperpolarizing_stimuli=False,
                         add_relativelinear_stimuli=False,
                         add_pulse_stimuli=False,
                         add_projections=False,
                         intersect_pre_gids=None,
                         interconnect_cells=True,
                         pre_spike_trains=None,
                         projection=None,
                         projections=None):
        """ Instantiate a list of GIDs

        Parameters
        ----------
        gids : list of integers
               List of GIDs. Must be a list,
               even in case of instantiation of a single GID.
        synapse_detail : {0 , 1, 2}
                         Level of detail. If chosen, all settings are taken
                         from the "large" cortical simulation.
                         Possible values:

                         * 0 No synapses

                         * 1 Add synapse of the correct type at the
                            simulated locations with all settings
                            as in the "large" simulation

                         * 2 As 1 but with minis

        add_replay : Boolean
                     Add presynaptic spiketrains from the large simulation
                     throws an exception if this is set when synapse_detail < 1
                     If pre_spike_trains is combined with this option the
                     spiketrains will be merged
        add_stimuli : Boolean
                      Add the same stimuli as in the large simulation
        add_synapses : Boolean
                       Add the touch-detected synapses, as described by the
                       circuit to the cell
                       (This option only influence the 'creation' of synapses,
                       it doesn't add any connections)
                       Default value is False
        add_minis : Boolean
                    Add synaptic minis to the synapses
                    (this requires add_synapses=True)
                    Default value is False
        add_noise_stimuli : Boolean
                            Process the 'noise' stimuli blocks of the
                            BlueConfig,
                            Setting add_stimuli=True,
                            will automatically set this option to True.
        add_hyperpolarizing_stimuli : Boolean
                                      Process the 'hyperpolarizing' stimuli
                                      blocks of the BlueConfig.
                                      Setting add_stimuli=True,
                                      will automatically set this option to
                                      True.
        add_relativelinear_stimuli : Boolean
                                      Process the 'relativelinear' stimuli
                                      blocks of the BlueConfig.
                                      Setting add_stimuli=True,
                                      will automatically set this option to
                                      True.
        add_pulse_stimuli : Boolean
                                      Process the 'pulse' stimuli
                                      blocks of the BlueConfig.
                                      Setting add_stimuli=True,
                                      will automatically set this option to
                                      True.
        add_projections: Boolean
                          If set True, adds all of the projection blocks of the
                          BlueConfig. This option assumes no additional
                           projection is passed using the `projections` option.
        intersect_pre_gids : list of gids
                             Only add synapses to the cells if their
                             presynaptic gid is in this list
        interconnect_cells : Boolean
                             When multiple gids are instantiated,
                             interconnect the cells with real (non-replay)
                             synapses. When this option is combined with
                             add_replay, replay spiketrains will only be added
                             for those presynaptic cells that are not in the
                             network that's instantiated.
                             This option requires add_synapses=True
        pre_spike_trains : dict
                           A dictionary with keys the presynaptic gids, and
                           values the list of spike timings of the
                           presynaptic cells with the given gids.
                           If this option is used in combination with
                           add_replay=True, the spike trains for the same
                           gids will be automatically merged
        projection: string
                    Name of the projection where all information about synapses
                    come from. Mixing different projections is not possible
                    at the moment.
                    Beware, this option might disappear in the future if
                    BluePy unifies the API to get the synapse information for
                    a certain gid.
        """

        if synapse_detail is not None:
            printv(
                'WARNING: SSim: synapse_detail is deprecated and will '
                'removed from future release of BGLibPy', 2)
            if synapse_detail > 0:
                if add_minis is False:
                    raise Exception('SSim: synapse_detail >= 1 cannot be used'
                                    ' with add_minis == False')
                add_synapses = True
            if synapse_detail > 1:
                if add_minis is False:
                    raise Exception('SSim: synapse_detail >= 2 cannot be used'
                                    ' with add_minis == False')
                add_minis = True

        if add_minis is None:
            add_minis = False

        if self.gids_instantiated:
            raise Exception("SSim: instantiate_gids() called twice on the \
                    same SSim, this is not supported yet")
        else:
            self.gids_instantiated = True

        if pre_spike_trains or add_replay:
            if add_synapses is not None and add_synapses is False:
                raise Exception("SSim: you need to set add_synapses to True "
                                "if you want to specify use add_replay or "
                                "pre_spike_trains")
            add_synapses = True
        else:
            if add_synapses is None:
                add_synapses = False

        if add_projections:
            if projections is not None:
                raise ValueError('SSim: projections and add_projections '
                                 'can not be used at the same time')
            projections = [proj.name
                           for proj in self.bc.typed_sections('Projection')]

        self._add_cells(gids)
        if add_stimuli:
            add_noise_stimuli = True
            add_hyperpolarizing_stimuli = True
            add_relativelinear_stimuli = True
            add_pulse_stimuli = True

        if add_noise_stimuli or \
                add_hyperpolarizing_stimuli or \
                add_pulse_stimuli or \
                add_relativelinear_stimuli:
            self._add_stimuli(
                add_noise_stimuli=add_noise_stimuli,
                add_hyperpolarizing_stimuli=add_hyperpolarizing_stimuli,
                add_relativelinear_stimuli=add_relativelinear_stimuli,
                add_pulse_stimuli=add_pulse_stimuli)
        if add_synapses:
            self._add_synapses(
                intersect_pre_gids=intersect_pre_gids,
                add_minis=add_minis,
                add_projections=add_projections,
                projection=projection,
                projections=projections)
        if add_replay or interconnect_cells or pre_spike_trains:
            if add_replay and not add_synapses:
                raise Exception("SSim: add_replay option can not be used if "
                                "add_synapses is False")
            self._add_connections(add_replay=add_replay,
                                  interconnect_cells=interconnect_cells,
                                  user_pre_spike_trains=pre_spike_trains)

    # pylint: enable=R0913

    def _add_stimuli(self, add_noise_stimuli=False,
                     add_hyperpolarizing_stimuli=False,
                     add_relativelinear_stimuli=False,
                     add_pulse_stimuli=False):
        """Instantiate all the stimuli"""
        for gid in self.gids:
            # Also add the injections / stimulations as in the cortical model
            self._add_stimuli_gid(
                gid,
                add_noise_stimuli=add_noise_stimuli,
                add_hyperpolarizing_stimuli=add_hyperpolarizing_stimuli,
                add_relativelinear_stimuli=add_relativelinear_stimuli,
                add_pulse_stimuli=add_pulse_stimuli)
            printv("Added stimuli for gid %d" % gid, 2)

    def _add_synapses(
            self, intersect_pre_gids=None, add_minis=None,
            add_projections=None, projection=None, projections=None):
        """Instantiate all the synapses"""

        if add_projections or (projection is None and projections is None):
            add_local_synapses = True
        else:
            add_local_synapses = False

        if projection is None and projections is None:
            add_projection_synapses = False
        else:
            add_projection_synapses = True

        for gid in self.gids:
            if add_local_synapses:
                self._add_gid_synapses(
                    gid, intersect_pre_gids=intersect_pre_gids,
                    add_minis=add_minis)
            if add_projection_synapses:
                self._add_gid_synapses(
                    gid, intersect_pre_gids=intersect_pre_gids,
                    add_minis=add_minis,
                    projection=projection,
                    projections=projections)

    def _add_gid_synapses(self, gid, intersect_pre_gids=None, add_minis=None,
                          projection=None, projections=None):
        syn_descriptions_popids = self.get_syn_descriptions_dict(
            gid,
            projection=projection, projections=projections)

        if intersect_pre_gids is not None:
            syn_descriptions_popids = {syn_id: syn_description_popids
                                       for syn_id, syn_description_popids in
                                       syn_descriptions_popids.items()
                                       if syn_description_popids[0][0] in
                                       intersect_pre_gids}

        # Check if there are any presynaptic cells, otherwise skip adding
        # synapses
        if syn_descriptions_popids is None:
            printv(
                "Warning: No presynaptic cells found for gid %d, "
                "no synapses added" % gid, 2)
        else:
            for syn_id, (syn_description,
                         popids) in syn_descriptions_popids.items():
                self._instantiate_synapse(gid, syn_id, syn_description,
                                          add_minis=add_minis, popids=popids)
            printv("Added %d synapses for gid %d" %
                   (len(syn_descriptions_popids), gid), 2)
            if add_minis:
                printv("Added minis for gid %d" % gid, 2)

    @tools.deprecated("get_syn_descriptions_dict")
    def get_syn_descriptions(self, gid, projection=None, projections=None):
        """Get synapse description arrays from bluepy"""

        syn_descriptions = [
            v for _,
            v in sorted(
                self.get_syn_descriptions_dict(
                    gid,
                    projection=projection,
                    projections=projections).items())]

        return syn_descriptions

    def get_syn_descriptions_dict(
            self, gid, projection=None, projections=None):
        """Get synapse descriptions dict from bluepy, Keys are synapse ids"""
        syn_descriptions_dict = {}

        if projection is not None:
            if projections is None:
                projections = [projection]
            else:
                raise ValueError(
                    'Cant combine projection and projections arguemnt')

        post_segment_id = BLPSynapse.POST_SEGMENT_ID
        post_segment_offset = BLPSynapse.POST_SEGMENT_OFFSET

        all_properties = [
            BLPSynapse.PRE_GID,
            BLPSynapse.AXONAL_DELAY,
            BLPSynapse.POST_SECTION_ID,
            post_segment_id,
            post_segment_offset,
            BLPSynapse.G_SYNX,
            BLPSynapse.U_SYN,
            BLPSynapse.D_SYN,
            BLPSynapse.F_SYN,
            BLPSynapse.DTC,
            BLPSynapse.TYPE,
            BLPSynapse.NRRP,
            BLPSynapse.U_HILL_COEFFICIENT,
            BLPSynapse.CONDUCTANCE_RATIO]

        if projections is None:
            connectomes = {'': self.bc_circuit.connectome}
        else:
            connectomes = {
                this_projection: self.bc_circuit.projection(this_projection)
                for this_projection in projections}

        all_synapse_sets = {}
        for proj_name, connectome in connectomes.items():
            using_sonata = False

            nrrp_defined = True

            # list() to make a copy
            connectome_properties = list(all_properties)

            # older circuit don't have these properties
            for test_property in [BLPSynapse.U_HILL_COEFFICIENT,
                                  BLPSynapse.CONDUCTANCE_RATIO,
                                  BLPSynapse.NRRP]:
                if test_property not in connectome.available_properties:
                    connectome_properties.remove(test_property)
                    if test_property == BLPSynapse.NRRP:
                        nrrp_defined = False
                    printv(
                        'WARNING: %s not found, disabling' %
                        test_property, 50)

            if isinstance(connectome._impl, SonataConnectome):
                # load 'afferent_section_pos' instead of '_POST_DISTANCE'
                if 'afferent_section_pos' in connectome.available_properties:
                    connectome_properties[
                        connectome_properties.index(post_segment_offset)
                    ] = 'afferent_section_pos'

                synapses = connectome.afferent_synapses(
                    gid, properties=connectome_properties
                )

                # replace '_POST_SEGMENT_ID' with -1 (as indicator for
                # synlocation_to_segx)
                if 'afferent_section_pos' in connectome.available_properties:
                    synapses[post_segment_id] = -1

                using_sonata = True
                printv('Using sonata style synapse file, not nrn.h5', 50)
            else:
                synapses = connectome.afferent_synapses(
                    gid, properties=connectome_properties
                )

            # io/synapse_reader.py:_patch_delay_fp_inaccuracies from
            # py-neurodamus
            dt = bglibpy.neuron.h.dt
            synapses[BLPSynapse.AXONAL_DELAY] = (
                synapses[BLPSynapse.AXONAL_DELAY] / dt + 1e-5
            ).astype('i4') * dt

            all_synapse_sets[proj_name] = (synapses, connectome_properties)

        if not all_synapse_sets:
            printv('No synapses found', 5)
        else:
            printv(
                'Adding a total of %d synapse sets' %
                len(all_synapse_sets), 5)

            for proj_name, (synapse_set,
                            connectome_properties) in all_synapse_sets.items():
                if proj_name in [
                    proj.name for proj in self.bc.typed_sections("Projection")
                ]:
                    if "PopulationID" in self.bc["Projection_" + proj_name]:
                        source_popid = int(
                            self.bc["Projection_" + proj_name]["PopulationID"]
                        )
                    else:
                        if self.ignore_populationid_error:
                            source_popid = 0
                        else:
                            raise bglibpy.PopulationIDMissingError(
                                "PopulationID is missing from projection,"
                                " block this will lead to wrong rng seeding."
                                " If you anyway want to overwrite this,"
                                " pass ignore_populationid_error=True param"
                                " to SSim constructor."
                            )
                else:
                    source_popid = 0
                # ATM hard coded in neurodamus
                target_popid = 0
                popids = (source_popid, target_popid)

                printv(
                    'Adding a total of %d synapses for set %s' %
                    (synapse_set.shape[0], proj_name), 5)

                for syn_id, (index, synapse) in enumerate(
                        synapse_set.iterrows()):
                    if not using_sonata:
                        syn_gid, syn_id = index
                        if syn_gid != gid:
                            raise Exception(
                                "BGLibPy SSim: synapse gid doesnt match with "
                                "cell gid !")
                    syn_id_proj = (proj_name, syn_id)
                    if syn_id_proj in syn_descriptions_dict:
                        raise Exception(
                            "BGLibPy SSim: trying to add "
                            "synapse id %s twice !" %
                            syn_id_proj)
                    if nrrp_defined:
                        old_syn_description = \
                            synapse[connectome_properties].values[:11]
                        nrrp = synapse[BLPSynapse.NRRP]

                        # if the following 2 variables don't exist in the
                        # circuit we put None, to detect that in the Synapse
                        # code
                        u_hill_coefficient = \
                            synapse[BLPSynapse.U_HILL_COEFFICIENT] \
                            if BLPSynapse.U_HILL_COEFFICIENT in synapse \
                            else None
                        conductance_ratio = \
                            synapse[BLPSynapse.CONDUCTANCE_RATIO] \
                            if BLPSynapse.CONDUCTANCE_RATIO in synapse \
                            else None
                        ext_syn_description = numpy.array(
                            [-1, -1, -1,
                             nrrp, u_hill_coefficient, conductance_ratio])
                        # 14 - 16 are dummy values, 17 is Nrrp
                        syn_description = numpy.append(
                            old_syn_description,
                            ext_syn_description)
                    else:
                        # old behavior
                        syn_description = synapse[connectome_properties].values

                    syn_description = numpy.insert(
                        syn_description, [5, 5, 5], [-1, -1, -1])

                    syn_descriptions_dict[syn_id_proj] = \
                        syn_description, popids

        return syn_descriptions_dict

    @staticmethod
    def merge_pre_spike_trains(*train_dicts):
        """Merge presynaptic spike train dicts"""

        ret_dict = None

        for train_dict in train_dicts:
            if train_dict is not None:
                if ret_dict is None:
                    ret_dict = train_dict.copy()
                    continue
                else:
                    # Not super efficient, but out.dats to tend not to be
                    # very large
                    for pre_gid, train in train_dict.items():
                        ret_dict.setdefault(pre_gid, []).extend(train)

        if ret_dict is not None:
            for pre_gid, train in ret_dict.items():
                if train is not None:
                    ret_dict[pre_gid] = numpy.array(sorted(train))

        return ret_dict

    # pylint: disable=R0913
    def _add_connections(
            self,
            add_replay=None,
            interconnect_cells=None,
            outdat_path=None,
            source=None,
            dest=None,
            user_pre_spike_trains=None):
        """Instantiate the (replay and real) connections in the network"""
        if add_replay:
            if outdat_path is None:
                outdat_path = os.path.join(
                    self.bc.Run['OutputRoot'],
                    'out.dat')
            pre_spike_trains = _parse_outdat2(outdat_path)
        else:
            pre_spike_trains = None

        pre_spike_trains = self.merge_pre_spike_trains(
            pre_spike_trains,
            user_pre_spike_trains)
        for post_gid in self.gids:
            if dest and post_gid not in dest:
                continue
            for syn_id in self.cells[post_gid].synapses:
                synapse = self.cells[post_gid].synapses[syn_id]
                syn_description = synapse.syn_description
                connection_parameters = synapse.connection_parameters
                pre_gid = syn_description[0]
                if source and pre_gid not in source:
                    continue
                real_synapse_connection = pre_gid in self.gids \
                    and interconnect_cells

                connection = None
                if real_synapse_connection:
                    connection = bglibpy.Connection(
                        self.cells[post_gid].synapses[syn_id],
                        pre_spiketrain=None,
                        pre_cell=self.cells[pre_gid],
                        stim_dt=self.dt,
                        spike_threshold=self.spike_threshold,
                        spike_location=self.spike_location)
                    printv("Added real connection between pre_gid %d and \
                            post_gid %d, syn_id %s" % (pre_gid,
                                                       post_gid,
                                                       str(syn_id)), 5)
                elif pre_spike_trains is not None:
                    pre_spiketrain = pre_spike_trains.setdefault(pre_gid, None)
                    connection = bglibpy.Connection(
                        self.cells[post_gid].synapses[syn_id],
                        pre_spiketrain=pre_spiketrain,
                        pre_cell=None,
                        stim_dt=self.dt,
                        spike_threshold=self.spike_threshold,
                        spike_location=self.spike_location)
                    printv(
                        "Added replay connection from pre_gid %d to "
                        "post_gid %d, syn_id %s" %
                        (pre_gid, post_gid, syn_id), 5)

                if connection is not None:
                    self.cells[post_gid].connections[syn_id] = connection
                    if "DelayWeights" in connection_parameters:
                        for delay, weight_scale in \
                                connection_parameters['DelayWeights']:
                            self.cells[post_gid].add_replay_delayed_weight(
                                syn_id, delay,
                                weight_scale * connection.weight)

            if len(self.cells[post_gid].connections) > 0:
                printv("Added synaptic connections for target post_gid %d" %
                       post_gid, 2)

    def _add_cells(self, gids):
        """Instantiate cells from a gid list"""
        self.gids = gids
        self.cells = {}

        for gid in self.gids:
            printv(
                'Adding gid %d from emodel %s and morph %s' %
                (gid,
                 self.fetch_emodel_name(gid),
                 self.fetch_morph_name(gid)),
                1)

            self.cells[gid] = bglibpy.Cell(**self.fetch_cell_kwargs(gid))

            if gid in self.neuronconfigure_expressions:
                for expression in self.neuronconfigure_expressions[gid]:
                    self.cells[gid].execute_neuronconfigure(expression)

    def _instantiate_synapse(self, gid, syn_id, syn_description,
                             add_minis=None, popids=None):
        """Instantiate one synapse for a given gid, syn_id and
        syn_description"""

        syn_type = syn_description[13]

        connection_parameters = self._evaluate_connection_parameters(
            syn_description[0],
            gid,
            syn_type)

        if connection_parameters["add_synapse"]:
            self.add_single_synapse(gid, syn_id, syn_description,
                                    connection_parameters, popids=popids)
            if add_minis:
                mini_frequencies = self.fetch_mini_frequencies(gid)
                printv(
                    'Adding minis for synapse %s: '
                    'syn_description=%s, connection=%s, frequency=%s' %
                    (str(syn_id),
                     str(syn_description),
                        str(connection_parameters),
                        str(mini_frequencies)),
                    50)

                self.add_replay_minis(
                    gid,
                    syn_id,
                    syn_description,
                    connection_parameters,
                    popids=popids,
                    mini_frequencies=mini_frequencies)

    def _add_stimuli_gid(self, gid,
                         add_noise_stimuli=False,
                         add_hyperpolarizing_stimuli=False,
                         add_relativelinear_stimuli=False,
                         add_pulse_stimuli=False):
        """ Adds indeitical stimuli to the simulated cell as in the 'large'
            model

        Parameters:
        -----------
        gid: gid of the simulated cell
        """
        # check in which StimulusInjects the gid is a target
        # Every noise stimulus gets a new seed
        noisestim_count = 0

        for entry in self.bc.values():
            if entry.section_type == 'StimulusInject':
                destination = entry.Target
                # retrieve the stimulus to apply
                stimulus_name = entry.Stimulus
                # bluepy magic to add underscore Stimulus underscore
                # stimulus_name
                stimulus = self.bc['Stimulus_%s' % stimulus_name]
                gids_of_target = self.bc_circuit.cells.ids(destination)
                if gid in gids_of_target:
                    if stimulus.Pattern == 'Noise':
                        if add_noise_stimuli:
                            self._add_replay_noise(
                                gid,
                                stimulus,
                                noisestim_count=noisestim_count)
                    elif stimulus.Pattern == 'Hyperpolarizing':
                        if add_hyperpolarizing_stimuli:
                            self._add_replay_hypamp_injection(
                                gid,
                                stimulus)
                    elif stimulus.Pattern == 'Pulse':
                        if add_pulse_stimuli:
                            self._add_pulse(gid, stimulus)
                    elif stimulus.Pattern == 'RelativeLinear':
                        if add_relativelinear_stimuli:
                            self._add_relativelinear(gid, stimulus)
                    elif stimulus.Pattern == 'SynapseReplay':
                        printv("Found stimulus with pattern %s, ignoring" %
                               stimulus['Pattern'], 1)
                    else:
                        raise Exception("Found stimulus with pattern %s, "
                                        "not supported" %
                                        stimulus.Pattern)
                if stimulus.Pattern == 'Noise':
                    noisestim_count += 1

    def _add_replay_hypamp_injection(self, gid, stimulus):
        """Add injections from the replay"""
        self.cells[gid].add_replay_hypamp(stimulus)

    def _add_relativelinear(self, gid, stimulus):
        """Add relative linear injections from the replay"""
        self.cells[gid].add_replay_relativelinear(stimulus)

    def _add_pulse(self, gid, stimulus):
        """Add injections from the replay"""
        self.cells[gid].add_pulse(stimulus)

    def _add_replay_noise(
            self,
            gid,
            stimulus,
            noisestim_count=None):
        """Add noise injection from the replay"""
        self.cells[gid].add_replay_noise(
            stimulus,
            noisestim_count=noisestim_count)

    def add_replay_minis(self, gid, syn_id, syn_description,
                         syn_parameters, popids=None, mini_frequencies=None):
        """Add minis from the replay"""
        self.cells[gid].add_replay_minis(
            syn_id,
            syn_description,
            syn_parameters,
            popids=popids,
            mini_frequencies=mini_frequencies)

    def add_single_synapse(self, gid, syn_id,
                           syn_description, connection_modifiers, popids=None):
        """Add a replay synapse on the cell

        Parameters
        ----------
        gid : int
              GID of the cell
        syn_id: int
              Synapse ID of the synapse
        syn_description: dict
              Description of the synapse
        connection_modifiers: dict
              Connection modifiers for the synapse
        """
        return self.cells[gid].add_replay_synapse(
            syn_id, syn_description, connection_modifiers, popids=popids,
            extracellular_calcium=self.extracellular_calcium)

    def check_connection_contents(self, contents):
        """Check the contents of a connection block,
           to see if we support all the fields"""

        allowed_keys = set(['Weight', 'SynapseID', 'SpontMinis',
                            'SynapseConfigure', 'Source',
                            'Destination', 'Delay', 'CreateMode'])
        for key in contents.keys():
            if key not in allowed_keys:
                raise Exception(
                    "Key %s in Connection blocks not supported by BGLibPy"
                    % key)

    def is_cell_target(self, target, gid):
        """Check if target is cell"""

        return target == 'a%d' % gid

    @cachedmethod(lambda self: self._caches["is_group_target"])
    def is_group_target(self, target):
        """Check if target is group of cells"""

        return target in self.bc_circuit.cells.targets

    @cachedmethod(lambda self: self._caches["target_has_gid"])
    def target_has_gid(self, target, gid):

        gid_found = (gid in self.bc_circuit.cells.ids(target))
        return gid_found

    def _evaluate_connection_parameters(self, pre_gid, post_gid, syn_type):
        """ Apply connection blocks in order for pre_gid, post_gid to
            determine a final connection override for this pair
            (pre_gid, post_gid)

        Parameters:
        -----------
        gid : int
              gid of the post-synaptic cell

        """
        parameters = {}
        parameters['add_synapse'] = True

        gid_pttrn = re.compile("^a[0-9]+")

        for entry in self.connection_entries:
            entry_name = entry.name
            self.check_connection_contents(entry)
            src = entry['Source']
            dest = entry['Destination']

            for target in (src, dest):
                if not (
                    self.is_group_target(target) or gid_pttrn.match(target)
                ):
                    raise bglibpy.TargetDoesNotExist(
                        "%s target does not exist" % target
                    )

            src_matches = self.is_cell_target(src, pre_gid) or \
                (self.target_has_gid(src, pre_gid))
            dest_matches = self.is_cell_target(dest, post_gid) or \
                (self.target_has_gid(dest, post_gid))

            if src_matches and dest_matches:
                # whatever specified in this block, is applied to gid
                apply_parameters = True
                keys = set(entry.keys())

                if 'SynapseID' in keys:
                    if int(entry['SynapseID']) != syn_type:
                        apply_parameters = False

                if 'Delay' in keys:
                    parameters.setdefault('DelayWeights', []).append((
                        float(entry['Delay']),
                        float(entry['Weight'])))
                    apply_parameters = False

                if apply_parameters:
                    if 'CreateMode' in keys:
                        if entry['CreateMode'] == 'NoCreate':
                            parameters['add_synapse'] = False
                        else:
                            raise Exception('Connection %s: Unknown '
                                            'CreateMode option %s'
                                            % (entry_name,
                                               entry['CreateMode']))
                    if 'Weight' in keys:
                        parameters['Weight'] = float(entry['Weight'])
                    if 'SpontMinis' in keys:
                        parameters['SpontMinis'] = float(
                            entry['SpontMinis'])
                    if 'SynapseConfigure' in keys:
                        conf = entry['SynapseConfigure']
                        # collect list of applicable configure blocks to be
                        # applied with a "hoc exec" statement
                        parameters.setdefault(
                            'SynapseConfigure', []).append(conf)

        return parameters

    def initialize_synapses(self):
        """ Resets the state of all synapses of all cells to initial values """
        for cell in self.cells.itervalues():
            cell.initialize_synapses()

    def run(self, t_stop=None, v_init=None, celsius=None, dt=None,
            forward_skip=True, forward_skip_value=None,
            cvode=False, show_progress=False):
        """Simulate the SSim

        Parameters
        ----------
        t_stop : int
                 This function will run the simulation until t_stop
        v_init : float
                 Voltage initial value when the simulation starts
        celsius : float
                  Temperature at which the simulation runs
        dt : float
             Timestep (delta-t) for the simulation
        forward_skip : boolean
                       Enable/disable ForwardSkip (default=True, when
                       forward_skip_value is None, forward skip will only be
                       enabled if BlueConfig has a ForwardSkip value)
        forward_skip_value : float
                       Overwrite the ForwardSkip value in the BlueConfig. If
                       this is set to None, the value in the BlueConfig is
                       used.
        cvode : boolean
                Force the simulation to run in variable timestep. Not possible
                when there are stochastic channels in the neuron model. When
                enabled results from a large network simulation will not be
                exactly reproduced.
        show_progress: boolean
                       Show a progress bar during simulations. When
                       enabled results from a large network simulation
                       will not be exactly reproduced.
        """
        if t_stop is None:
            t_stop = float(self.bc.Run['Duration'])
        if dt is None:
            dt = float(self.bc.Run['Dt'])
        if forward_skip_value is None:
            if 'ForwardSkip' in self.bc.Run:
                forward_skip_value = float(
                    self.bc.Run['ForwardSkip'])
        if celsius is None:
            if 'Celsius' in self.bc.Run:
                celsius = float(self.bc.Run['Celsius'])
            else:
                celsius = 34  # default
        if v_init is None:
            if 'V_Init' in self.bc.Run:
                v_init = float(self.bc.Run['V_Init'])
            else:
                v_init = -65  # default

        sim = bglibpy.Simulation()
        for gid in self.gids:
            sim.add_cell(self.cells[gid])

        if show_progress:
            printv("Warning: show_progress enabled, this will very likely"
                   "break the exact reproducibility of large network"
                   "simulations", 2)

        sim.run(
            t_stop,
            cvode=cvode,
            dt=dt,
            celsius=celsius,
            v_init=v_init,
            forward_skip=forward_skip,
            forward_skip_value=forward_skip_value,
            show_progress=show_progress)

    def get_mainsim_voltage_trace(
        self, gid=None, t_start=None, t_stop=None, t_step=None
    ):
        """Get the voltage trace from a cell from the main simulation

        Parameters:
        -----------
        gid: GID of interest
        t_start, t_stop: time range of interest,
        report time range is used by default.
        t_step: time step (should be a multiple of report time step T;
        equals T by default)

        Returns:
            One dimentional numpy.ndarray to represent the voltages.
        """

        voltage = (
            self.bc_simulation.report("soma")
            .get_gid(gid, t_start=t_start, t_end=t_stop, t_step=t_step)
            .values
        )
        return voltage

    def get_mainsim_time_trace(self):
        """Get the time trace from the main simulation"""

        report = self.bc_simulation.report('soma')
        time = report.get_gid(report.gids[0]).index
        return time

    @tools.deprecated("get_voltage_trace")
    def get_voltage_traces(self):
        """Get the voltage traces from all the cells as a dictionary
           based on gid"""
        vm = {}
        for gid in self.gids:
            vm[gid] = self.cells[gid].get_soma_voltage()
        return vm

    @tools.deprecated("get_time_trace")
    def get_time(self):
        """Get the time vector for the recordings"""
        return self.cells[self.gids[0]].get_time()

    def get_time_trace(self):
        """Get the time vector for the recordings, negative times removed"""

        time = self.cells[self.gids[0]].get_time()
        pos_time = time[numpy.where(time >= 0.0)]
        return pos_time

    def get_voltage_trace(self, gid):
        """Get the voltage vector for the gid, negative times removed"""

        time = self.get_time_trace()
        voltage = self.cells[gid].get_soma_voltage()
        pos_voltage = voltage[numpy.where(time >= 0.0)]
        return pos_voltage

    def delete(self):
        """Delete ssim"""

        # This code might look weird, but it's to make sure cells are properly
        # delete by python
        if hasattr(self, 'cells'):
            for _, cell in self.cells.items():
                cell.delete()
            gids = list(self.cells.keys())
            for gid in gids:
                del self.cells[gid]

    def __del__(self):
        """Destructor"""
        self.delete()

    # Auxialliary methods ###

    def get_mecombo_emodels(self):
        """Create a dict matching me_combo names to template_names"""

        mecombo_filename = self.bc.Run['MEComboInfoFile']

        with open(mecombo_filename) as mecombo_file:
            mecombo_content = mecombo_file.read()

        mecombo_emodels = {}
        mecombo_thresholds = {}
        mecombo_hypamps = {}

        for line in mecombo_content.split('\n')[1:-1]:
            mecombo_info = line.split('\t')
            emodel = mecombo_info[4]
            me_combo = mecombo_info[5]
            try:
                threshold = float(mecombo_info[6])
            except (ValueError, IndexError):
                threshold = 0.0
            try:
                hypamp = float(mecombo_info[7])
            except (ValueError, IndexError):
                hypamp = 0.0
            mecombo_emodels[me_combo] = emodel
            mecombo_thresholds[me_combo] = threshold
            mecombo_hypamps[me_combo] = hypamp

        return mecombo_emodels, mecombo_thresholds, mecombo_hypamps

    def get_sonata_mecombo_emodels(self):
        """Extracts the holding and threshold currents.

        Returns:
            tuple of dicts containing the threshold and holding
            currents for every cell.
        """
        cell_props = self.bc_circuit.cells.get(
            None,
            properties=[
                "@dynamics:holding_current",
                "@dynamics:threshold_current",
                "model_template",
            ],
        )

        cell_props["model_template"] = (
            cell_props["model_template"].str.split("hoc:").str[1]
        )
        cell_props = cell_props.set_index("model_template").to_dict()

        mecombo_thresholds = cell_props["@dynamics:threshold_current"]
        mecombo_holding_currs = cell_props["@dynamics:holding_current"]

        return mecombo_thresholds, mecombo_holding_currs

    @cachedmethod(lambda self: self._caches["fetch_gid_cell_info"])
    def fetch_gid_cell_info(self, gid):
        """Fetch bluepy cell info of a gid"""
        if gid in self.bc_circuit.cells.ids():
            cell_info = self.bc_circuit.cells.get(gid)
        else:
            raise Exception("Gid %d not found in circuit" % gid)

        return cell_info

    def fetch_mecombo_name(self, gid):
        """Fetch mecombo name for a certain gid"""

        cell_info = self.fetch_gid_cell_info(gid)
        if self.node_properties_available:
            me_combo = str(cell_info['model_template'])
            me_combo = me_combo.split('hoc:')[1]
        else:
            me_combo = str(cell_info['me_combo'])
        return me_combo

    def fetch_emodel_name(self, gid):
        """Get the emodel path of a gid"""

        me_combo = self.fetch_mecombo_name(gid)

        if self.use_mecombotsv:
            emodel_name = self.mecombo_emodels[me_combo]
        else:
            emodel_name = me_combo

        return emodel_name

    def fetch_morph_name(self, gid):
        """Get the morph name of a gid"""

        cell_info = self.fetch_gid_cell_info(gid)

        morph_name = str(cell_info['morphology'])

        return morph_name

    def fetch_mini_frequencies(self, gid):
        """Get inhibitory frequency of gid"""

        cell_info = self.fetch_gid_cell_info(gid)

        inh_mini_frequency = cell_info['inh_mini_frequency'] \
            if 'inh_mini_frequency' in cell_info else None

        exc_mini_frequency = cell_info['exc_mini_frequency'] \
            if 'exc_mini_frequency' in cell_info else None

        return exc_mini_frequency, inh_mini_frequency

    def fetch_cell_kwargs(self, gid):
        """Get the kwargs to instantiate a gid's Cell object"""
        emodel_path = os.path.join(
            self.emodels_dir,
            self.fetch_emodel_name(gid) +
            '.hoc')

        morph_filename = '%s.%s' % \
            (self.fetch_morph_name(gid), self.morph_extension)

        if self.use_mecombotsv or self.node_properties_available:
            me_combo = self.fetch_mecombo_name(gid)
            extra_values = {
                'threshold_current': self.mecombo_thresholds[me_combo],
                'holding_current': self.mecombo_hypamps[me_combo]
            }
            cell_kwargs = {
                'template_filename': emodel_path,
                'morphology_name': morph_filename,
                'gid': gid,
                'record_dt': self.record_dt,
                'morph_dir': self.morph_dir,
                'template_format': 'v6',
                'extra_values': extra_values,
                'rng_settings': self.rng_settings
            }
        else:
            cell_kwargs = {
                'template_filename': emodel_path,
                'morphology_name': self.morph_dir,
                'gid': gid,
                'record_dt': self.record_dt,
                'rng_settings': self.rng_settings
            }

        return cell_kwargs

    def get_gids_of_targets(self, targets=None):

        gids = []
        for target in targets:
            gids.extend(self.bc_circuit.cells.ids(target))

        return gids

    def get_gids_of_mtypes(self, mtypes=None):
        """
        Helper function that, provided a BlueConfig, returns all the GIDs \
        associated with a specified M-type. (For instance, when you only want \
        to insert synapses of a specific pathway)


        Parameters
        ----------
        mtypes : list
            List of M-types (each as a string). Wildcards are *not* allowed, \
            the strings must represent the true M-type names. A list with \
            names can be found here: \
            bbpteam.epfl.ch/projects/spaces/display/MEETMORPH/m-types

        Returns
        -------
        gids : list
            List of all GIDs associated with one of the specified M-types

        """
        gids = []
        for mtype in mtypes:
            gids.extend(
                self.bc_circuit.cells.get({bluepy.Cell.MTYPE: mtype}).
                index.values)

        return gids


def _parse_outdat2(path):
    """Parse the replay spiketrains in a out.dat formatted file
       pointed to by path"""

    import bluepy.impl.spike_report
    spikes = bluepy.impl.spike_report.SpikeReport.load(path)

    outdat = {}

    for gid in spikes.gids:
        spike_times = spikes.get_gid(gid)
        if any(spike_times < 0):
            printv(
                'WARNING: SSim: Found negative spike times in out.dat ! '
                'Clipping them to 0', 2)
            spike_times = spike_times.clip(min=0.0)

        outdat[gid] = spike_times

    return outdat


def _parse_outdat(path, outdat_name='out.dat'):
    """Parse the replay spiketrains in out.dat"""
    full_outdat_name = os.path.join(path, outdat_name)
    return _parse_outdat2(full_outdat_name)

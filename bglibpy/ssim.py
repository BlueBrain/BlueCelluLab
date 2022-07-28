#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SSim Class

@remarks Copyright (c) BBP/EPFL 2012; All rights reserved.
         Do not distribute without further notice.

"""

# pylint: disable=C0103, R0914, R0912, F0401, R0101

import collections
import os
from typing import Any, DefaultDict, List
import warnings

import numpy as np
from bluepy.enums import Synapse as BLPSynapse
import pandas as pd

import bglibpy
from bglibpy import lazy_printv
from bglibpy.cell.sonata_proxy import SonataProxy
from bglibpy.circuit import CircuitAccess, parse_outdat, SimulationValidator
from bglibpy.exceptions import BGLibPyError
from bglibpy.simulation import (
    set_minis_single_vesicle_values,
    set_global_condition_parameters,
    set_tstop_value
)


class SSim:

    """Class that can load a BGLib BlueConfig,
               and instantiate the simulation"""

    # pylint: disable=R0913

    def __init__(self, blueconfig_filename, dt=0.025, record_dt=None,
                 base_seed=None, base_noise_seed=None, rng_mode=None,
                 ignore_populationid_error=False, print_cellstate=False):
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
        print_cellstate: bool
                    Flag to use NEURON prcellstate for simulation GIDs
        """
        self.dt = dt
        self.record_dt = record_dt
        self.blueconfig_filename = blueconfig_filename
        self.circuit_access = CircuitAccess(blueconfig_filename)
        SimulationValidator(self.circuit_access).validate()

        self.pc = bglibpy.neuron.h.ParallelContext() if print_cellstate else None

        self.rng_settings = bglibpy.RNGSettings(
            rng_mode,
            self.circuit_access.bc,
            base_seed=base_seed,
            base_noise_seed=base_noise_seed)

        self.ignore_populationid_error = ignore_populationid_error

        self.gids = []
        self.cells = {}

        self.gids_instantiated = False
        self.connections = \
            collections.defaultdict(
                lambda: collections.defaultdict(
                    lambda: None))

        # Make sure tstop is set correctly, because it is used by the
        # TStim noise stimulus
        if 'Duration' in self.circuit_access.bc.Run:
            set_tstop_value(float(self.circuit_access.bc.Run['Duration']))

        self.spike_threshold = \
            float(self.circuit_access.bc.Run['SpikeThreshold']) \
            if 'SpikeThreshold' in self.circuit_access.bc.Run else -30

        if 'SpikeLocation' in self.circuit_access.bc.Run:
            self.spike_location = self.circuit_access.bc.Run['SpikeLocation']
        else:
            self.spike_location = "soma"

        if "MinisSingleVesicle" in self.circuit_access.bc.Run:
            warnings.warn("""Specifying MinisSingleVesicle in the run block
                             is deprecated, please use the conditions block.""")
            minis_single_vesicle = int(self.circuit_access.bc.Run["MinisSingleVesicle"])
            set_minis_single_vesicle_values(minis_single_vesicle)

        condition_parameters = self.circuit_access.condition_parameters_dict()
        set_global_condition_parameters(condition_parameters)

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
                         projections=None,
                         add_shotnoise_stimuli=False):
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
        add_shotnoise_stimuli : Boolean
                            Process the 'shotnoise' stimuli blocks of the
                            BlueConfig,
                            Setting add_stimuli=True,
                            will automatically set this option to True.
        """

        if synapse_detail is not None:
            lazy_printv(
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
        elif add_synapses is None:
            add_synapses = False

        if add_projections:
            if projections is not None:
                raise ValueError('SSim: projections and add_projections '
                                 'can not be used at the same time')
            projections = [proj.name
                           for proj in self.circuit_access.bc.typed_sections('Projection')]

        self._add_cells(gids)
        if add_stimuli:
            add_noise_stimuli = True
            add_hyperpolarizing_stimuli = True
            add_relativelinear_stimuli = True
            add_pulse_stimuli = True
            add_shotnoise_stimuli = True

        if add_noise_stimuli or \
                add_hyperpolarizing_stimuli or \
                add_pulse_stimuli or \
                add_relativelinear_stimuli or \
                add_shotnoise_stimuli:
            self._add_stimuli(
                add_noise_stimuli=add_noise_stimuli,
                add_hyperpolarizing_stimuli=add_hyperpolarizing_stimuli,
                add_relativelinear_stimuli=add_relativelinear_stimuli,
                add_pulse_stimuli=add_pulse_stimuli,
                add_shotnoise_stimuli=add_shotnoise_stimuli)
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
                     add_pulse_stimuli=False,
                     add_shotnoise_stimuli=False):
        """Instantiate all the stimuli"""
        for gid in self.gids:
            # Also add the injections / stimulations as in the cortical model
            self._add_stimuli_gid(
                gid,
                add_noise_stimuli=add_noise_stimuli,
                add_hyperpolarizing_stimuli=add_hyperpolarizing_stimuli,
                add_relativelinear_stimuli=add_relativelinear_stimuli,
                add_pulse_stimuli=add_pulse_stimuli,
                add_shotnoise_stimuli=add_shotnoise_stimuli)
            lazy_printv("Added stimuli for gid {gid}", 2, gid=gid)

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
                    gid, pre_gids=intersect_pre_gids,
                    add_minis=add_minis)
            if add_projection_synapses:
                self._add_gid_synapses(
                    gid, pre_gids=intersect_pre_gids,
                    add_minis=add_minis,
                    projection=projection,
                    projections=projections)

    def _add_gid_synapses(
        self, gid: int, pre_gids=None, add_minis=None, projection=None, projections=None
    ) -> None:
        syn_descriptions = self.get_syn_descriptions(
            gid, projection=projection, projections=projections)

        if pre_gids is not None:
            syn_descriptions = self._intersect_pre_gids(
                syn_descriptions, pre_gids)

        # Check if there are any presynaptic cells, otherwise skip adding
        # synapses
        if syn_descriptions.empty:
            lazy_printv(
                "Warning: No presynaptic cells found for gid {gid}, "
                "no synapses added", 2, gid=gid)
        else:
            for idx, syn_description in syn_descriptions.iterrows():
                popids = syn_description["source_popid"], syn_description["target_popid"]
                self._instantiate_synapse(gid, idx, syn_description,
                                          add_minis=add_minis, popids=popids)
            lazy_printv("Added {s_desc_len} synapses for gid {gid}",
                        2, s_desc_len=len(syn_descriptions), gid=gid)
            if add_minis:
                lazy_printv("Added minis for gid %d" % gid, 2)

    @staticmethod
    def _intersect_pre_gids(syn_descriptions, pre_gids) -> pd.DataFrame:
        """Return the synapse descriptions with pre_gids intersected."""
        return syn_descriptions[syn_descriptions[BLPSynapse.PRE_GID].isin(pre_gids)]

    def get_syn_descriptions(
        self, gid, projection=None, projections=None
    ) -> pd.DataFrame:
        """Get synapse descriptions dataframe."""
        syn_description_builder = bglibpy.synapse.SynDescription()
        if self.circuit_access.is_glusynapse_used:
            return syn_description_builder.glusynapse_syn_description(
                self.circuit_access.bluepy_circuit,
                self.circuit_access.bc,
                self.ignore_populationid_error,
                gid,
                projection,
                projections,
            )
        else:
            return syn_description_builder.gabaab_ampanmda_syn_description(
                self.circuit_access.bluepy_circuit,
                self.circuit_access.bc,
                self.ignore_populationid_error,
                gid,
                projection,
                projections,
            )

    @staticmethod
    def merge_pre_spike_trains(*train_dicts) -> dict:
        """Merge presynaptic spike train dicts"""
        filtered_dicts = [d for d in train_dicts if d not in [None, {}, [], ()]]

        all_keys = set().union(*[d.keys() for d in filtered_dicts])
        return {
            k: np.sort(np.concatenate([d[k] for d in filtered_dicts if k in d]))
            for k in all_keys
        }

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
                    self.circuit_access.bc.Run['OutputRoot'],
                    'out.dat')
            pre_spike_trains = parse_outdat(outdat_path)
        else:
            pre_spike_trains = {}

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
                pre_gid = syn_description[BLPSynapse.PRE_GID]
                if source and pre_gid not in source:
                    continue
                real_synapse_connection = pre_gid in self.gids \
                    and interconnect_cells

                connection = None
                if real_synapse_connection:
                    if (
                            user_pre_spike_trains is not None
                            and pre_gid in user_pre_spike_trains
                    ):
                        raise BGLibPyError(
                            """Specifying prespike trains of real connections"""
                            """ is not allowed."""
                        )
                    connection = bglibpy.Connection(
                        self.cells[post_gid].synapses[syn_id],
                        pre_spiketrain=None,
                        pre_cell=self.cells[pre_gid],
                        stim_dt=self.dt,
                        parallel_context=self.pc,
                        spike_threshold=self.spike_threshold,
                        spike_location=self.spike_location)
                    lazy_printv("Added real connection between pre_gid %d and \
                            post_gid %d, syn_id %s" % (pre_gid,
                                                       post_gid,
                                                       str(syn_id)), 5)
                else:
                    pre_spiketrain = pre_spike_trains.setdefault(pre_gid, None)
                    connection = bglibpy.Connection(
                        self.cells[post_gid].synapses[syn_id],
                        pre_spiketrain=pre_spiketrain,
                        pre_cell=None,
                        stim_dt=self.dt,
                        spike_threshold=self.spike_threshold,
                        spike_location=self.spike_location)
                    lazy_printv(
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
                lazy_printv("Added synaptic connections for target post_gid %d" %
                            post_gid, 2)

    def _add_cells(self, gids: List[int]) -> None:
        """Instantiate cells from a gid list."""
        self.gids = gids
        self.cells = {}

        for gid in self.gids:
            lazy_printv(
                'Adding gid {gid} from emodel {emodel} and morph {morph}',
                1, gid=gid, emodel=self.circuit_access.fetch_emodel_name(gid),
                morph=self.circuit_access.fetch_morph_name(gid))

            self.cells[gid] = cell = bglibpy.Cell(**self.fetch_cell_kwargs(gid))
            if self.circuit_access.node_properties_available:
                cell.connect_to_circuit(SonataProxy(gid, self.circuit_access))
            if self.pc is not None:
                self.pc.set_gid2node(gid, self.pc.id())  # register GID for this node
                nc = self.cells[gid].create_netcon_spikedetector(
                    None, location=self.spike_location, threshold=self.spike_threshold)
                self.pc.cell(gid, nc)  # register cell spike detector

    def _instantiate_synapse(self, gid: int, syn_id, syn_description,
                             add_minis=None, popids=(0, 0)) -> None:
        """Instantiate one synapse for a given gid, syn_id and
        syn_description"""

        syn_type = syn_description[BLPSynapse.TYPE]

        connection_parameters = self._evaluate_connection_parameters(
            syn_description[BLPSynapse.PRE_GID],
            gid,
            syn_type)

        if connection_parameters["add_synapse"]:
            condition_parameters = self.circuit_access.condition_parameters_dict()

            self.cells[gid].add_replay_synapse(
                syn_id, syn_description, connection_parameters, condition_parameters,
                popids=popids, extracellular_calcium=self.circuit_access.extracellular_calcium)
            if add_minis:
                mini_frequencies = self.circuit_access.fetch_mini_frequencies(gid)
                lazy_printv('Adding minis for synapse {sid}: syn_description={s_desc}, '
                            'connection={conn_params}, frequency={freq}',
                            50, sid=syn_id, s_desc=syn_description,
                            conn_params=connection_parameters, freq=mini_frequencies)

                self.cells[gid].add_replay_minis(
                    syn_id,
                    syn_description,
                    connection_parameters,
                    popids=popids,
                    mini_frequencies=mini_frequencies)

    def _add_stimuli_gid(self,
                         gid: int,
                         add_noise_stimuli=False,
                         add_hyperpolarizing_stimuli=False,
                         add_relativelinear_stimuli=False,
                         add_pulse_stimuli=False,
                         add_shotnoise_stimuli=False):
        """ Adds identical stimuli to the simulated cell as in the 'large'
            model

        Parameters
        -----------
        gid: gid of the simulated cell
        """
        # check in which StimulusInjects the gid is a target
        # Every noise or shot noise stimulus gets a new seed
        noisestim_count = 0
        shotnoise_stim_count = 0

        for entry in self.circuit_access.bc.values():
            if entry.section_type == 'StimulusInject':
                destination = entry.Target
                # retrieve the stimulus to apply
                stimulus_name = entry.Stimulus
                # bluepy magic to add underscore Stimulus underscore
                # stimulus_name
                stimulus = self.circuit_access.bc['Stimulus_%s' % stimulus_name]
                gids_of_target = self.circuit_access.bluepy_circuit.cells.ids(destination)
                if gid in gids_of_target:
                    if stimulus.Pattern == 'Noise':
                        if add_noise_stimuli:
                            self.cells[gid].add_replay_noise(
                                stimulus, noisestim_count=noisestim_count)
                    elif stimulus.Pattern == 'Hyperpolarizing':
                        if add_hyperpolarizing_stimuli:
                            self.cells[gid].add_replay_hypamp(stimulus)
                    elif stimulus.Pattern == 'Pulse':
                        if add_pulse_stimuli:
                            self.cells[gid].add_pulse(stimulus)
                    elif stimulus.Pattern == 'RelativeLinear':
                        if add_relativelinear_stimuli:
                            self.cells[gid].add_replay_relativelinear(stimulus)
                    elif stimulus.Pattern == 'SynapseReplay':
                        lazy_printv("Found stimulus with pattern %s, ignoring" %
                                    stimulus['Pattern'], 1)
                    elif stimulus.Pattern == 'ShotNoise':
                        if add_shotnoise_stimuli:
                            self.cells[gid].add_replay_shotnoise(
                                self.cells[gid].soma, 0.5, stimulus,
                                shotnoise_stim_count=shotnoise_stim_count)
                    elif stimulus.Pattern == 'RelativeShotNoise':
                        if add_shotnoise_stimuli:
                            self.cells[gid].add_replay_relative_shotnoise(
                                self.cells[gid].soma, 0.5, stimulus,
                                shotnoise_stim_count=shotnoise_stim_count)
                    else:
                        raise Exception("Found stimulus with pattern %s, "
                                        "not supported" %
                                        stimulus.Pattern)
                if stimulus.Pattern == 'Noise':
                    noisestim_count += 1
                if stimulus.Pattern in ['ShotNoise', 'RelativeShotNoise']:
                    shotnoise_stim_count += 1

    def _evaluate_connection_parameters(self, pre_gid, post_gid, syn_type) -> dict:
        """ Apply connection blocks in order for pre_gid, post_gid to determine
            a final connection override for this pair (pre_gid, post_gid)."""
        parameters: DefaultDict[str, Any] = collections.defaultdict(list)
        parameters['add_synapse'] = True

        for entry in self.circuit_access.connection_entries:
            src, dest = entry['Source'], entry['Destination']

            src_matches = self.circuit_access.is_cell_target(src, pre_gid) or \
                (self.circuit_access.target_has_gid(src, pre_gid))
            dest_matches = self.circuit_access.is_cell_target(dest, post_gid) or \
                (self.circuit_access.target_has_gid(dest, post_gid))

            if src_matches and dest_matches:
                # whatever specified in this block, is applied to gid
                apply_parameters = True

                if 'SynapseID' in entry and int(entry['SynapseID']) != syn_type:
                    apply_parameters = False

                if 'Delay' in entry:
                    parameters['DelayWeights'].append((
                        float(entry['Delay']),
                        float(entry['Weight'])))
                    apply_parameters = False

                if apply_parameters:
                    if 'CreateMode' in entry:
                        if entry['CreateMode'] == 'NoCreate':
                            parameters['add_synapse'] = False
                        else:
                            raise Exception('Connection %s: Unknown '
                                            'CreateMode option %s'
                                            % (entry.name,
                                               entry['CreateMode']))
                    if 'Weight' in entry:
                        parameters['Weight'] = float(entry['Weight'])
                    if 'SpontMinis' in entry:
                        parameters['SpontMinis'] = float(
                            entry['SpontMinis'])
                    if 'SynapseConfigure' in entry:
                        conf = entry['SynapseConfigure']
                        # collect list of applicable configure blocks to be
                        # applied with a "hoc exec" statement
                        parameters['SynapseConfigure'].append(conf)
                    if 'ModOverride' in entry:
                        mod_name = entry['ModOverride']
                        parameters['ModOverride'] = mod_name

        return parameters

    def initialize_synapses(self):
        """ Resets the state of all synapses of all cells to initial values """
        for cell in self.cells.values():
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
            t_stop = float(self.circuit_access.bc.Run['Duration'])
        if dt is None:
            dt = float(self.circuit_access.bc.Run['Dt'])
        if forward_skip_value is None:
            if 'ForwardSkip' in self.circuit_access.bc.Run:
                forward_skip_value = float(
                    self.circuit_access.bc.Run['ForwardSkip'])
        if celsius is None:
            if 'Celsius' in self.circuit_access.bc.Run:
                celsius = float(self.circuit_access.bc.Run['Celsius'])
            else:
                celsius = 34  # default
        if v_init is None:
            if 'V_Init' in self.circuit_access.bc.Run:
                v_init = float(self.circuit_access.bc.Run['V_Init'])
            else:
                v_init = -65  # default

        sim = bglibpy.Simulation(self.pc)
        for gid in self.gids:
            sim.add_cell(self.cells[gid])

        if show_progress:
            lazy_printv("Warning: show_progress enabled, this will very likely"
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
        """Get the voltage trace from a cell from the main simulation.

        Parameters
        -----------
        gid: GID of interest.
        t_start, t_stop: time range of interest,
        report time range is used by default.
        t_step: time step (should be a multiple of report time step T;
        equals T by default)

        Returns:
            One dimentional np.ndarray to represent the voltages.
        """

        voltage = (
            self.circuit_access.bluepy_sim.report("soma")
            .get_gid(gid, t_start=t_start, t_end=t_stop, t_step=t_step)
            .to_numpy()
        )
        return voltage

    def get_mainsim_time_trace(self):
        """Get the time trace from the main simulation"""

        report = self.circuit_access.bluepy_sim.report('soma')
        time = report.get_gid(report.gids[0]).index.to_numpy()
        return time

    def get_time(self):
        """Get the time vector for the recordings, contains negative times.

        The negative times occur as a result of ForwardSkip.
        """
        return self.cells[self.gids[0]].get_time()

    def get_time_trace(self):
        """Get the time vector for the recordings, negative times removed"""
        time = self.get_time()
        return time[np.where(time >= 0.0)]

    def get_voltage_trace(self, gid):
        """Get the voltage vector for the gid, negative times removed"""

        time = self.get_time()
        voltage = self.cells[gid].get_soma_voltage()
        return voltage[np.where(time >= 0.0)]

    def delete(self):
        """Delete ssim and all of its attributes.

        NEURON objects are explicitly needed to be deleted.
        """
        if hasattr(self, 'cells'):
            for _, cell in self.cells.items():
                cell.delete()
            gids = list(self.cells.keys())
            for gid in gids:
                del self.cells[gid]

    def __del__(self):
        """Destructor"""
        self.delete()

    def fetch_cell_kwargs(self, gid):
        """Get the kwargs to instantiate a gid's Cell object"""
        if self.circuit_access.use_mecombo_tsv or self.circuit_access.node_properties_available:
            template_format = 'v6'

            if self.circuit_access.use_mecombo_tsv:
                emodel_properties = self.circuit_access.bluepy_circuit.emodels.get_mecombo_info(gid)
                extra_values = {
                    'threshold_current': emodel_properties["threshold_current"],
                    'holding_current': emodel_properties["holding_current"]
                }
            elif self.circuit_access.node_properties_available:
                emodel_properties = self.circuit_access.bluepy_circuit.cells.get(
                    gid,
                    properties=["@dynamics:threshold_current", "@dynamics:holding_current", ],
                )
                extra_values = {
                    'threshold_current': emodel_properties["@dynamics:threshold_current"],
                    'holding_current': emodel_properties["@dynamics:holding_current"]
                }

                if "@dynamics:AIS_scaler" in self.circuit_access.bluepy_circuit.cells.available_properties:
                    template_format = 'v6_ais_scaler'
                    extra_values['AIS_scaler'] = self.circuit_access.bluepy_circuit.cells.get(
                        gid,
                        properties=["@dynamics:AIS_scaler", ])["@dynamics:AIS_scaler"]

            cell_kwargs = {
                'template_filename': self.circuit_access.emodel_path(gid),
                'morphology_name': self.circuit_access.morph_filename(gid),
                'gid': gid,
                'record_dt': self.record_dt,
                'rng_settings': self.rng_settings,

                'template_format': template_format,
                'morph_dir': self.circuit_access.morph_dir,
                'extra_values': extra_values,
            }
        else:
            cell_kwargs = {
                'template_filename': self.circuit_access.emodel_path(gid),
                'morphology_name': self.circuit_access.morph_dir,
                'gid': gid,
                'record_dt': self.record_dt,
                'rng_settings': self.rng_settings,
            }

        return cell_kwargs

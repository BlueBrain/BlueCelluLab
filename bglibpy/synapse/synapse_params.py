"""Hoc compatible synapse parameters representation."""

import pandas as pd

from bluepy.enums import Synapse as BLPSynapse
from bluepy.impl.connectome_sonata import SonataConnectome

import bglibpy
from bglibpy import printv, BGLibPyError


def create_syn_description_dict(bc_circuit, bc, ignore_populationid_error, gid, projection=None, projections=None):
    """Get synapse descriptions dict from bluepy, Keys are synapse ids

    Args:
        bc_circuit (bluepy.circuit.Circuit): bluepy circuit object
        bc (bluepy_configfile.configfile.BlueConfigFile): blueconfig object
        ignore_populationid_error (bool): whether to ignore the error
        gid (int): cell identification
        projection (string): name of the projection of interest. Defaults to None.
        projections (list): list of projections. Defaults to None.

    Raises:
        BGLibPyError: raised when gid doesn't match with syn_gid.
        BGLibPyError: raised when there is a duplicated synapse id.

    Returns:
        dict: indexed by projection name and synapse id.
        Values contain synapse properties series and popids tuple.
    """
    all_properties = [
        BLPSynapse.PRE_GID,
        BLPSynapse.AXONAL_DELAY,
        BLPSynapse.POST_SECTION_ID,
        BLPSynapse.POST_SEGMENT_ID,
        BLPSynapse.POST_SEGMENT_OFFSET,
        BLPSynapse.G_SYNX,
        BLPSynapse.U_SYN,
        BLPSynapse.D_SYN,
        BLPSynapse.F_SYN,
        BLPSynapse.DTC,
        BLPSynapse.TYPE,
        BLPSynapse.NRRP,
        BLPSynapse.U_HILL_COEFFICIENT,
        BLPSynapse.CONDUCTANCE_RATIO]

    connectomes = get_connectomes_dict(bc_circuit, projection=projection, projections=projections)

    all_synapse_sets = get_synapses_by_connectomes(connectomes, all_properties, gid)

    syn_descriptions_dict = {}
    if not all_synapse_sets:
        printv('No synapses found', 5)
    else:
        printv(
            'Adding a total of %d synapse sets' %
            len(all_synapse_sets), 5)

        for proj_name, (synapse_set, using_sonata) in all_synapse_sets.items():
            printv(
                'Adding a total of %d synapses for set %s' %
                (synapse_set.shape[0], proj_name), 5)

            popids = get_popids(bc, ignore_populationid_error, proj_name)

            for syn_id, (edge_id, synapse) in enumerate(
                    synapse_set.iterrows()):
                if not using_sonata:
                    edge_id_series = pd.Series(data=edge_id, index=("syn_gid", "syn_id"))
                    if edge_id_series.syn_gid != gid:
                        raise BGLibPyError(
                            "BGLibPy SSim: synapse gid doesnt match with "
                            "cell gid !")
                else:  # edge id is a single value
                    edge_id_series = pd.Series(data=[edge_id], index=["edge_id"])
                syn_id_proj = (proj_name, syn_id)
                if syn_id_proj in syn_descriptions_dict:
                    raise BGLibPyError(
                        "BGLibPy SSim: trying to add "
                        "synapse id %s twice!" %
                        syn_id_proj)
                if BLPSynapse.NRRP in synapse:
                    old_syn_description = synapse[:11]  # deletes nrrp at 11

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
                    # 14 - 16 are dummy values, 17 is Nrrp
                    ext_description = pd.Series(
                        data=[-1, -1, -1, synapse[BLPSynapse.NRRP],
                              u_hill_coefficient, conductance_ratio],
                        index=["dummy1", "dummy2", "dummy3", BLPSynapse.NRRP,
                               BLPSynapse.U_HILL_COEFFICIENT, BLPSynapse.CONDUCTANCE_RATIO],
                        dtype=object)
                    synapse = old_syn_description.append(ext_description)

                series_idx = synapse.index
                series_idx = series_idx.insert(5, "placeholder-3")
                series_idx = series_idx.insert(5, "placeholder-2")
                series_idx = series_idx.insert(5, "placeholder-1")

                synapse = synapse.reindex(series_idx, fill_value=-1)

                synapse = synapse.append(edge_id_series)
                syn_descriptions_dict[syn_id_proj] = synapse, popids

    return syn_descriptions_dict


def get_popids(bc, ignore_populationid_error, proj_name):
    """Retrieve the population ids of a projection.

    Args:
        bc (bluepy_configfile.configfile.BlueConfigFile): blueconfig object
        ignore_populationid_error (bool): whether to ignore the error
        proj_name (str): name of the projection

    Raises:
        bglibpy.PopulationIDMissingError: if the id is missing and this error
        is not ignored.

    Returns:
        tuple: source and target population ids.
    """
    if proj_name in [proj.name for proj in bc.typed_sections("Projection")]:
        if "PopulationID" in bc["Projection_" + proj_name]:
            source_popid = int(bc["Projection_" + proj_name]["PopulationID"])
        else:
            if ignore_populationid_error:
                source_popid = 0
            else:
                raise bglibpy.PopulationIDMissingError(
                    "PopulationID is missing from projection,"
                    " block this will lead to wrong rng seeding."
                    " If you anyway want to overwrite this,"
                    " pass ignore_populationid_error=True param"
                    " to SSim constructor.")
    else:
        source_popid = 0
    # ATM hard coded in neurodamus
    target_popid = 0
    popids = (source_popid, target_popid)
    return popids


def get_connectomes_dict(bc_circuit, projection, projections):
    """Get the connectomes dictionary indexed by projections or connectome ids.
    If projections are missing, indexed by ''.

    Args:
        bc_circuit (bluepy.circuit.Circuit): bluepy circuit object
        projection (string): name of the projection of interest
        projections (list): list of projections

    Raises:
        ValueError: raised when projection and projections are both present.

    Returns:
        dict: connectome dictionary indexed by projections, if present.
    """
    if projection is not None:
        if projections is None:
            projections = [projection]
        else:
            raise ValueError(
                'Cannot combine projection and projections arguments.')

    if projections is None:
        connectomes = {'': bc_circuit.connectome}
    else:
        connectomes = {
            this_projection: bc_circuit.projection(this_projection)
            for this_projection in projections}
    return connectomes


def get_synapses_by_connectomes(connectomes, all_properties, gid):
    """Creates a dict of connectome ids and synapse properties dataframe.

    Args:
        connectomes (dict): Connectome dictionary
        all_properties (list): list of synapse properties
        gid (int): cell identification

    Returns:
        dict of pandas.DataFrame: synapses dataframes indexed by connectome ids
    """
    all_synapse_sets = {}
    for proj_name, connectome in connectomes.items():
        using_sonata = False

        # list() to make a copy
        connectome_properties = list(all_properties)

        # older circuit don't have these properties
        for test_property in [BLPSynapse.U_HILL_COEFFICIENT,
                              BLPSynapse.CONDUCTANCE_RATIO,
                              BLPSynapse.NRRP]:
            if test_property not in connectome.available_properties:
                connectome_properties.remove(test_property)
                printv(
                    'WARNING: %s not found, disabling' %
                    test_property, 50)

        if isinstance(connectome._impl, SonataConnectome):
            using_sonata = True
            printv('Using sonata style synapse file, not nrn.h5', 50)
            # load 'afferent_section_pos' instead of '_POST_DISTANCE'
            if 'afferent_section_pos' in connectome.available_properties:
                connectome_properties[
                    connectome_properties.index(BLPSynapse.POST_SEGMENT_OFFSET)
                ] = 'afferent_section_pos'

            synapses = connectome.afferent_synapses(
                gid, properties=connectome_properties
            )

            # replace '_POST_SEGMENT_ID' with -1 (as indicator for
            # synlocation_to_segx)
            if 'afferent_section_pos' in connectome.available_properties:
                synapses[BLPSynapse.POST_SEGMENT_ID] = -1

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

        all_synapse_sets[proj_name] = (synapses, using_sonata)
    return all_synapse_sets

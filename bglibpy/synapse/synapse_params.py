"""Hoc compatible synapse parameters representation."""

from bluepy.enums import Synapse as BLPSynapse
from bluepy.impl.connectome_sonata import SonataConnectome

import bglibpy
from bglibpy import printv, BGLibPyError


class SynDescription:
    """Retrieve syn descriptions for the defined properties."""

    def __init__(self) -> None:
        self.common_properties = [
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
            BLPSynapse.CONDUCTANCE_RATIO,
        ]

    def gabaab_ampanmda_syn_description(self, bc_circuit, bc,
                                        ignore_populationid_error, gid,
                                        projection=None, projections=None):
        """Wraps create_syn_description dict w. ampanmda/gabaab properties."""
        return create_syn_description_dict(
            self.common_properties, bc_circuit, bc, ignore_populationid_error,
            gid, projection, projections)

    def glusynapse_syn_description(self, bc_circuit, bc,
                                   ignore_populationid_error, gid,
                                   projection=None, projections=None):
        """Wraps create_syn_description dict with glusynapse properties."""
        glusynapse_only_properties = [
            "volume_CR", "rho0_GB", "Use_d_TM", "Use_p_TM", "gmax_d_AMPA",
            "gmax_p_AMPA", "theta_d", "theta_p"]
        all_properties = self.common_properties + glusynapse_only_properties
        return create_syn_description_dict(
            all_properties, bc_circuit, bc,
            ignore_populationid_error, gid, projection, projections)


def create_syn_description_dict(all_properties, bc_circuit, bc,
                                ignore_populationid_error, gid,
                                projection=None, projections=None):
    """Get synapse descriptions dict from bluepy, Keys are synapse ids

    Args:
        all_properties: list of properties of synapses to be retrieved
        bc_circuit (bluepy.circuit.Circuit): bluepy circuit object
        bc (bluepy_configfile.configfile.BlueConfigFile): blueconfig object
        ignore_populationid_error (bool): whether to ignore the error
        gid (int): cell identification
        projection (string): name of projection of interest. Defaults to None.
        projections (list): list of projections. Defaults to None.

    Raises:
        BGLibPyError: raised when gid doesn't match with syn_gid.
        BGLibPyError: raised when there is a duplicated synapse id.

    Returns:
        dict: indexed by projection name and synapse id.
        Values contain synapse properties series and popids tuple.
    """
    connectomes = get_connectomes_dict(bc_circuit, projection=projection,
                                       projections=projections)

    all_synapse_sets = get_synapses_by_connectomes(connectomes,
                                                   all_properties, gid)

    syn_descriptions_dict = {}
    if not all_synapse_sets:
        printv('No synapses found', 5)
    else:
        printv(f'Found a total of {len(all_synapse_sets)} synapse sets', 5)

        for proj_name, (synapse_set, using_sonata) in all_synapse_sets.items():
            printv('Retrieving a total of %d synapses for set %s' %
                   (synapse_set.shape[0], proj_name), 5)

            popids = get_popids(bc, ignore_populationid_error, proj_name)

            for syn_id, (edge_id, synapse) in enumerate(
                    synapse_set.iterrows()):

                syn_id_proj = (proj_name, syn_id)
                if syn_id_proj in syn_descriptions_dict:
                    raise BGLibPyError("BGLibPy SSim: trying to add "
                                       f"synapse id {syn_id_proj} twice!")
                if BLPSynapse.NRRP in synapse:
                    check_nrrp_value(gid, syn_id, synapse)

                if using_sonata:
                    synapse["edge_id"] = edge_id

                syn_descriptions_dict[syn_id_proj] = synapse, popids

    return syn_descriptions_dict


def check_nrrp_value(gid, syn_id, synapse_desc):
    """Assures the nrrp values fits the conditions.

    Args:
        gid (int): cell identification
        syn_id (int): synapse identification
        synapse_desc (pandas.Series): synapse description

    Raises:
        ValueError: when NRRP is <= 0
        ValueError: when NRRP cannot ve cast to integer
    """
    if synapse_desc[BLPSynapse.NRRP] <= 0:
        raise ValueError(
            'Value smaller than 0.0 found for Nrrp: '
            f'{synapse_desc[BLPSynapse.NRRP]} at synapse {syn_id} in gid {gid}'
        )
    if synapse_desc[BLPSynapse.NRRP] != int(synapse_desc[BLPSynapse.NRRP]):
        raise ValueError(
            'Non-integer value for Nrrp found: '
            f'{synapse_desc[BLPSynapse.NRRP]} at synapse {syn_id} in gid {gid}'
        )


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
        if "PopulationID" in bc[f"Projection_{proj_name}"]:
            source_popid = int(bc[f"Projection_{proj_name}"]["PopulationID"])
        elif ignore_populationid_error:
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
    return source_popid, target_popid


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

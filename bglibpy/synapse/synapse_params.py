"""Hoc compatible synapse parameters representation."""

from bluepy.enums import Synapse as BLPSynapse
from bluepy.impl.connectome_sonata import SonataConnectome
import pandas as pd

import bglibpy
from bglibpy import lazy_printv


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
        """Wraps create_syn_descriptions ampanmda/gabaab properties."""
        return create_syn_descriptions(
            self.common_properties, bc_circuit, bc, ignore_populationid_error,
            gid, projection, projections)

    def glusynapse_syn_description(self, bc_circuit, bc,
                                   ignore_populationid_error, gid,
                                   projection=None, projections=None):
        """Wraps create_syn_descriptions with glusynapse properties."""
        glusynapse_only_properties = [
            "volume_CR", "rho0_GB", "Use_d_TM", "Use_p_TM", "gmax_d_AMPA",
            "gmax_p_AMPA", "theta_d", "theta_p"]
        all_properties = self.common_properties + glusynapse_only_properties
        return create_syn_descriptions(
            all_properties, bc_circuit, bc,
            ignore_populationid_error, gid, projection, projections)


def create_syn_descriptions(all_properties, bc_circuit, bc,
                            ignore_populationid_error, gid,
                            projection=None, projections=None) -> pd.DataFrame:
    """Get synapse descriptions dataframe annotated with popids.

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
        pd.DataFrame: multiindex containing synapse description with popids
    """
    connectomes = get_connectomes_dict(bc_circuit, projection=projection,
                                       projections=projections)

    synapses = get_synapses_by_connectomes(connectomes, all_properties, gid)

    if BLPSynapse.NRRP in synapses:
        check_nrrp_value(synapses)

    proj_ids = synapses.index.get_level_values(0)
    pop_ids = [get_popids(bc, ignore_populationid_error, proj_id) for proj_id in proj_ids]
    source_popids, target_popids = zip(*pop_ids)
    synapses = synapses.assign(source_popid=source_popids, target_popid=target_popids)
    return synapses


def check_nrrp_value(synapses: pd.DataFrame) -> None:
    """Assures the nrrp values fits the conditions.

    Args:
        synapses: synapse description

    Raises:
        ValueError: when NRRP is <= 0
        ValueError: when NRRP cannot ve cast to integer
    """
    if any(synapses[BLPSynapse.NRRP] <= 0):
        raise ValueError(
            'Value smaller than 0.0 found for Nrrp: '
            f'{synapses[BLPSynapse.NRRP]} at synapse {synapses}.'
        )
    if any(synapses[BLPSynapse.NRRP] != synapses[BLPSynapse.NRRP].astype(int)):
        raise ValueError(
            'Non-integer value for Nrrp found: '
            f'{synapses[BLPSynapse.NRRP]} at synapse {synapses}.'
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


def get_synapses_by_connectomes(connectomes, all_properties, gid) -> pd.DataFrame:
    """Creates a dict of connectome ids and synapse properties dataframe.

    Args:
        connectomes (dict): Connectome dictionary
        all_properties (list): list of synapse properties
        gid (int): cell identification

    Returns:
        pandas.DataFrame: synapses dataframes indexed by projection, edge and synapse ids
    """
    all_synapses = pd.DataFrame()
    for proj_name, connectome in connectomes.items():
        connectome_properties = list(all_properties)

        # older circuit don't have these properties
        for test_property in [BLPSynapse.U_HILL_COEFFICIENT,
                              BLPSynapse.CONDUCTANCE_RATIO,
                              BLPSynapse.NRRP]:
            if test_property not in connectome.available_properties:
                connectome_properties.remove(test_property)
                lazy_printv(f'WARNING: {test_property} not found, disabling', 50)

        if isinstance(connectome._impl, SonataConnectome):
            lazy_printv('Using sonata style synapse file, not nrn.h5', 50)
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

        synapses = synapses.reset_index(drop=True)
        synapses.index = pd.MultiIndex.from_tuples(
            [(proj_name, x) for x in synapses.index],
            names=["proj_id", "synapse_id"])
        # io/synapse_reader.py:_patch_delay_fp_inaccuracies from
        # py-neurodamus
        dt = bglibpy.neuron.h.dt
        synapses[BLPSynapse.AXONAL_DELAY] = (
            synapses[BLPSynapse.AXONAL_DELAY] / dt + 1e-5
        ).astype('i4') * dt

        lazy_printv('Retrieving a total of {n_syn} synapses for set {syn_set}',
                    5, n_syn=synapses.shape[0], syn_set=proj_name)

        all_synapses = pd.concat([all_synapses, synapses])

    if all_synapses.empty:
        lazy_printv('No synapses found', 5)
    else:
        lazy_printv('Found a total of {n_syn_sets} synapse sets',
                    5, n_syn_sets=len(synapses))

    return all_synapses

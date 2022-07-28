"""Module that handles the global NEURON parameters."""

import bglibpy
from bglibpy import lazy_printv


def set_global_condition_parameters(condition_parameters: dict) -> None:
    """Sets the global condition parameters in NEURON objects."""
    if "cao_CR_GluSynapse" in condition_parameters:
        cao_cr_glusynapse = condition_parameters["cao_CR_GluSynapse"]
        bglibpy.neuron.h.cao_CR_GluSynapse = cao_cr_glusynapse

    if "SYNAPSES__init_depleted" in condition_parameters:
        init_depleted = condition_parameters["SYNAPSES__init_depleted"]
        bglibpy.neuron.h.init_depleted_GluSynapse = init_depleted
        bglibpy.neuron.h.init_depleted_ProbAMPANMDA_EMS = init_depleted
        bglibpy.neuron.h.init_depleted_ProbGABAAB_EMS = init_depleted
    if "SYNAPSES__minis_single_vesicle" in condition_parameters:
        minis_single_vesicle = int(
            condition_parameters["SYNAPSES__minis_single_vesicle"])
        set_minis_single_vesicle_values(minis_single_vesicle)


def set_minis_single_vesicle_values(minis_single_vesicle: int) -> None:
    """Set the minis_single_vesicle values in NEURON."""
    lazy_printv(
        f"Setting synapses minis_single_vesicle to {minis_single_vesicle}",
        50)
    bglibpy.neuron.h.minis_single_vesicle_ProbAMPANMDA_EMS = (
        minis_single_vesicle
    )
    bglibpy.neuron.h.minis_single_vesicle_ProbGABAAB_EMS = (
        minis_single_vesicle
    )
    bglibpy.neuron.h.minis_single_vesicle_GluSynapse = (
        minis_single_vesicle
    )


def set_tstop_value(tstop: float) -> None:
    """Set the tstop value required my Tstim noise stimuli."""
    bglibpy.neuron.h.tstop = tstop

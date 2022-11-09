"""Functionality for validation the simulation configuration."""

import re

import bglibpy
from bglibpy import ConfigError, TargetDoesNotExist
from bglibpy.circuit import CircuitAccess


class SimulationValidator:
    """Validates the simulation configuration, should be called before simulation."""

    def __init__(self, circuit_access: CircuitAccess) -> None:
        self.circuit_access = circuit_access

    def validate(self) -> None:
        self.check_connection_entries()
        self.check_single_vesicle_minis_settings()
        self.check_randomize_gaba_risetime()
        self.check_cao_cr_glusynapse_value()
        self.check_spike_location()
        self.check_mod_override_file()

    def check_connection_entries(self):
        """Check all connection entries at once"""
        gid_pttrn = re.compile("^a[0-9]+")
        for entry in self.circuit_access.config.connection_entries:
            self.check_connection_contents(entry)
            src = entry['Source']
            dest = entry['Destination']
            for target in (src, dest):
                if not (self.circuit_access.is_group_target(target) or gid_pttrn.match(target)):
                    raise TargetDoesNotExist("%s target does not exist" % target)

    @staticmethod
    def check_connection_contents(contents):
        """Check the contents of a connection block,
        to see if we support all the fields"""

        allowed_keys = set(['Weight', 'SynapseID', 'SpontMinis',
                            'SynapseConfigure', 'Source', 'ModOverride',
                            'Destination', 'Delay', 'CreateMode'])
        for key in contents.keys():
            if key not in allowed_keys:
                raise ConfigError(f"Key {key} in Connection blocks not supported by BGLibPy")

    def check_single_vesicle_minis_settings(self):
        """Check against missing single vesicle attribute."""
        condition_parameters = self.circuit_access.config.condition_parameters_dict()
        if "SYNAPSES__minis_single_vesicle" in condition_parameters:
            if not hasattr(
                    bglibpy.neuron.h, "minis_single_vesicle_ProbAMPANMDA_EMS"):
                raise bglibpy.OldNeurodamusVersionError(
                    "Synapses don't implement minis_single_vesicle."
                    "More recent neurodamus model required."
                )

    def check_randomize_gaba_risetime(self):
        """Make sure the gaba risetime has an expected value."""
        condition_parameters = self.circuit_access.config.condition_parameters_dict()
        if "randomize_Gaba_risetime" in condition_parameters:
            randomize_gaba_risetime = condition_parameters["randomize_Gaba_risetime"]

            if randomize_gaba_risetime not in ["True", "False", "0", "false"]:
                raise ConfigError("Invalid randomize_Gaba_risetime value"
                                  f": {randomize_gaba_risetime}.")

    def check_cao_cr_glusynapse_value(self):
        """Make sure cao_CR_GluSynapse is equal to ExtracellularCalcium."""
        condition_parameters = self.circuit_access.config.condition_parameters_dict()
        if "cao_CR_GluSynapse" in condition_parameters:
            cao_cr_glusynapse = condition_parameters["cao_CR_GluSynapse"]

            if cao_cr_glusynapse != self.circuit_access.config.extracellular_calcium:
                raise ConfigError("cao_CR_GluSynapse is not equal to ExtracellularCalcium")

    def check_spike_location(self):
        """Allow only accepted spike locations."""
        if 'SpikeLocation' in self.circuit_access.config.bc.Run:
            spike_location = self.circuit_access.config.bc.Run['SpikeLocation']
            if spike_location not in ["soma", "AIS"]:
                raise bglibpy.ConfigError(
                    "Possible options for SpikeLocation are 'soma' and 'AIS'")

    def check_mod_override_file(self):
        """Assure mod files are present for all overrides in connection blocks."""
        for entry in self.circuit_access.config.connection_entries:
            if 'ModOverride' in entry:
                mod_name = entry['ModOverride']
                if not hasattr(bglibpy.neuron.h, mod_name):
                    raise bglibpy.ConfigError(
                        f"Mod file for {mod_name} is not found.")

"""Module that interacts with the simulation config."""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
import warnings

from bluepy_configfile.configfile import BlueConfig
from bluepy.utils import open_utf8
from cachetools import LRUCache, cachedmethod

from bglibpy.exceptions import PopulationIDMissingError


class SimulationConfig:
    """Class that handles access to simulation config."""

    def __init__(self, config: str | BlueConfig) -> None:
        if isinstance(config, str):
            if not Path(config).exists():
                raise FileNotFoundError(f"Circuit config file {config} not found.")
            else:
                with open_utf8(config) as f:
                    self.bc = BlueConfig(f)
        else:
            self.bc = config

        self._caches: dict = {
            "condition_parameters_dict": LRUCache(maxsize=1)
        }

    def get_population_ids(
        self, ignore_populationid_error: bool, projection: str
    ) -> tuple[int, int]:
        """Retrieve the population ids of a projection.

        Raises:
            bglibpy.PopulationIDMissingError: if the id is missing and this error
            is not ignored.
        """
        if projection in self.get_all_projection_names():
            if "PopulationID" in self.bc[f"Projection_{projection}"]:
                source_popid = int(self.bc[f"Projection_{projection}"]["PopulationID"])
            elif ignore_populationid_error:
                source_popid = 0
            else:
                raise PopulationIDMissingError(
                    "PopulationID is missing from projection,"
                    " block this will lead to wrong rng seeding."
                    " If you anyway want to overwrite this,"
                    " pass ignore_populationid_error=True param"
                    " to SSim constructor."
                )
        else:
            source_popid = 0
        # ATM hard coded in neurodamus, commit: cd26654
        target_popid = 0
        return source_popid, target_popid

    def get_all_stimuli_entries(self) -> list[dict]:
        """Get all stimuli entries."""
        result = []
        for entry in self.bc.typed_sections('StimulusInject'):
            # retrieve the stimulus to apply
            stimulus_name: str = entry.Stimulus
            # bluepy magic to add underscore Stimulus underscore
            # stimulus_name
            stimulus = self.bc['Stimulus_%s' % stimulus_name].to_dict()
            stimulus["Target"] = entry.Target
            result.append(stimulus)
        return result

    def get_all_projection_names(self) -> list[str]:
        unique_names = {proj.name for proj in self.bc.typed_sections('Projection')}
        return list(unique_names)

    @cachedmethod(lambda self: self._caches["condition_parameters_dict"])
    def condition_parameters_dict(self) -> dict:
        """Returns parameters of global condition block of the blueconfig."""
        try:
            condition_entries = self.bc.typed_sections('Conditions')[0]
        except IndexError:
            return {}
        return condition_entries.to_dict()

    @property
    def connection_entries(self) -> list:
        return self.bc.typed_sections('Connection')

    @property
    def is_glusynapse_used(self) -> bool:
        """Checks if glusynapse is used in the simulation config."""
        is_glusynapse_used = any(
            x
            for x in self.connection_entries
            if "ModOverride" in x and x["ModOverride"] == "GluSynapse"
        )
        return is_glusynapse_used

    def get_morph_dir_and_extension(self) -> tuple[str, str]:
        """Get the tuple of morph_dir and extension."""
        morph_dir = self.bc.Run['MorphologyPath']

        if 'MorphologyType' in self.bc.Run:
            morph_extension = self.bc.Run['MorphologyType']
        else:
            # backwards compatible
            if morph_dir[-3:] == "/h5":
                morph_dir = morph_dir[:-3]

            # latest circuits don't have asc dir
            asc_dir = os.path.join(morph_dir, 'ascii')
            if os.path.exists(asc_dir):
                morph_dir = asc_dir

            morph_extension = 'asc'

        return morph_dir, morph_extension

    @property
    def morph_dir(self) -> str:
        _morph_dir, _ = self.get_morph_dir_and_extension()
        return _morph_dir

    @property
    def morph_extension(self) -> str:
        _, _morph_extension = self.get_morph_dir_and_extension()
        return _morph_extension

    @property
    def emodels_dir(self) -> str:
        return self.bc.Run['METypePath']

    @property
    def base_seed(self) -> int:
        """Base seed of blueconfig."""
        return int(self.bc.Run['BaseSeed']) if 'BaseSeed' in self.bc.Run else 0

    @property
    def synapse_seed(self) -> int:
        """Synapse seed of blueconfig."""
        return int(self.bc.Run['SynapseSeed']) if 'SynapseSeed' in self.bc.Run else 0

    @property
    def ionchannel_seed(self) -> int:
        """Ion channel seed of blueconfig."""
        return int(self.bc.Run['IonChannelSeed']) if 'IonChannelSeed' in self.bc.Run else 0

    @property
    def stimulus_seed(self) -> int:
        """Stimulus seed of blueconfig."""
        return int(self.bc.Run['StimulusSeed']) if 'StimulusSeed' in self.bc.Run else 0

    @property
    def minis_seed(self) -> int:
        """Minis seed of blueconfig."""
        return int(self.bc.Run['MinisSeed']) if 'MinisSeed' in self.bc.Run else 0

    @property
    def rng_mode(self) -> str:
        """Gets the rng mode defined in simulation."""
        # Ugly, but mimicking neurodamus
        if 'Simulator' in self.bc.Run and self.bc.Run['Simulator'] != 'NEURON':
            return 'Random123'
        elif 'RNGMode' in self.bc.Run:
            return self.bc.Run['RNGMode']
        else:
            return "Compatibility"  # default rng mode

    @property
    def spike_threshold(self) -> float:
        """Get the spike threshold from simulation config."""
        if 'SpikeThreshold' in self.bc.Run:
            return float(self.bc.Run['SpikeThreshold'])
        else:
            return -30.0

    @property
    def spike_location(self) -> str:
        """Get the spike location from simulation config."""
        if 'SpikeLocation' in self.bc.Run:
            return self.bc.Run['SpikeLocation']
        else:
            return "soma"

    @property
    def duration(self) -> Optional[float]:
        """Get the duration of the simulation."""
        if 'Duration' in self.bc.Run:
            return float(self.bc.Run['Duration'])
        else:
            return None

    @property
    def dt(self) -> float:
        return float(self.bc.Run['Dt'])

    @property
    def forward_skip(self) -> Optional[float]:
        if 'ForwardSkip' in self.bc.Run:
            return float(self.bc.Run['ForwardSkip'])
        return None

    @property
    def celsius(self) -> float:
        if 'Celsius' in self.bc.Run:
            return float(self.bc.Run['Celsius'])
        else:
            return 34.0  # default

    @property
    def v_init(self) -> float:
        if 'V_Init' in self.bc.Run:
            return float(self.bc.Run['V_Init'])
        else:
            return -65.0

    @property
    def output_root_path(self) -> str:
        """Get the output root path."""
        return self.bc.Run['OutputRoot']

    @property
    def deprecated_minis_single_vesicle(self) -> Optional[int]:
        """Get the minis single vesicle from simulation config."""
        if 'MinisSingleVesicle' in self.bc.Run:
            warnings.warn("""Specifying MinisSingleVesicle in the run block
                             is deprecated, please use the conditions block.""")
            return int(self.bc.Run['MinisSingleVesicle'])
        else:
            return None

    @property
    def extracellular_calcium(self) -> Optional[float]:
        """Get the extracellular calcium value."""
        if 'ExtracellularCalcium' in self.bc.Run:
            return float(self.bc.Run['ExtracellularCalcium'])
        else:
            return None

    def add_section(
        self,
        section_type: str,
        name: str,
        contents: str,
        span: Optional[tuple[int, int]] = None,
        decl_comment: str = "",
    ) -> None:
        self.bc.add_section(section_type, name, contents, span, decl_comment)

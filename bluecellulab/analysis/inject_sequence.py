"""Module for injecting a sequence of protocols to the cell."""
from __future__ import annotations
from enum import Enum, auto
from typing import NamedTuple

import neuron
import numpy as np
from bluecellulab.cell.core import Cell
from bluecellulab.cell.template import TemplateParams
from bluecellulab.simulation.simulation import Simulation
from bluecellulab.stimulus.factory import Stimulus, StimulusFactory
from bluecellulab.type_aliases import NeuronSection
from bluecellulab.utils import IsolatedProcess


class StimulusName(Enum):
    """Allowed values for the StimulusName."""
    AP_WAVEFORM = auto()
    IDREST = auto()
    IV = auto()
    FIRE_PATTERN = auto()


class Recording(NamedTuple):
    """A tuple of the current, voltage and time recordings."""
    current: np.ndarray
    voltage: np.ndarray
    time: np.ndarray


StimulusRecordings = dict[str, Recording]


def run_stimulus(
    template_params: TemplateParams,
    stimulus: Stimulus,
    section: NeuronSection | None,
    segment: float,
    duration: float,
) -> Recording:
    """Creates a cell and stimulates it with a given stimulus.

    Args:
        template_params: The parameters to create the cell from a template.
        stimulus: The input stimulus to inject into the cell.
        section: The section of the cell where the stimulus is to be injected.
        segment: The segment of the section where the stimulus is to be injected.
        duration: The duration for which the simulation is to be run.

    Returns:
        The voltage-time recording at the specified location.

    Raises:
        ValueError: If the time and voltage arrays are not the same length.
    """
    cell = Cell.from_template_parameters(template_params)
    cell.add_voltage_recording(section, segment)
    iclamp, _ = cell.inject_current_waveform(
        stimulus.time, stimulus.current, section=section, segx=segment
    )
    current_vector = neuron.h.Vector()
    current_vector.record(iclamp._ref_i)
    simulation = Simulation(cell)
    simulation.run(duration)
    current = np.array(current_vector.to_python())
    voltage = cell.get_voltage_recording(section, segment)
    time = cell.get_time()
    if len(time) != len(voltage) or len(time) != len(current):
        raise ValueError("Time, current and voltage arrays are not the same length")
    return Recording(current, voltage, time)


def apply_multiple_step_stimuli(
    cell: Cell,
    stimulus_name: StimulusName,
    amplitudes: list[float],
    duration: float,
    section: NeuronSection | None = None,
    segment: float = 0.5,
    n_processes: int | None = None,
) -> StimulusRecordings:
    """Apply multiple stimuli to the cell on isolated processes.

    Args:
        cell: The cell to which the stimuli are applied.
        stimulus_name: The name of the stimulus to apply.
        amplitudes: The amplitudes of the stimuli to apply.
        duration: The duration for which each stimulus is applied.
        section: The section of the cell where the stimuli are applied.
          If None, the stimuli are applied at the soma of the cell.
        segment: The segment of the section where the stimuli are applied.
        n_processes: The number of processes to use for running the stimuli.

    Returns:
        A dictionary where the keys are the names of the stimuli and the values
        are the recordings of the cell's response to each stimulus.

    Raises:
        ValueError: If the stimulus name is not recognized.
    """
    res: StimulusRecordings = {}
    stim_factory = StimulusFactory(dt=1.0)
    tasks = []
    with IsolatedProcess(processes=n_processes) as pool:
        for amplitude in amplitudes:
            if stimulus_name == StimulusName.AP_WAVEFORM:
                stimulus = stim_factory.ap_waveform(threshold_current=cell.threshold, threshold_percentage=amplitude)
            elif stimulus_name == StimulusName.IDREST:
                stimulus = stim_factory.idrest(threshold_current=cell.threshold, threshold_percentage=amplitude)
            elif stimulus_name == StimulusName.IV:
                stimulus = stim_factory.iv(threshold_current=cell.threshold, threshold_percentage=amplitude)
            elif stimulus_name == StimulusName.FIRE_PATTERN:
                stimulus = stim_factory.fire_pattern(threshold_current=cell.threshold, threshold_percentage=amplitude)
            else:
                raise ValueError("Unknown stimulus name.")

            result = pool.apply_async(run_stimulus, [cell.template_params, stimulus, section, segment, duration])
            tasks.append((f"{stimulus}_{amplitude}", result))

        for key, result in tasks:
            res[key] = result.get()

    return res

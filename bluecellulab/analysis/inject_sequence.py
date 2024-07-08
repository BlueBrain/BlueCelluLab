"""Module for injecting a sequence of protocols to the cell."""
from __future__ import annotations
from enum import Enum, auto
from typing import NamedTuple, Sequence, Dict

import neuron
import numpy as np
from bluecellulab.cell.core import Cell
from bluecellulab.cell.template import TemplateParams
from bluecellulab.simulation.parallel import IsolatedProcess
from bluecellulab.simulation.simulation import Simulation
from bluecellulab.stimulus.factory import Stimulus, StimulusFactory


class StimulusName(Enum):
    """Allowed values for the StimulusName."""
    AP_WAVEFORM = auto()
    IDREST = auto()
    IV = auto()
    FIRE_PATTERN = auto()
    POS_CHEOPS = auto()
    NEG_CHEOPS = auto()


class Recording(NamedTuple):
    """A tuple of the current, voltage and time recordings."""
    current: np.ndarray
    voltage: np.ndarray
    time: np.ndarray


StimulusRecordings = Dict[str, Recording]


def run_stimulus(
    template_params: TemplateParams,
    stimulus: Stimulus,
    section: str,
    segment: float,
    cvode: bool = True,
) -> Recording:
    """Creates a cell and stimulates it with a given stimulus.

    Args:
        template_params: The parameters to create the cell from a template.
        stimulus: The input stimulus to inject into the cell.
        section: Name of the section of cell where the stimulus is to be injected.
        segment: The segment of the section where the stimulus is to be injected.
        cvode: True to use variable time-steps. False for fixed time-steps.

    Returns:
        The voltage-time recording at the specified location.

    Raises:
        ValueError: If the time and voltage arrays are not the same length.
    """
    cell = Cell.from_template_parameters(template_params)
    neuron_section = cell.sections[section]
    cell.add_voltage_recording(neuron_section, segment)
    iclamp, _ = cell.inject_current_waveform(
        stimulus.time, stimulus.current, section=neuron_section, segx=segment
    )
    current_vector = neuron.h.Vector()
    current_vector.record(iclamp._ref_i)
    simulation = Simulation(cell)
    simulation.run(stimulus.stimulus_time, cvode=cvode)
    current = np.array(current_vector.to_python())
    voltage = cell.get_voltage_recording(neuron_section, segment)
    time = cell.get_time()
    if len(time) != len(voltage) or len(time) != len(current):
        raise ValueError("Time, current and voltage arrays are not the same length")
    return Recording(current, voltage, time)


def apply_multiple_stimuli(
    cell: Cell,
    stimulus_name: StimulusName,
    amplitudes: Sequence[float],
    threshold_based: bool = True,
    section_name: str | None = None,
    segment: float = 0.5,
    n_processes: int | None = None,
    cvode: bool = True,
) -> StimulusRecordings:
    """Apply multiple stimuli to the cell on isolated processes.

    Args:
        cell: The cell to which the stimuli are applied.
        stimulus_name: The name of the stimulus to apply.
        amplitudes: The amplitudes of the stimuli to apply.
        threshold_based: Whether to consider amplitudes to be
            threshold percentages or to be raw amplitudes.
        section_name: Section name of the cell where the stimuli are applied.
          If None, the stimuli are applied at the soma[0] of the cell.
        segment: The segment of the section where the stimuli are applied.
        n_processes: The number of processes to use for running the stimuli.
        cvode: True to use variable time-steps. False for fixed time-steps.

    Returns:
        A dictionary where the keys are the names of the stimuli and the values
        are the recordings of the cell's response to each stimulus.

    Raises:
        ValueError: If the stimulus name is not recognized.
    """
    res: StimulusRecordings = {}
    stim_factory = StimulusFactory(dt=1.0)
    task_args = []
    section_name = section_name if section_name is not None else "soma[0]"

    # Prepare arguments for each stimulus
    for amplitude in amplitudes:
        if threshold_based:
            thres_perc = amplitude
            amp = None
        else:
            thres_perc = None
            amp = amplitude

        if stimulus_name == StimulusName.AP_WAVEFORM:
            stimulus = stim_factory.ap_waveform(
                threshold_current=cell.threshold, threshold_percentage=thres_perc, amplitude=amp
            )
        elif stimulus_name == StimulusName.IDREST:
            stimulus = stim_factory.idrest(
                threshold_current=cell.threshold, threshold_percentage=thres_perc, amplitude=amp
            )
        elif stimulus_name == StimulusName.IV:
            stimulus = stim_factory.iv(
                threshold_current=cell.threshold, threshold_percentage=thres_perc, amplitude=amp
            )
        elif stimulus_name == StimulusName.FIRE_PATTERN:
            stimulus = stim_factory.fire_pattern(
                threshold_current=cell.threshold, threshold_percentage=thres_perc, amplitude=amp
            )
        elif stimulus_name == StimulusName.POS_CHEOPS:
            stimulus = stim_factory.pos_cheops(
                threshold_current=cell.threshold, threshold_percentage=thres_perc, amplitude=amp
            )
        elif stimulus_name == StimulusName.NEG_CHEOPS:
            stimulus = stim_factory.neg_cheops(
                threshold_current=cell.threshold, threshold_percentage=thres_perc, amplitude=amp
            )
        else:
            raise ValueError("Unknown stimulus name.")

        task_args.append((cell.template_params, stimulus, section_name, segment, cvode))

    with IsolatedProcess(processes=n_processes) as pool:
        # Map expects a function and a list of argument tuples
        results = pool.starmap(run_stimulus, task_args)

    # Associate each result with a key
    for amplitude, result in zip(amplitudes, results):
        key = f"{stimulus_name}_{amplitude}"
        res[key] = result

    return res

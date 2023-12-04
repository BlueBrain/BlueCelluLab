# Copyright 2012-2023 Blue Brain Project / EPFL

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Static tools for BluecellulabError."""


from __future__ import annotations
import io
import json
import math
import multiprocessing
import multiprocessing.pool
import os
from pathlib import Path
import sys
from typing import Any, Optional, Tuple
import logging

import numpy as np

import bluecellulab
from bluecellulab.circuit.circuit_access import EmodelProperties
from bluecellulab.exceptions import UnsteadyCellError

logger = logging.getLogger(__name__)

VERBOSE_LEVEL = 0
ENV_VERBOSE_LEVEL: Optional[str] = None


def set_verbose(level: int = 1) -> None:
    """Set the verbose level of BluecellulabError.

    Parameters
    ----------
    level :
            Verbose level, the higher the more verbosity.
            Level 0 means 'completely quiet', except if some very serious
            errors or warnings are encountered.
    """
    bluecellulab.VERBOSE_LEVEL = level

    if level <= 0:
        logging.getLogger('bluecellulab').setLevel(logging.CRITICAL)
    elif level == 1:
        logging.getLogger('bluecellulab').setLevel(logging.ERROR)
    elif level == 2:
        logging.getLogger('bluecellulab').setLevel(logging.WARNING)
    elif level > 2 and level <= 5:
        logging.getLogger('bluecellulab').setLevel(logging.INFO)
    else:
        logging.getLogger('bluecellulab').setLevel(logging.DEBUG)


def set_verbose_from_env() -> None:
    """Get verbose level from environment."""
    bluecellulab.ENV_VERBOSE_LEVEL = os.environ.get('BLUECELLULAB_VERBOSE_LEVEL')

    if bluecellulab.ENV_VERBOSE_LEVEL is not None:
        set_verbose(int(bluecellulab.ENV_VERBOSE_LEVEL))


set_verbose_from_env()


def calculate_input_resistance(
    template_path: str | Path,
    morphology_path: str | Path,
    template_format: str,
    emodel_properties: EmodelProperties | None,
    current_delta: float = 0.01,
) -> float:
    """Calculate the input resistance at rest of the cell."""
    rest_voltage = calculate_SS_voltage(
        template_path, morphology_path, template_format, emodel_properties, 0.0
    )
    step_voltage = calculate_SS_voltage(
        template_path,
        morphology_path,
        template_format,
        emodel_properties,
        current_delta,
    )

    voltage_delta = step_voltage - rest_voltage

    return voltage_delta / current_delta


def calculate_SS_voltage(
    template_path: str | Path,
    morphology_path: str | Path,
    template_format: str,
    emodel_properties: EmodelProperties | None,
    step_level: float,
) -> float:
    """Calculate the steady state voltage at a certain current step.

    The use of Pool is safe here since it will just run a single task.
    """
    pool = multiprocessing.Pool(processes=1)
    SS_voltage = pool.apply(
        calculate_SS_voltage_subprocess,
        [
            template_path,
            morphology_path,
            template_format,
            emodel_properties,
            step_level,
        ],
    )
    pool.terminate()
    return SS_voltage


def calculate_SS_voltage_subprocess(
    template_path: str | Path,
    morphology_path: str | Path,
    template_format: str,
    emodel_properties: EmodelProperties | None,
    step_level: float,
    check_for_spiking=False,
    spike_threshold=-20.0,
) -> float:
    """Subprocess wrapper of calculate_SS_voltage.

    This code should be run in a separate process. If check_for_spiking
    is True, this function will return None if the cell spikes from
    100ms to the end of the simulation indicating no steady state was
    reached.
    """
    cell = bluecellulab.Cell(
        template_path=template_path,
        morphology_path=morphology_path,
        template_format=template_format,
        emodel_properties=emodel_properties,
    )
    cell.add_ramp(500, 5000, step_level, step_level)
    simulation = bluecellulab.Simulation()
    simulation.run(1000, cvode=template_accepts_cvode(template_path))
    time = cell.get_time()
    voltage = cell.get_soma_voltage()
    SS_voltage = np.mean(voltage[np.where((time < 1000) & (time > 800))])
    cell.delete()

    if check_for_spiking:
        # check for voltage crossings
        if len(np.nonzero(voltage[np.where(time > 100.0)] > spike_threshold)[0]) > 0:
            raise UnsteadyCellError(
                "Cell spikes from 100ms to the end of the simulation."
            )

    return SS_voltage


def holding_current_subprocess(v_hold, enable_ttx, cell_kwargs):
    """Subprocess wrapper of holding_current."""
    cell = bluecellulab.Cell(**cell_kwargs)

    if enable_ttx:
        cell.enable_ttx()

    vclamp = bluecellulab.neuron.h.SEClamp(0.5, sec=cell.soma)
    vclamp.rs = 0.01
    vclamp.dur1 = 2000
    vclamp.amp1 = v_hold

    simulation = bluecellulab.Simulation()
    simulation.run(1000, cvode=False)

    i_hold = vclamp.i
    v_control = vclamp.vc

    cell.delete()

    return i_hold, v_control


def holding_current(
    v_hold: float,
    cell_id: int | tuple[str, int],
    circuit_path: str | Path,
    enable_ttx=False,
) -> Tuple[float, float]:
    """Calculate the holding current necessary for a given holding voltage."""
    cell_id = bluecellulab.circuit.node_id.create_cell_id(cell_id)
    ssim = bluecellulab.SSim(circuit_path)

    cell_kwargs = ssim.fetch_cell_kwargs(cell_id)

    # using a pool with NEURON here is safe since it'll run one task only
    pool = multiprocessing.Pool(processes=1)
    i_hold, v_control = pool.apply(
        holding_current_subprocess, [v_hold, enable_ttx, cell_kwargs]
    )
    pool.terminate()

    return i_hold, v_control


def template_accepts_cvode(template_name: str | Path) -> bool:
    """Return True if template_name can be run with cvode."""
    with open(template_name, "r") as template_file:
        template_content = template_file.read()
    if "StochKv" in template_content:
        accepts_cvode = False
    else:
        accepts_cvode = True
    return accepts_cvode


def search_hyp_current(
    template_path: str | Path,
    morphology_path: str | Path,
    template_format: str,
    emodel_properties: Optional[EmodelProperties],
    target_voltage: float,
    min_current: float,
    max_current: float,
) -> float:
    """Search current necessary to bring cell to -85 mV."""
    med_current = min_current + abs(min_current - max_current) / 2
    new_target_voltage = calculate_SS_voltage(
        template_path,
        morphology_path,
        template_format,
        emodel_properties,
        med_current,
    )
    logger.info("Detected voltage: %f" % new_target_voltage)
    if abs(new_target_voltage - target_voltage) < 0.5:
        return med_current
    elif new_target_voltage > target_voltage:
        return search_hyp_current(
            template_path=template_path,
            morphology_path=morphology_path,
            template_format=template_format,
            emodel_properties=emodel_properties,
            target_voltage=target_voltage,
            min_current=min_current,
            max_current=med_current,
        )
    else:  # new_target_voltage < target_voltage:
        return search_hyp_current(
            template_path=template_path,
            morphology_path=morphology_path,
            template_format=template_format,
            emodel_properties=emodel_properties,
            target_voltage=target_voltage,
            min_current=med_current,
            max_current=max_current,
        )


def detect_hyp_current(
    template_path: str | Path,
    morphology_path: str | Path,
    template_format: str,
    emodel_properties: EmodelProperties | None,
    target_voltage: float,
) -> float:
    """Search current necessary to bring cell to -85 mV.

    Compared to using NEURON's SEClamp object, the binary search better
    replicates what experimentalists use
    """
    return search_hyp_current(
        template_path=template_path,
        morphology_path=morphology_path,
        template_format=template_format,
        emodel_properties=emodel_properties,
        target_voltage=target_voltage,
        min_current=-1.0,
        max_current=0.0,
    )


def detect_spike_step(
    template_path: str | Path,
    morphology_path: str | Path,
    template_format: str,
    emodel_properties: EmodelProperties | None,
    hyp_level: float,
    inj_start: float,
    inj_stop: float,
    step_level: float,
) -> bool:
    """Detect if there is a spike at a certain step level."""
    # Here it is safe to use a pool with NEURON since it'll run one task only
    pool = multiprocessing.Pool(processes=1)
    spike_detected = pool.apply(
        detect_spike_step_subprocess,
        [
            template_path,
            morphology_path,
            template_format,
            emodel_properties,
            hyp_level,
            inj_start,
            inj_stop,
            step_level,
        ],
    )
    pool.terminate()
    return spike_detected


def detect_spike_step_subprocess(
    template_path: str | Path,
    morphology_path: str | Path,
    template_format: str,
    emodel_properties: EmodelProperties | None,
    hyp_level: float,
    inj_start: float,
    inj_stop: float,
    step_level: float
) -> bool:
    """Detect if there is a spike at a certain step level."""
    cell = bluecellulab.Cell(
        template_path=template_path,
        morphology_path=morphology_path,
        template_format=template_format,
        emodel_properties=emodel_properties)
    cell.add_ramp(0, 5000, hyp_level, hyp_level)
    cell.add_ramp(inj_start, inj_stop, step_level, step_level)
    simulation = bluecellulab.Simulation()
    simulation.run(int(inj_stop), cvode=template_accepts_cvode(template_path))

    time = cell.get_time()
    voltage = cell.get_soma_voltage()
    time_step = time[np.where((time > inj_start) & (time < inj_stop))]
    voltage_step = voltage[np.where((time_step > inj_start) & (time_step < inj_stop))]
    spike_detected = detect_spike(voltage_step)

    cell.delete()

    return spike_detected


def detect_spike(voltage: np.ndarray) -> bool:
    """Detect if there is a spike in the voltage trace."""
    if len(voltage) == 0:
        return False
    else:
        return bool(np.max(voltage) > -20)  # bool not np.bool_


def search_threshold_current(template_name, morphology_name, hyp_level,
                             inj_start, inj_stop, min_current, max_current):
    """Search current necessary to reach threshold."""
    med_current = min_current + abs(min_current - max_current) / 2
    logger.info("Med current %d" % med_current)

    spike_detected = detect_spike_step(
        template_name, morphology_name, hyp_level, inj_start, inj_stop,
        med_current)
    logger.info("Spike threshold detection at: %f nA" % med_current)

    if abs(max_current - min_current) < .01:
        return max_current
    elif spike_detected:
        return search_threshold_current(template_name, morphology_name,
                                        hyp_level, inj_start, inj_stop,
                                        min_current, med_current)
    else:
        return search_threshold_current(template_name, morphology_name,
                                        hyp_level, inj_start, inj_stop,
                                        med_current, max_current)


def detect_threshold_current(template_name, morphology_name, hyp_level,
                             inj_start, inj_stop):
    """Search current necessary to reach threshold."""
    return search_threshold_current(template_name, morphology_name,
                                    hyp_level, inj_start, inj_stop, 0.0, 1.0)


def calculate_SS_voltage_replay(blueconfig, gid, step_level, start_time=None,
                                stop_time=None, ignore_timerange=False,
                                timeout=600):
    """Calculate the steady state voltage at a certain current step."""
    pool = multiprocessing.Pool(processes=1)
    # print "Calculate_SS_voltage_replay %f" % step_level
    result = pool.apply_async(calculate_SS_voltage_replay_subprocess,
                              [blueconfig, gid, step_level, start_time,
                               stop_time, ignore_timerange])

    try:
        output = result.get(timeout=timeout)
        # (SS_voltage, (time, voltage)) = result.get(timeout=timeout)
    except multiprocessing.TimeoutError:
        output = (float('nan'), (None, None))

    # (SS_voltage, voltage) = calculate_SS_voltage_replay_subprocess(
    # blueconfig, gid, step_level)
    pool.terminate()
    return output


def calculate_SS_voltage_replay_subprocess(blueconfig, gid, step_level,
                                           start_time=None, stop_time=None,
                                           ignore_timerange=False):
    """Subprocess wrapper of calculate_SS_voltage."""
    process_name = multiprocessing.current_process().name
    ssim = bluecellulab.SSim(blueconfig)
    if ignore_timerange:
        tstart = 0
        tstop = int(ssim.circuit_access.config.duration)
    else:
        tstart = start_time
        tstop = stop_time
    # print "%s: Calculating SS voltage of step level %f nA" %
    # (process_name, step_level)
    # print "Calculate_SS_voltage_replay_subprocess instantiating gid ..."
    ssim.instantiate_gids(
        [gid], add_synapses=True, add_minis=True, add_stimuli=True, add_replay=True)
    # print "Calculate_SS_voltage_replay_subprocess instantiating gid done"

    ssim.cells[gid].add_ramp(0, tstop, step_level, step_level)
    ssim.run(t_stop=tstop)
    time = ssim.get_time_trace()
    voltage = ssim.get_voltage_trace(gid)
    SS_voltage = np.mean(voltage[np.where(
        (time < tstop) & (time > tstart))])
    logger.info("%s: Calculated SS voltage for gid %d "
                "with step level %f nA: %s mV" %
                (process_name, gid, step_level, SS_voltage))
    # print "Calculate_SS_voltage_replay_subprocess voltage:%f" % SS_voltage

    return (SS_voltage, (time, voltage))


class NoDaemonProcess(multiprocessing.Process):
    """Class that represents a non-daemon process."""

    def _get_daemon(self):
        """Get daemon flag."""
        return False

    def _set_daemon(self, value):
        """Set daemon flag."""
        pass
    daemon = property(_get_daemon, _set_daemon)  # type:ignore


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.


class NestedPool(multiprocessing.pool.Pool):
    """Class that represents a MultiProcessing nested pool."""
    Process = NoDaemonProcess


def search_hyp_current_replay(blueconfig, gid, target_voltage=-80,
                              min_current=-1.0, max_current=0.0,
                              precision=.5,
                              max_nestlevel=10,
                              nestlevel=1,
                              start_time=500, stop_time=2000,
                              return_fullrange=True,
                              timeout=600):
    """Search current to bring cell to target_voltage in a network replay."""
    process_name = multiprocessing.current_process().name

    if nestlevel > max_nestlevel:
        return (float('nan'), (None, None))
    elif nestlevel == 1:
        logger.info("%s: Searching for current to bring gid %d to %f mV" %
                    (process_name, gid, target_voltage))
    med_current = min_current + abs(min_current - max_current) / 2
    (new_target_voltage, (time, voltage)) = \
        calculate_SS_voltage_replay(blueconfig, gid, med_current,
                                    start_time=start_time,
                                    stop_time=stop_time, timeout=timeout)
    if math.isnan(new_target_voltage):
        return (float('nan'), (None, None))
    if abs(new_target_voltage - target_voltage) < precision:
        if return_fullrange:
            # We're calculating the full voltage range,
            # just reusing calculate_SS_voltage_replay for this
            # Variable names that start with full_ point to values that are
            # related to the full voltage range
            (full_SS_voltage, (full_time, full_voltage)) = \
                calculate_SS_voltage_replay(
                    blueconfig, gid, med_current,
                    start_time=start_time, timeout=timeout,
                    ignore_timerange=True)
            if math.isnan(full_SS_voltage):
                return (float('nan'), (None, None))
            return (med_current, (full_time, full_voltage))
        else:
            return (med_current, (time, voltage))
    elif new_target_voltage > target_voltage:
        return search_hyp_current_replay(blueconfig, gid, target_voltage,
                                         min_current=min_current,
                                         max_current=med_current,
                                         precision=precision,
                                         nestlevel=nestlevel + 1,
                                         start_time=start_time,
                                         stop_time=stop_time,
                                         max_nestlevel=max_nestlevel,
                                         return_fullrange=return_fullrange)
    elif new_target_voltage < target_voltage:
        return search_hyp_current_replay(blueconfig, gid, target_voltage,
                                         min_current=med_current,
                                         max_current=max_current,
                                         precision=precision,
                                         nestlevel=nestlevel + 1,
                                         start_time=start_time,
                                         stop_time=stop_time,
                                         max_nestlevel=max_nestlevel,
                                         return_fullrange=return_fullrange)


class search_hyp_function:
    """Function object."""

    def __init__(self, blueconfig, **kwargs):
        self.blueconfig = blueconfig
        self.kwargs = kwargs

    def __call__(self, gid):
        return search_hyp_current_replay(self.blueconfig, gid, **self.kwargs)


class search_hyp_function_gid:
    """Function object, return a tuple (gid, results)"""

    def __init__(self, blueconfig, **kwargs):
        self.blueconfig = blueconfig
        self.kwargs = kwargs

    def __call__(self, gid):
        return (
            gid,
            search_hyp_current_replay(
                self.blueconfig,
                gid,
                **self.kwargs))


def search_hyp_current_replay_gidlist(blueconfig, gid_list, **kwargs):
    """Search, using bisection, for the current necessary to bring a cell to
    target_voltage in a network replay for a list of gids. This function will
    use multiprocessing to parallelize the task, running one gid per available
    core.

    Parameters
    ----------
    blueconfig : string
                 Path to simulation BlueConfig
    gid_list : list of integers
               List of the gids
    target_voltage : float
                     Voltage you want to bring to cell to
    min_current, max_current : float
                               The algorithm will search in
                               ]min_current, max_current[
    precision: float
               The algorithm stops when
               abs(calculated_voltage - target_voltage) < precision
    max_nestlevel : integer
                    The maximum number of nested levels the algorithm explores
    start_time, stop_time : float
                            The time range for which the voltage is simulated
                            and average for comparison against target_voltage
    return_fullrange: boolean
                      Defaults to True. Set to False if you don't want to
                      return the voltage in full time range of the large
                      simulation, but rather the time between
                      start_time, stop_time

    Returns
    -------
    result: dictionary
            A dictionary where the keys are gids, and the values tuples of the
            form (detected_level, time_voltage).
            time_voltage is a tuple of the time and voltage trace at the
            current injection level (=detected_level) that matches the target
            target_voltage within user specified precision.

            If the algorithm reaches max_nestlevel+1 iterations without
            converging to the requested precision, (nan, None) is returned
            for that gid.
    """

    pool = NestedPool(multiprocessing.cpu_count())
    results = pool.map(search_hyp_function(blueconfig, **kwargs), gid_list)
    pool.terminate()

    currentlevels_timevoltagetraces = {}
    for gid, result in zip(gid_list, results):
        currentlevels_timevoltagetraces[gid] = result

    return currentlevels_timevoltagetraces


def search_hyp_current_replay_imap(blueconfig, gid_list, timeout=600,
                                   cpu_count=None, **kwargs):
    """Same functionality as search_hyp_current_gidlist(), except that this
    function returns an unordered generator.

    Loop over this generator will return the unordered results one by
    one. The results returned will be of the form (gid, (current_step,
    (time, voltage))) When there are results that take more that
    'timeout' time to retrieve, these results will be (None, None). The
    user should stop iterating the generating after receiving this
    (None, None) result. In this case also probably a broke pipe error
    from some of the parallel process will be shown on the stdout, these
    can be ignored.
    """
    if cpu_count is None:
        pool = NestedPool(multiprocessing.cpu_count())
    else:
        pool = NestedPool(cpu_count)

    results = pool.imap_unordered(search_hyp_function_gid(
        blueconfig, **kwargs), gid_list)
    for _ in gid_list:
        try:
            (gid, result) = results.next(timeout=timeout)
            yield (gid, result)
        except multiprocessing.TimeoutError:
            pool.terminate()
            yield (None, None)
    pool.terminate()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class get_stdout(list):
    def __enter__(self):
        self.orig_stdout = sys.stdout
        sys.stdout = self.stringio = io.StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self.stringio.getvalue().splitlines())
        del self.stringio
        sys.stdout = self.orig_stdout


def check_empty_topology() -> bool:
    """Return true if NEURON simulator topology command is empty."""
    with get_stdout() as stdout:
        bluecellulab.neuron.h.topology()

    return stdout == ['', '']


class Singleton(type):
    """Singleton metaclass implementation.

    Source: https://stackoverflow.com/a/6798042/1935611
    """
    _instances: dict[Any, Any] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs)
        else:  # to run init on the same object
            cls._instances[cls].__init__(*args, **kwargs)
        return cls._instances[cls]

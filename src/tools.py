'''
Static tools for bglibpy
'''

import sys
import inspect
from bglibpy.importer import neuron
import multiprocessing
import multiprocessing.pool
import bglibpy
#from bglibpy.importer import neuron
import numpy
import warnings
import math

BLUECONFIG_KEYWORDS = ['Run', 'Stimulus', 'StimulusInject', 'Report', 'Connection']
VERBOSE_LEVEL = 0

def get_gids_of_mtypes(mtypes=['L5_UTPC','L6_TPC_L4']) :
    """
    Helper function that, provided a BlueConfig, returns all the GIDs \
    associated with a specified M-type. (For instance, when you only want \
    to insert synapses of a specific pathway)

    Parameters
    ----------
    mtypes : list
        List of M-types (each as a string)

    Returns
    -------
    gids : list
        List of all GIDs associated with one of the specified M-types

    """
    a = 1
    b = a +1
    return b

def deprecated(func):
    """A decorator that shows a warning message when a deprecated function is used"""
    def rep_func(*args, **kwargs):
        """Replacement function"""
        warnings.warn("Call to deprecated function {%s}." % func.__name__,
                      category=DeprecationWarning)
        return func(*args, **kwargs)
    rep_func.__name__ = func.__name__
    rep_func.__doc__ = func.__doc__
    rep_func.__dict__.update(func.__dict__)
    return rep_func

def printv(message, verbose_level):
    """Print the message depending on the verbose level"""
    if verbose_level <= bglibpy.VERBOSE_LEVEL:
        print message

def printv_err(message, verbose_level):
    """Print the message depending on the verbose level"""
    if verbose_level <= bglibpy.VERBOSE_LEVEL:
        print >> sys.stderr,  message

def _me():
    '''Used for debgugging. Reads the stack and provides info about which
    function called  '''
    print 'Call -> from %s::%s' % (inspect.stack()[1][1], inspect.stack()[1][3])


def load_nrnmechanisms(libnrnmech_location):
    """Load another shared library with neuron mechanisms"""
    neuron.h.nrn_load_dll(libnrnmech_location)


def parse_complete_BlueConfig(fName):
    """ Simplistic parser of the BlueConfig file """
    bc = open(fName, 'r')
    uber_hash = {}  # collections.OrderedDict
    for keyword in BLUECONFIG_KEYWORDS:
        uber_hash[keyword] = {}
    line = bc.next()

    block_number = 0

    while(line != ''):
        stripped_line = line.strip()
        if stripped_line.startswith('#'):
            ''' continue to next line '''
            line = bc.next()
        elif stripped_line == '':
            # print 'found empty line'
            try:
                line = bc.next()
            except StopIteration:
                # print 'I think i am at the end of the file'
                break
        elif stripped_line.split()[0].strip() in BLUECONFIG_KEYWORDS:
            key = stripped_line.split()[0].strip()
            value = stripped_line.split()[1].strip()
            parsed_dict = _parse_block_statement(bc)
            parsed_dict['block_number'] = block_number
            uber_hash[key][value] = parsed_dict
            block_number = block_number + 1
            line = bc.next()
        else:
            line = bc.next()
    return uber_hash


def _parse_block_statement(file_object):
    ''' parse the content of the blocks in BlueConfig'''
    file_object.next()  # skip the opening "}"
    line = file_object.next().strip()
    ret_dict = {}
    while(not line.startswith('}')):
        if(len(line) == 0 or line.startswith('#')):
            line = file_object.next().strip()
        else:
            key = line.split(' ')[0].strip()
            values = line.split(' ')[1:]
            for value in values:
                if(value == ''):
                    pass
                else:
                    ret_dict[key] = value
            line = file_object.next().strip()
    return ret_dict


def calculate_inputresistance(template_name, morphology_name, current_delta=0.01):
    """Calculate the input resistance at rest of the cell"""
    rest_voltage = calculate_SS_voltage(template_name, morphology_name, 0.0)
    step_voltage = calculate_SS_voltage(template_name, morphology_name, current_delta)

    voltage_delta = step_voltage - rest_voltage

    return voltage_delta / current_delta


def calculate_SS_voltage(template_name, morphology_name, step_level):
    """Calculate the steady state voltage at a certain current step"""
    pool = multiprocessing.Pool(processes=1)
    SS_voltage = pool.apply(calculate_SS_voltage_subprocess, [template_name, morphology_name, step_level])
    pool.terminate()
    return SS_voltage


def calculate_SS_voltage_subprocess(template_name, morphology_name, step_level, check_for_spiking=False, spike_threshold=-20.0):
    """Subprocess wrapper of calculate_SS_voltage

    if check_for_spiking is True,
    this function will return None if the cell spikes from from 100ms to the end of the simulation
    indicating no steady state was reached.

    """
    cell = bglibpy.Cell(template_name, morphology_name)
    cell.add_ramp(500, 5000, step_level, step_level, dt=1.0)
    simulation = bglibpy.Simulation()
    simulation.run(1000, cvode=template_accepts_cvode(template_name))
    time = cell.get_time()
    voltage = cell.get_soma_voltage()
    SS_voltage = numpy.mean(voltage[numpy.where((time < 1000) & (time > 800))])
    cell.delete()

    if check_for_spiking:
        # check for voltage crossings
        if len(numpy.nonzero(voltage[numpy.where(time > 100.0)]>spike_threshold)[0])>0:
            return None

    return SS_voltage



def template_accepts_cvode(template_name):
    """Return True if template_name can be run with cvode"""
    with open(template_name, "r") as template_file:
        template_content = template_file.read()
    if "StochKv" in template_content:
        accepts_cvode = False
    else:
        accepts_cvode = True
    return accepts_cvode


def search_hyp_current(template_name, morphology_name, target_voltage, min_current, max_current):
    """Search current necessary to bring cell to -85 mV"""
    med_current = min_current + abs(min_current - max_current) / 2
    new_target_voltage = calculate_SS_voltage(template_name, morphology_name, med_current)
    print "Detected voltage: ", new_target_voltage
    if abs(new_target_voltage - target_voltage) < .5:
        return med_current
    elif new_target_voltage > target_voltage:
        return search_hyp_current(template_name, morphology_name, target_voltage, min_current, med_current)
    elif new_target_voltage < target_voltage:
        return search_hyp_current(template_name, morphology_name, target_voltage, med_current, max_current)


def detect_hyp_current(template_name, morphology_name, target_voltage):
    """Search current necessary to bring cell to -85 mV"""
    return search_hyp_current(template_name, morphology_name, target_voltage, -1.0, 0.0)


def detect_spike_step(template_name, morphology_name, hyp_level, inj_start, inj_stop, step_level):
    """Detect if there is a spike at a certain step level"""
    pool = multiprocessing.Pool(processes=1)
    spike_detected = pool.apply(detect_spike_step_subprocess, [template_name, morphology_name, hyp_level, inj_start, inj_stop, step_level])
    pool.terminate()
    return spike_detected


def detect_spike_step_subprocess(template_name, morphology_name, hyp_level, inj_start, inj_stop, step_level):
    """Detect if there is a spike at a certain step level"""
    cell = bglibpy.Cell(template_name, morphology_name)
    cell.add_ramp(0, 5000, hyp_level, hyp_level, dt=1.0)
    cell.add_ramp(inj_start, inj_stop, step_level, step_level, dt=1.0)
    simulation = bglibpy.Simulation()
    simulation.run(int(inj_stop), cvode=template_accepts_cvode(template_name))

    time = cell.get_time()
    voltage = cell.get_soma_voltage()
    time_step = time[numpy.where((time > inj_start) & (time < inj_stop))]
    voltage_step = voltage[numpy.where((time_step > inj_start) & (time_step < inj_stop))]
    spike_detected = detect_spike(voltage_step)

    cell.delete()

    return spike_detected

def detect_spike(voltage):
    """Detect if there is a spike in the voltage trace"""
    return numpy.max(voltage) > -20

def search_threshold_current(template_name, morphology_name, hyp_level, inj_start, inj_stop, min_current, max_current):
    """Search current necessary to reach threshold"""
    med_current = min_current + abs(min_current - max_current) / 2
    print "Med current %d" % med_current

    spike_detected = detect_spike_step(template_name, morphology_name, hyp_level, inj_start, inj_stop, med_current)
    print "Spike threshold detection at: ", med_current, "nA", spike_detected

    if abs(max_current - min_current) < .01:
        return max_current
    elif spike_detected:
        return search_threshold_current(template_name, morphology_name, hyp_level, inj_start, inj_stop, min_current, med_current)
    elif not spike_detected:
        return search_threshold_current(template_name, morphology_name, hyp_level, inj_start, inj_stop, med_current, max_current)

def detect_threshold_current(template_name, morphology_name, hyp_level, inj_start, inj_stop):
    """Search current necessary to reach threshold"""
    return search_threshold_current(template_name, morphology_name, hyp_level, inj_start, inj_stop, 0.0, 1.0)

def calculate_SS_voltage_replay(blueconfig, gid, step_level, start_time=None, stop_time=None, ignore_timerange=False, timeout=600):
    """Calculate the steady state voltage at a certain current step"""
    pool = multiprocessing.Pool(processes=1)
    #print "Calculate_SS_voltage_replay %f" % step_level
    result = pool.apply_async(calculate_SS_voltage_replay_subprocess, [blueconfig, gid, step_level, start_time, stop_time, ignore_timerange])

    try:
        output = result.get(timeout=timeout)
        #(SS_voltage, (time, voltage)) = result.get(timeout=timeout)
    except multiprocessing.TimeoutError:
        output = (float('nan'), (None, None))

    #(SS_voltage, voltage) = calculate_SS_voltage_replay_subprocess(blueconfig, gid, step_level)
    pool.terminate()
    return output

def calculate_SS_voltage_replay_subprocess(blueconfig, gid, step_level, start_time=None, stop_time=None, ignore_timerange=False):
    """Subprocess wrapper of calculate_SS_voltage"""
    process_name = multiprocessing.current_process().name
    ssim = bglibpy.SSim(blueconfig)
    if ignore_timerange:
        tstart = 0
        tstop = int(ssim.bc.entry_map['Default'].CONTENTS.Duration)
    else:
        tstart = start_time
        tstop = stop_time
    #print "%s: Calculating SS voltage of step level %f nA" % (process_name, step_level)
    #print "Calculate_SS_voltage_replay_subprocess instantiating gid ..."
    ssim.instantiate_gids([gid], synapse_detail=2, add_stimuli=True, add_replay=True)
    #print "Calculate_SS_voltage_replay_subprocess instantiating gid done"

    ssim.cells[gid].add_ramp(0, tstop, step_level, step_level)
    ssim.run(t_stop=tstop)
    time = ssim.get_time()
    voltage = ssim.get_voltage_traces()[gid]
    SS_voltage = numpy.mean(voltage[numpy.where((time < tstop) & (time > tstart))])
    printv("%s: Calculated SS voltage for gid %d with step level %f nA: %s mV" % (process_name, gid, step_level, SS_voltage), 1)

    #print "Calculate_SS_voltage_replay_subprocess voltage:%f" % SS_voltage

    return (SS_voltage, (time, voltage))

class NoDaemonProcess(multiprocessing.Process):
    """Class that represents a non-daemon process"""

    # pylint: disable=R0201
    def _get_daemon(self):
        """Get daemon flag"""
        return False
    def _set_daemon(self, value):
        """Set daemon flag"""
        pass
    daemon = property(_get_daemon, _set_daemon)

# pylint: disable=W0223, R0911

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NestedPool(multiprocessing.pool.Pool):
    """Class that represents a MultiProcessing nested pool"""
    Process = NoDaemonProcess


def search_hyp_current_replay(blueconfig, gid, target_voltage=-80,
        min_current=-1.0, max_current=0.0,
        precision=.5,
        max_nestlevel=10,
        nestlevel=1,
        start_time=500, stop_time=2000,
        return_fullrange=True,
        timeout=600):
    """Search current necessary to bring cell to target_voltage in a network replay"""
    process_name = multiprocessing.current_process().name

    if nestlevel > max_nestlevel:
        return (float('nan'), (None, None))
    elif nestlevel == 1:
        printv("%s: Searching for current to bring gid %d to %f mV" % (process_name, gid, target_voltage), 1)
    med_current = min_current + abs(min_current - max_current) / 2
    (new_target_voltage, (time, voltage)) = calculate_SS_voltage_replay(blueconfig, gid, med_current, start_time=start_time, stop_time=stop_time, timeout=timeout)
    if math.isnan(new_target_voltage):
        return (float('nan'), (None, None))
    if abs(new_target_voltage - target_voltage) < precision:
        if return_fullrange:
            # We're calculating the full voltage range, just reusing calculate_SS_voltage_replay for this
            # Variable names that start with full_ point to values that are related to the full voltage range
            (full_SS_voltage, (full_time, full_voltage)) = calculate_SS_voltage_replay(blueconfig, gid, med_current, start_time=start_time, timeout=timeout, ignore_timerange=True)
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
                nestlevel=nestlevel+1,
                start_time=start_time, stop_time=stop_time,
                max_nestlevel=max_nestlevel,
                return_fullrange=return_fullrange)
    elif new_target_voltage < target_voltage:
        return search_hyp_current_replay(blueconfig, gid, target_voltage,
                min_current=med_current,
                max_current=max_current,
                precision=precision,
                nestlevel=nestlevel+1,
                start_time=start_time, stop_time=stop_time,
                max_nestlevel=max_nestlevel,
                return_fullrange=return_fullrange)

class search_hyp_function(object):
    """Function object"""
    def __init__(self, blueconfig, **kwargs):
        self.blueconfig = blueconfig
        self.kwargs = kwargs
    def __call__(self, gid):
        return search_hyp_current_replay(self.blueconfig, gid, **self.kwargs)

class search_hyp_function_gid(object):
    """Function object, return a tuple (gid, results)"""
    def __init__(self, blueconfig, **kwargs):
        self.blueconfig = blueconfig
        self.kwargs = kwargs
    def __call__(self, gid):
        return (gid, search_hyp_current_replay(self.blueconfig, gid, **self.kwargs))

def search_hyp_current_replay_gidlist(blueconfig, gid_list, **kwargs):
    """
    Search, using bisection, for the current necessary to bring a cell to target_voltage in a network replay for a list of gids.
    This function will use multiprocessing to parallelize the task, running one gid per available core.

    Parameters
    ----------
    blueconfig : Simulation BlueConfig
    gid_list : list
        of gids to process
    target_voltage: voltage you want to bring to cell to
    min_current, max_current: The algorithm will search in ]min_current, max_current[
    precision: algorithm stops when abs(calculated_voltage - target_voltage) < precision
    max_nestlevel = the maximum number of nested levels the algorithm explores
    start_time, stop_time: the time range for which the voltage is simulated and average for comparison against target_voltage
    return_fullrange: Defaults to True.  Set to False if you don't want to return the voltage in full time range of the large simulation, but rather the time between start_time, stop_time

    Returns
    -------
    A dictionary where the keys are gids, and the values tuples of the form (detected_level, time_voltage).
    time_voltage is a tuple of the time and voltage trace at the current injection level (=detected_level) that matches the target target_voltage within user specified precision.

    If the algorithm reaches max_nestlevel+1 iterations without converging to the requested precision, (nan, None) is returned for that gid.
    """

    pool = NestedPool(multiprocessing.cpu_count())
    results = pool.map(search_hyp_function(blueconfig, **kwargs), gid_list)
    pool.terminate()

    currentlevels_timevoltagetraces = {}
    for gid, result in zip(gid_list, results):
        currentlevels_timevoltagetraces[gid] = result

    return currentlevels_timevoltagetraces

def search_hyp_current_replay_imap(blueconfig, gid_list, timeout=600, cpu_count = None, **kwargs):
    """
    Same functionality as search_hyp_current_gidlist(), except that this function returns an unordered generator.
    Loop over this generator will return the unordered results one by one.
    The results returned will be of the form (gid, (current_step, (time, voltage)))
    When there are results that take more that 'timeout' time to retrieve, these results will be (None, None). The
    user should stop iterating the generating after receiving this (None, None) result. In this case also probably
    a broke pipe error from some of the parallel process will be shown on the stdout, these can be ignored.
    """
    if cpu_count == None:
        pool = NestedPool(multiprocessing.cpu_count())
    else:
        pool = NestedPool(cpu_count)
        
    results = pool.imap_unordered(search_hyp_function_gid(blueconfig, **kwargs), gid_list)
    for _ in gid_list:
        try:
            (gid, result) = results.next(timeout=timeout)
            yield (gid, result)
        except multiprocessing.TimeoutError:
            pool.terminate()
            yield (None, None)
    pool.terminate()

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

BLUECONFIG_KEYWORDS = ['Run', 'Stimulus', 'StimulusInject', 'Report', 'Connection']
VERBOSE_LEVEL = 0

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


def calculate_SS_voltage_subprocess(template_name, morphology_name, step_level):
    """Subprocess wrapper of calculate_SS_voltage"""
    cell = bglibpy.Cell(template_name, morphology_name)
    cell.addRamp(500, 5000, step_level, step_level, dt=1.0)
    simulation = bglibpy.Simulation()
    simulation.run(1000, cvode=template_accepts_cvode(template_name))
    time = cell.get_time()
    voltage = cell.get_soma_voltage()
    SS_voltage = numpy.mean(voltage[numpy.where((time < 1000) & (time > 800))])
    cell.delete()

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


def search_hyp_current(template_name, morphology_name, hyp_voltage, min_current, max_current):
    """Search current necessary to bring cell to -85 mV"""
    med_current = min_current + abs(min_current - max_current) / 2
    new_hyp_voltage = calculate_SS_voltage(template_name, morphology_name, med_current)
    print "Detected voltage: ", new_hyp_voltage
    if abs(new_hyp_voltage - hyp_voltage) < .5:
        return med_current
    elif new_hyp_voltage > hyp_voltage:
        return search_hyp_current(template_name, morphology_name, hyp_voltage, min_current, med_current)
    elif new_hyp_voltage < hyp_voltage:
        return search_hyp_current(template_name, morphology_name, hyp_voltage, med_current, max_current)


def detect_hyp_current(template_name, morphology_name, hyp_voltage):
    """Search current necessary to bring cell to -85 mV"""
    return search_hyp_current(template_name, morphology_name, hyp_voltage, -1.0, 0.0)


def detect_spike_step(template_name, morphology_name, hyp_level, inj_start, inj_stop, step_level):
    """Detect if there is a spike at a certain step level"""
    pool = multiprocessing.Pool(processes=1)
    spike_detected = pool.apply(detect_spike_step_subprocess, [template_name, morphology_name, hyp_level, inj_start, inj_stop, step_level])
    pool.terminate()
    return spike_detected


def detect_spike_step_subprocess(template_name, morphology_name, hyp_level, inj_start, inj_stop, step_level):
    """Detect if there is a spike at a certain step level"""
    cell = bglibpy.Cell(template_name, morphology_name)
    cell.addRamp(0, 5000, hyp_level, hyp_level, dt=1.0)
    cell.addRamp(inj_start, inj_stop, step_level, step_level, dt=1.0)
    simulation = bglibpy.Simulation()
    simulation.run(int(inj_stop), cvode=template_accepts_cvode(template_name))

    time = cell.getTime()
    voltage = cell.getSomaVoltage()
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

def calculate_SS_voltage_replay(blueconfig, gid, step_level, start_time=None, stop_time=None):
    """Calculate the steady state voltage at a certain current step"""
    pool = multiprocessing.Pool(processes=1)
    #print "Calculate_SS_voltage_replay %f" % step_level
    (SS_voltage, voltage) = pool.apply(calculate_SS_voltage_replay_subprocess, [blueconfig, gid, step_level, start_time, stop_time])
    #(SS_voltage, voltage) = calculate_SS_voltage_replay_subprocess(blueconfig, gid, step_level)
    pool.terminate()
    del pool
    return (SS_voltage, voltage)


def calculate_SS_voltage_replay_subprocess(blueconfig, gid, step_level, start_time=None, stop_time=None):
    """Subprocess wrapper of calculate_SS_voltage"""
    process_name = multiprocessing.current_process().name
    ssim = bglibpy.SSim(blueconfig)
    #print "%s: Calculating SS voltage of step level %f nA" % (process_name, step_level)
    #print "Calculate_SS_voltage_replay_subprocess instantiating gid ..."
    ssim.instantiate_gids([gid], synapse_detail=2, add_stimuli=True, add_replay=True)
    #print "Calculate_SS_voltage_replay_subprocess instantiating gid done"

    ssim.cells[gid].addRamp(0, stop_time, step_level, step_level)
    ssim.run(t_stop=stop_time)
    time = ssim.get_time()
    voltage = ssim.get_voltage_traces()[gid]
    SS_voltage = numpy.mean(voltage[numpy.where((time < stop_time) & (time > start_time))])
    printv("%s: Calculated SS voltage for gid %d with step level %f nA: %s mV" % (process_name, gid, step_level, SS_voltage), 1)

    #print "Calculate_SS_voltage_replay_subprocess voltage:%f" % SS_voltage

    return (SS_voltage, voltage)

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

# pylint: disable=W0223

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NestedPool(multiprocessing.pool.Pool):
    """Class that represents a MultiProcessing nested pool"""
    Process = NoDaemonProcess


def search_hyp_current_replay(blueconfig, gid, hyp_voltage,
        min_current=None, max_current=None,
        precision=None,
        nestlevel=1, max_nestlevel=None,
        start_time=None, stop_time=None):
    """Search current necessary to bring cell to hyp_voltage in a network replay"""
    process_name = multiprocessing.current_process().name

    if nestlevel > max_nestlevel:
        return (float('nan'), None)
    elif nestlevel == 1:
        printv("%s: Searching for current to bring gid %d to %f mV" % (process_name, gid, hyp_voltage), 1)
    med_current = min_current + abs(min_current - max_current) / 2
    (new_hyp_voltage, new_hyp_voltage_trace) = calculate_SS_voltage_replay(blueconfig, gid, med_current, start_time=start_time, stop_time=stop_time)
    #print "Detected voltage: ", new_hyp_voltage
    if abs(new_hyp_voltage - hyp_voltage) < precision:
        return (med_current, new_hyp_voltage_trace)
    elif new_hyp_voltage > hyp_voltage:
        return search_hyp_current_replay(blueconfig, gid, hyp_voltage,
                min_current=min_current,
                max_current=med_current,
                precision=precision,
                nestlevel=nestlevel+1,
                start_time=start_time, stop_time=stop_time,
                max_nestlevel=max_nestlevel)
    elif new_hyp_voltage < hyp_voltage:
        return search_hyp_current_replay(blueconfig, gid, hyp_voltage,
                min_current=med_current,
                max_current=max_current,
                precision=precision,
                nestlevel=nestlevel+1,
                start_time=start_time, stop_time=stop_time,
                max_nestlevel=max_nestlevel)

class search_hyp_function(object):
    """Function object"""
    def __init__(self, blueconfig, target_voltage=None, min_current=None, max_current=None, precision=None, max_nestlevel=None, start_time=None, stop_time=None):
        self.blueconfig = blueconfig
        self.target_voltage = target_voltage
        self.min_current = min_current
        self.max_current = max_current
        self.start_time = start_time
        self.stop_time = stop_time
        self.precision = precision
        self.max_nestlevel = max_nestlevel
    def __call__(self, gid):
        return search_hyp_current_replay(self.blueconfig, gid,
                self.target_voltage,
                min_current=self.min_current, max_current=self.max_current,
                precision=self.precision,
                max_nestlevel=self.max_nestlevel,
                start_time=self.start_time, stop_time=self.stop_time)

def search_hyp_current_replay_gidlist(blueconfig, gid_list,
        target_voltage=-80,
        min_current=-1.0, max_current=0.0,
        precision=.5,
        max_nestlevel=10,
        start_time=500, stop_time=2000):
    """Search current necessary to bring cell to hyp_voltage in a network replay for a list of gids"""

    pool = NestedPool(len(gid_list))
    results = pool.map(search_hyp_function(blueconfig,
        target_voltage=target_voltage,
        min_current=min_current, max_current=max_current,
        precision=precision,
        max_nestlevel=max_nestlevel,
        start_time=start_time, stop_time=stop_time
        ), gid_list)
    pool.terminate()

    currentlevels_voltagetraces = {}
    for gid, result in zip(gid_list, results):
        currentlevels_voltagetraces[gid] = result

    return currentlevels_voltagetraces


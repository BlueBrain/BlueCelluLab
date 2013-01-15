'''
Static tools for bglibpy
'''

import inspect
from bglibpy.importer import neuron
import multiprocessing
import bglibpy
from bglibpy.importer import neuron
import numpy
import warnings

BLUECONFIG_KEYWORDS = ['Run', 'Stimulus', 'StimulusInject', 'Report', 'Connection']

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
    simulation.run(1000)
    time = cell.getTime()
    voltage = cell.getSomaVoltage()
    SS_voltage = numpy.mean(voltage[numpy.where((time < 1000) & (time > 800))])
    cell.delete()

    return SS_voltage


def search_hyp_current(template_name, morphology_name, hyp_voltage, start_current, stop_current):
    """Search current necessary to bring cell to -85 mV"""
    med_current = start_current + abs(start_current - stop_current) / 2
    new_hyp_voltage = calculate_SS_voltage(template_name, morphology_name, med_current)
    print "Detected voltage: ", new_hyp_voltage
    if abs(new_hyp_voltage - hyp_voltage) < .5:
        return med_current
    elif new_hyp_voltage > hyp_voltage:
        return search_hyp_current(template_name, morphology_name, hyp_voltage, start_current, med_current)
    elif new_hyp_voltage < hyp_voltage:
        return search_hyp_current(template_name, morphology_name, hyp_voltage, med_current, stop_current)


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
    simulation.run(int(inj_stop))

    time = cell.getTime()
    voltage = cell.getSomaVoltage()
    time_step = time[numpy.where((time > inj_start) & (time < inj_stop))]
    voltage_step = voltage[numpy.where((time_step > inj_start) & (time_step < inj_stop))]
    spike_detected = numpy.max(voltage_step) > -20

    cell.delete()

    return spike_detected


def search_threshold_current(template_name, morphology_name, hyp_level, inj_start, inj_stop, start_current, stop_current):
    """Search current necessary to reach threshold"""
    med_current = start_current + abs(start_current - stop_current) / 2
    spike_detected = detect_spike_step(template_name, morphology_name, hyp_level, inj_start, inj_stop, med_current)
    print "Spike threshold detection at: ", med_current, "nA", spike_detected

    if abs(stop_current - start_current) < .01:
        return stop_current
    elif spike_detected:
        return search_threshold_current(template_name, morphology_name, hyp_level, inj_start, inj_stop, start_current, med_current)
    elif not spike_detected:
        return search_threshold_current(template_name, morphology_name, hyp_level, inj_start, inj_stop, med_current, stop_current)


def detect_threshold_current(template_name, morphology_name, hyp_level, inj_start, inj_stop):
    """Search current necessary to reach threshold"""
    return search_threshold_current(template_name, morphology_name, hyp_level, inj_start, inj_stop, 0.0, 1.0)

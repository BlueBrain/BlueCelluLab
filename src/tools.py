'''
Static tools for bglibpy
'''

import inspect

BLUECONFIG_KEYWORDS = ['Run', 'Stimulus', 'StimulusInject', 'Report', 'Connection']

def _me():
    '''Used for debgugging. Reads the stack and provides info about which
    function called  '''
    print 'Call -> from %s::%s' % (inspect.stack()[1][1],
    inspect.stack()[1][3])
    

def load_nrnmechanisms(libnrnmech_location):
    """Load another shared library with neuron mechanisms"""
    neuron.h.nrn_load_dll(libnrnmech_location)

def parse_complete_BlueConfig(fName) :
    """ Simplistic parser of the BlueConfig file """
    bc = open(fName,'r')
    uber_hash = {}#collections.OrderedDict
    for keyword in BLUECONFIG_KEYWORDS :
        uber_hash[keyword] = {}
    line = bc.next()

    while(line != '') :
        stripped_line = line.strip()
        if(stripped_line.startswith('#')) :
            ''' continue to next line '''
            line = bc.next()
        elif(stripped_line == '') :
            # print 'found empty line'
            try :
                line = bc.next()
            except StopIteration :
                # print 'I think i am at the end of the file'
                break
        elif(stripped_line.split()[0].strip() in BLUECONFIG_KEYWORDS ) :
            key = stripped_line.split()[0].strip()
            value = stripped_line.split()[1].strip()
            # print 'came accross key >',key,'<, value: >',value,'<'
            # parse the entries in that block
            parsed_dict = _parse_block_statement(bc)
            # add to the correct uber-dictionary
            uber_hash[key][value] = parsed_dict
            # print 'came accross key >',key,'<, value: >',value,'<'
            # print 'added the following dict:\n',parsed_dict
            line = bc.next()
        else :
            line = bc.next()
    return uber_hash

def _parse_block_statement(file_object) :
    ''' parse the content of the blocks in BlueConfig'''
    file_object.next() # skip the opening "}"
    line = file_object.next().strip()
    ret_dict = {}
    while(not line.startswith('}')) :
        # print '_parse_block_statement, line: >',line,'<'
        if(len(line) == 0 or line.startswith('#')) :
            line = file_object.next().strip()
        else :
            key = line.split(' ')[0].strip()
            values = line.split(' ')[1:]
            for value in values :
                if(value == '') :
                    pass
                else :
                    ret_dict[key] = value
                # print '_parse_block... stored [',key,']: >', value,'<'
            line = file_object.next().strip()
        # raw_input('press ENTER')
    # print 'line after the }, returning: ', ret_dict
    return ret_dict

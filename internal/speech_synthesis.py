from internal import tdklatt
import numpy as np
import warnings

### Klatt Synthesis using formant frequencies at pre-defined measurement nodes

### Part of the VTSim Package

def do_synthesis(speech_qualities, F0, max_number_of_formants, sim_name, nodes, sim_output_path):
    warnings.filterwarnings("ignore") ### tdklatt can generate some math warnings, but these don't have a noticeable effect on outputs
    header_file = np.loadtxt(sim_output_path+'\\'+str(sim_name)+'.txt', delimiter=': ',dtype='str')
    header = {row[0]:row[1] for row in header_file}
    fs = int(header['Sim Frequency'])

    node_positions = nodes[1:] ### Don't have formant values for source node

    node_index = 0
    for parameters in speech_qualities:
        formants, bandwidths = parameters
        params = tdklatt.KlattParam1980(F0 = F0, FF = formants, BW = bandwidths)
        s = tdklatt.klatt_make(params)
        s.run()
        s.save(sim_output_path+'\\output'+str(node_positions[node_index])+'.wav')
        node_index+=1

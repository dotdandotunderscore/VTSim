from internal import voxelise ### Reads stl model and creates voxel list
# from internal import boundaries_calc ### takes voxel list and creates boundary locations for DWM - Designed for models where glottis is enclosed
from internal import boundaries_calc_openglot ### takes voxel list and creates boundary locations for DWM - Designed for models where glottis is at min_z
from internal import find_nodes ### takes boundary locations and source node and generates a list of nodes to interrogate after simulation
# from internal import acoustic_simulation ### takes boundary locations and does DWM simulation - Designed for models where glottis is enclosed
from internal import acoustic_simulation_openglot ### takes boundary locations and does DWM simulation - Designed for models where glottis is at min_z
from internal import vttf_gen ### takes simulation output and calculates VTTFs at given node points
from internal import source_filter ### takes VTTF and convolves with LF Model to make a temporary speech signal
from internal import lpc ### takes voice output and uses LPC to calculate formant values
from internal import speech_synthesis ### takes formant values and creates speech output with a Klatt Synthesiser

import numpy as np
import argparse
import os
# Create the parser
my_parser = argparse.ArgumentParser(description='Run an Acoustic Simulation of a Human Vocal Tract')
# Add the arguments
my_parser.add_argument('params_name',
                       metavar='params_name',
                       type=str,
                       nargs='?',
                       const = 'VT_Acoustic_Sim_Params.txt',
                       help='the name of the parameters file')
# Execute the parse_args() method
args = my_parser.parse_args()
params_file_name = args.params_name

cwd = os.getcwd()

# cwd = r'D:\Dan\Documents\Dan Woods PhD\Research Outputs\AcousticSimulation\Acoustic\DWM\FullRoutine'
# params_file_name = 'VT_Acoustic_Sim_Params.txt'

params_file = np.loadtxt(cwd+'\\'+params_file_name, delimiter=': ',dtype='str')
params = {row[0]:row[1] for row in params_file}

sim_name = params['sim_name']
stl_file = params['stl_file']
sim_output_path = params['sim_output_path']

if 'nodes_file_path' in params:
    nodes_file_path = params['nodes_file_path']
else:
    nodes_file_path = False
if 'edge_admittance' in params:
    edge_admittance = float(params['edge_admittance'])
else:
    edge_admittance = 1.0
if 'glottis_admittance' in params:
    glottis_admittance = float(params['glottis_admittance'])
else:
    glottis_admittance = 1.0
if 'max_number_of_formants' in params:
    max_number_of_formants = int(params['max_number_of_formants'])
else:
    max_number_of_formants = 5
if 'use_wang_impedance' in params:
    use_wang_impedance = bool(params['use_wang_impedance'])
else:
    use_wang_impedance = False

F0 = 130.8 ### Fundamental frequency for Klatt synthesis, like vocal "pitch"
node_separation = 20 ### Distance between generated nodes, must be large enough to prevent local minima (dead ends)

if not os.path.exists(sim_output_path):
    os.makedirs(sim_output_path)

print('------ Voxelisation:')
voxs = voxelise.voxelise_stl(stl_file)
print('------ Voxelisation Complete.\n------ Make Boundaries:')
empty_space, boundaries, Yvals, source_node = boundaries_calc_openglot.make_boundaries(voxs, use_wang_impedance)
print('------ Boundaries Made.')
###
if nodes_file_path:
    node1,node2,node3 = np.loadtxt(nodes_file_path, delimiter=',', unpack=True, dtype='int')
    nodes = np.stack((node1,node2,node3), axis=1)
    find_nodes.visual_node_confirmation(sim_output_path, nodes, boundaries, empty_space)
    nodes = [tuple(i) for i in nodes]
    print('------ Do Acoustic Simulation:')
elif os.path.isfile(sim_output_path+'\\nodes.txt'):
    print('------ Pre-existing nodes file found.')
    node1,node2,node3 = np.loadtxt(sim_output_path+'\\nodes.txt', delimiter=',', unpack=True, dtype='int')
    nodes = np.stack((node1,node2,node3), axis=1)
    find_nodes.visual_node_confirmation(sim_output_path, nodes, boundaries, empty_space)
    nodes = [tuple(i) for i in nodes]
    print('------ Do Acoustic Simulation:')
else:
    print('------ nodes.txt Not Found. Generating Nodes:')
    nodes = find_nodes.sphere_fill(boundaries, source_node, empty_space, node_separation,sim_output_path)
    print('------ Nodes Generated.\n------ Do Acoustic Simulation:')

###
acoustic_simulation_openglot.DWM3D(boundaries, Yvals, nodes,
                          sim_name, sim_output_path,
                          edge_admittance, glottis_admittance,
                          use_wang_impedance)
print('------ Simulation Done.\n------ Generate VTTFs:')
vttfs = vttf_gen.make_vttf(sim_name, nodes, sim_output_path)
print('------ VTTFs Made.\n------ Do Temporary Speech Signal Generation:')
node_signals = source_filter.make_temp_speech_signal(vttfs, sim_name, nodes, sim_output_path)
print('------ Temporary Speech Signal Generation Done.\n------ Calculate Formants:')
speech_qualities = lpc.find_formants(node_signals, max_number_of_formants,
                                     sim_name, nodes, sim_output_path)
print('------ Formants Found.\n------ Do Klatt Synthesis:')
if max_number_of_formants == 5:
    speech_synthesis.do_synthesis(speech_qualities, F0, max_number_of_formants,
                                  sim_name, nodes, sim_output_path)
else:
    print('Can\'t do Klatt Synthesis with 4 formants.')
print('------ Klatt Synthesis Done.')
print('------ Modelling Routine Complete. ------')

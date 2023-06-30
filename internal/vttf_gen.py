import numpy as np
from scipy.fft import fft, fftfreq
import os
import math

### Generates VTTFs from simulation of pressure domain and input source at pre-defined measurement nodes

### Part of the VTSim Package

fname = '\\snapshot'

def make_vttf(sim_name, nodes, sim_output_path):
    header_file = np.loadtxt(sim_output_path+'\\'+str(sim_name)+'.txt', delimiter=': ',dtype='str')
    header = {row[0]:row[1] for row in header_file}

    N = int(header['Sim Steps']) ### Number of time steps in simulation
    deltaT = float(header['Sim Time Step'])
    T = 0.646/20E3

    x = np.linspace(0,N,N)
    u_in = np.exp(-(((deltaT*x - T)/(0.29*T))**2))

    node_positions = nodes[1:] ### Don't need VTTF of source node

    num_nodes = len(node_positions)
    nodes_values = np.empty((num_nodes,N))

    nodes_indexes = []
    for i in range(len(node_positions)):
        if not os.path.isfile(sim_output_path+'\\nodevalues'+str(node_positions[i])+'.npy'):
            nodes_indexes.append(i)
    print('Nodes to make:',nodes_indexes)
    if len(nodes_indexes) > 0:
        for i in range(N):
            if i% math.ceil(N/100) == 0:
                print('Current step:',i)
            pj = np.load(sim_output_path+fname+str(i)+'.npy')
            for node in nodes_indexes:
                nodes_values[node,i] = pj[tuple(node_positions[node])]
    for i in range(len(node_positions)):
        if i in nodes_indexes:
            np.save(sim_output_path+'\\nodevalues'+str(node_positions[i]),nodes_values[i])
        else:
            print('Load node:','nodevalues'+str(node_positions[i])+'.npy')
            node_values = np.load(sim_output_path+'\\nodevalues'+str(node_positions[i])+'.npy')
            nodes_values[i] = node_values

    ### Here, node_values is of shape (len(node_positions),N) where each element contains the pressure data at a certain node throughout the simulation time

    node_index = 0

    vttf = []
    for node in nodes_values:
        node = node[:len(x)]
        f_p = fft(node) ### F domain of pressure data
        f_u = fft(u_in) ### F domain of input source
        xf = fftfreq(len(x),deltaT)
        VTTF = f_p/f_u
        scaled_VTTF = 2/len(x)*np.abs(VTTF)

        vttf.append([scaled_VTTF.astype(np.complex128),xf.astype(np.complex128)])

        node_index+=1

    return vttf

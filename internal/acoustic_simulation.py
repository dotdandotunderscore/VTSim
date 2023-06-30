import numpy as np
import math
import os, sys
import time

### W-DWM Simulation
### Reimplement of DWM3D.py on 14/11/2022

### Part of the VTSim Package

### fixed parmeters, dont change
deltaX = 1E-3
snapshot_every_x_frames = 1
c_air = 344 #ms^-1
f_s = round(c_air*math.sqrt(3)/deltaX) #Hz
sim_time = 5E-3
sim_steps = math.ceil(sim_time/(1/f_s))
admittance = 1.0
###

start_time = time.time()

def calc_p_free_space(pj,pguide_l,pguide_m,pguide_n,Y_l,Y_m,Y_n,l,m,n):
    ''' Calculates value for pressure at junction in free space using the 6 adjacent points in space.
    Step 1: Calculate pressure at junction = (sum of incident pressures*admittances of guides)/ sum of admittances
    Step 2: Calculate outgoing pressure for next time step = junction pressure - incident pressure
    '''
    sum_of_adjacent_junctions = 0
    sum_of_admittances = 0
    # for junction l,m,n
    # Need guides: l+1,m,n,0 , l,m,n,1 , l,m+1,n,0 , l,m,n,1 , l,m,n+1,0 , l,m,n,1
    backward = [pguide_l[1,l,m,n,1],pguide_m[1,l,m,n,1],pguide_n[1,l,m,n,1]]
    Ybackward = [Y_l[l,m,n],Y_m[l,m,n],Y_n[l,m,n]]
    forwards = [pguide_l[1,l+1,m,n,0],pguide_m[1,l,m+1,n,0],pguide_n[1,l,m,n+1,0]]
    Yforwards = [Y_l[l+1,m,n],Y_m[l,m+1,n],Y_n[l,m,n+1]]
    for i in range(3):
        sum_of_adjacent_junctions += Yforwards[i]*forwards[i]
        sum_of_admittances += Yforwards[i]
        sum_of_adjacent_junctions += Ybackward[i]*backward[i]
        sum_of_admittances += Ybackward[i]
    pj_next_step = 2*sum_of_adjacent_junctions/sum_of_admittances
    pj[0,l,m,n] = pj_next_step
    # l+1,m,n,1 = pj_next_step - l+1,m,n,0
    # or l,m,n,0 = pj_next_step - l,m,n,1
    pguide_l[0,l+1,m,n,1] = pj_next_step - forwards[0]
    pguide_m[0,l,m+1,n,1] = pj_next_step - forwards[1]
    pguide_n[0,l,m,n+1,1] = pj_next_step - forwards[2]
    pguide_l[0,l,m,n,0] = pj_next_step - backward[0]
    pguide_m[0,l,m,n,0] = pj_next_step - backward[1]
    pguide_n[0,l,m,n,0] = pj_next_step - backward[2]
    return pj,pguide_l,pguide_m,pguide_n

def calc_p_boundary(pj,pguide_l,pguide_m,pguide_n,Y_l,Y_m,Y_n,l,m,n,ghost_point_indices):
    ''' Calculates value for pressure at junction which is adjacent to ghost points.
    Step 1: for each adjacent point, check if point and point on opposite side
    of current junction are ghosts and add to the appropriate sums.
    Step 2: Use update equation given in Gulley Thesis.
    Step 3: Calculate outgoing pressure for next time step = junction pressure - incident pressure
    '''
    # set up a series of lists to simplify indexing
    adjacent_junctions = [[l+1,m,n],[l-1,m,n],[l,m+1,n],[l,m-1,n],[l,m,n+1],[l,m,n-1]]
    Y_vals = [Y_l[l,m,n],Y_l[l+1,m,n],Y_m[l,m,n],Y_m[l,m+1,n],Y_n[l,m,n],Y_n[l,m,n+1]]
    ghost_points = [adjacent_junctions[i] for i in ghost_point_indices]
    free_space = [i for j, i in enumerate(adjacent_junctions) if j not in ghost_point_indices]
    back_indices = [1,0,3,2,5,4]
    # Boundary equation given by Gulley from Kowalczyk2008 are actually entirely
    # general to different boundaries, so we use a general form of the algorithm
    # here
    term1_sum = 0
    term2_sum = 0
    G_sum = 0
    for i in range(len(adjacent_junctions)):
        if adjacent_junctions[i] in free_space:
            # True if point not in wall
            l_adj,m_adj,n_adj = adjacent_junctions[i]
            if adjacent_junctions[back_indices[i]] in ghost_points:
                # True if opposite point is in wall
                term1_sum += pj[1,l_adj,m_adj,n_adj]
                G_sum += admittance#Y_vals[i]
            else:
                # True if neither point is in wall
                term2_sum += pj[1,l_adj,m_adj,n_adj]
    term1_const = (2*math.sqrt(3))/(3*(math.sqrt(3)+G_sum))
    term2_const = (math.sqrt(3))/(3*(math.sqrt(3)+G_sum))
    term3_const = (G_sum-math.sqrt(3))/(G_sum+math.sqrt(3))
    pj_next_step = term1_const*term1_sum + term2_const*term2_sum + term3_const*pj[2,l,m,n]
    pj[0,l,m,n] = pj_next_step
    # Update the incident pressures for adjacent junctions in free space
    # l+1,m,n,1 = pj_next_step - l+1,m,n,0
    # or l,m,n,0 = pj_next_step - l,m,n,1
    backward = [pguide_l[1,l,m,n,1],pguide_m[1,l,m,n,1],pguide_n[1,l,m,n,1]]
    forwards = [pguide_l[1,l+1,m,n,0],pguide_m[1,l,m+1,n,0],pguide_n[1,l,m,n+1,0]]
    pguide_l[0,l+1,m,n,1] = pj_next_step - forwards[0]
    pguide_m[0,l,m+1,n,1] = pj_next_step - forwards[1]
    pguide_n[0,l,m,n+1,1] = pj_next_step - forwards[2]
    pguide_l[0,l,m,n,0] = pj_next_step - backward[0]
    pguide_m[0,l,m,n,0] = pj_next_step - backward[1]
    pguide_n[0,l,m,n,0] = pj_next_step - backward[2]
    return pj,pguide_l,pguide_m,pguide_n

def calc_p_SRL(pj,boundaries,pguide_l,pguide_m,pguide_n,Y_l,Y_m,Y_n):
    ''' Runs the Digital Waveguide Mesh pressure update algorithm on a
    Standard Rectilinear (3D cartesian grid) stencil.
    '''
    # Loops through all points in grid
    # bouundaries equals 2 in areas outside the model, 1 in the walls of the model
    # and 0 in free space inside the model
    for l in range(1,pj.shape[1]-1):
        for m in range(1,pj.shape[2]-1):
            for n in range(1,pj.shape[3]-1):
                if boundaries[l,m,n] == 2:
                    continue
                adjacent_boundaries = np.array([boundaries[l+1,m,n],boundaries[l-1,m,n],boundaries[l,m+1,n],boundaries[l,m-1,n],boundaries[l,m,n+1],boundaries[l,m,n-1]])
                ghost_point_indices = np.where(adjacent_boundaries == 2)[0]
                if len(ghost_point_indices) > 0:
                    # True if there are walls adjacent to the junction, requiring
                    # boundary related calculations
                    pj,pguide_l,pguide_m,pguide_n = calc_p_boundary(pj,pguide_l,pguide_m,pguide_n,Y_l,Y_m,Y_n,l,m,n,ghost_point_indices)
                else:
                    # True if no walls adjacent to junction
                    pj,pguide_l,pguide_m,pguide_n = calc_p_free_space(pj,pguide_l,pguide_m,pguide_n,Y_l,Y_m,Y_n,l,m,n)
    # Deals with time stepping. 0 index represents the next time step,
    # 1 index represents current time step, 2 index represents past time step
    pj[2] = pj[1]
    pj[1] = pj[0]
    pguide_l[1] = pguide_l[0]
    pguide_m[1] = pguide_m[0]
    pguide_n[1] = pguide_n[0]
    return pj,pguide_l,pguide_m,pguide_n

def transfer_function_source(t,f_s, source_node, pguide_l, pguide_m, pguide_n):
    val = math.exp(-(((1/f_s)*t - (0.646/20E3))/(0.29*(0.646/20E3)))**2)
    pguide_l[1,source_node[0],source_node[1],source_node[2],0] = val
    pguide_l[1,source_node[0]+1,source_node[1],source_node[2],1] = val
    pguide_m[1,source_node[0],source_node[1],source_node[2],0] = val
    pguide_m[1,source_node[0],source_node[1]+1,source_node[2],1] = val
    pguide_n[1,source_node[0],source_node[1],source_node[2],0] = val
    pguide_n[1,source_node[0],source_node[1],source_node[2]+1,1] = val
    return pguide_l,pguide_m,pguide_n

def snapshot3d(i, pj, path, start_time, loop_start):
    fname = '\\snapshot'+str(i)
    np.save(path+fname,pj)
    time_for_step = time.time()-loop_start
    total_time = time.time()-start_time
    if i % 5 == 0 and i != 0:
        print('Step',i,'finished.',
              'Time for step =', '{:.3f}'.format(time_for_step),
              'Total time =','{:.3f}'.format(total_time),
              'Time Left =','{:.3f}mins'.format(((total_time/i)*(sim_steps-i))/60))

def DWM3D(boundaries, Yvals, sim_name, sim_output_path, nodes_file_path):
    if os.path.isfile(sim_output_path+'\\snapshot'+str(sim_steps-1)+'.npy'):
        print('Sim already done.')
        return
    if not os.path.exists(sim_output_path):
        os.makedirs(sim_output_path)

    SizeX,SizeY,SizeZ = boundaries.shape
    pj = np.zeros((3,SizeX, SizeY, SizeZ))
    # pguide and Y indexed as: lmn+1 is forward junction, lmn is backward junction
    # pguide is further indexed with 0 in the negative direction and 1 in the positive for pressure "flow"
    pguide_l = np.zeros((2,SizeX, SizeY, SizeZ,2))
    pguide_m = np.zeros((2,SizeX, SizeY, SizeZ,2))
    pguide_n = np.zeros((2,SizeX, SizeY, SizeZ,2))
    Y_l = Yvals[0]
    Y_m = Yvals[1]
    Y_n = Yvals[2]

    print('Model Name:',sim_name)
    print('Simulation Frequency:',f_s)

    print('Total time to simulate:',str(sim_time)+'. Simulation steps:',str(sim_steps)+'.')

    node1,node2,node3 = np.loadtxt(nodes_file_path, delimiter=',', unpack=True, dtype='int')
    source_node = [node1[0],node2[0],node3[0]]

    sim_conditions = ('Simulation Name: '+str(sim_name)+'\n'
                     'Sim Time: '+str(sim_time)+'\n'
                     'Sim Steps: '+str(sim_steps)+'\n'
                     'Snapshot Frequency: '+str(snapshot_every_x_frames)+'\n'
                     'Source Node Location: '+str(source_node)+'\n'
                     'Sim Frequency: '+str(f_s)+'\n'
                     'Sim Time Step: '+str(1/f_s)+'\n'
                     'Sim Output Path: '+str(sim_output_path))

    conditions_file = open(nodes_file_path.split(nodes_file_path.split('\\')[-1])[0]+str(sim_name)+'.txt', 'w')
    conditions_file.write(sim_conditions)
    conditions_file.close()

    for t in range(sim_steps):
        loop_start = time.time()
        pj,pguide_l,pguide_m,pguide_n = calc_p_SRL(pj,boundaries,pguide_l,pguide_m,pguide_n,Y_l,Y_m,Y_n)
        pguide_l,pguide_m,pguide_n = transfer_function_source(t, f_s, source_node,pguide_l,pguide_m,pguide_n)
        if t % snapshot_every_x_frames == 0:
            snapshot3d(t, pj[1], sim_output_path, start_time, loop_start)

    print('Simulation Done.')

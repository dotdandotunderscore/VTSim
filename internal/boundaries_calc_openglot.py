import numpy as np
import math

import matplotlib.pyplot as plt

### Boundary calculation mini-module
### Reimplement of MakeBoundariesDWM.py on 14/11/2022

### Part of the VTSim Package

def flood_fill_set(boundaries,start_l,start_m,start_n):
    toFill = set()
    toFill.add((start_l,start_m,start_n))
    i=0
    while len(toFill) != 0:
        if i >= 5000000:
            print('failed')
            break
        l,m,n = toFill.pop()
        if boundaries[l,m,n] == 0:
            boundaries[l,m,n] = 1
            if l > 0:
                toFill.add((l-1,m,n))
            if l < len(boundaries)-1:
                toFill.add((l+1,m,n))
            if m > 0:
                toFill.add((l,m-1,n))
            if m < len(boundaries[0])-1:
                toFill.add((l,m+1,n))
            if n > 0:
                toFill.add((l,m,n-1))
            if n < len(boundaries[0][0])-1:
                toFill.add((l,m,n+1))
        i+=1
    return boundaries

def make_boundaries(voxels, use_wang_impedance):
    ### voxels is a XxYxZ array of the simulation domain with a 1 in full voxels
    ### X is the width, Y is the length, Z is the height
    SizeX, SizeY, SizeZ = voxels.shape

    walls_before = np.count_nonzero(voxels)
    voxels = flood_fill_set(voxels, 2, 2, 2)
    walls_after = np.count_nonzero(voxels)

    ### VT models, apply propagation space in front of lips
    empty_space =  math.floor(SizeY*0.2)
    npad = ((0, 0), (0, empty_space), (0, 0))
    boundaries = np.pad(voxels, pad_width=npad, mode='constant', constant_values=0)
    print('Old Domain Shape:',voxels.shape)
    print('New Domain Shape:',boundaries.shape)
    print('New Domain Size:',boundaries.size)

    print('Walls before fill:', walls_before)
    print('Walls after fill:', walls_after)
    print('Empty space:', boundaries.size - walls_after)

    npad = ((1, 1), (1, 1), (1, 1))
    boundaries = np.pad(boundaries, pad_width=npad, mode='constant', constant_values=2)

    if use_wang_impedance:
        # Wang Impedance Values
        rho_air = 1.17 #kgm^-3
        c_air = 346.3 # ms^-1
        rho_wall = 1000 #kgm^-3
        c_wall = 1500 #ms^-1
        Z_air = rho_air*c_air
        Z_walls = rho_wall*c_wall #Pasm^-1
    else:
        # Arnela Impedance Values
        rho_air = 1.14 #kgm^-3
        c_air = 350 #ms^-1
        Z_air = rho_air*c_air #Pasm^-1
        Z_walls = 83666 # Arnela Acoustic Impedance

    Y_air = 1/Z_air
    Y_walls = 1/Z_walls

    Y_l = np.zeros(boundaries.shape)+Y_walls
    Y_m = np.zeros(boundaries.shape)+Y_walls
    Y_n = np.zeros(boundaries.shape)+Y_walls
    for l in range(1,len(boundaries)-1):
        for m in range(1,len(boundaries[0])-1):
            for n in range(1,len(boundaries[0][0])-1):
                if boundaries[l,m,n] == 0:
                    if boundaries[l+1,m,n] == 0:
                        Y_l[l+1,m,n] = Y_air
                    if boundaries[l-1,m,n] == 0:
                        Y_l[l,m,n] = Y_air
                    if boundaries[l,m+1,n] == 0:
                        Y_m[l,m+1,n] = Y_air
                    if boundaries[l,m-1,n] == 0:
                        Y_m[l,m,n] = Y_air
                    if boundaries[l,m,n+1] == 0:
                        Y_n[l,m,n+1] = Y_air
                    if boundaries[l,m,n-1] == 0:
                        Y_n[l,m,n] = Y_air

    # boundaries[0,:,:] = 2
    # boundaries[-1,:,:] = 2
    # boundaries[:,0,:] = 2
    # boundaries[:,-1,:] = 2
    # boundaries[:,:,0] = 2
    # boundaries[:,:,-1] = 2
    Yvals = np.array((Y_l,Y_m,Y_n))

    ### Finding Glottis for source node
    n=1
    l_max,l_min,m_max,m_min,n_max,n_min=0,300,0,300,0,300
    # emptys = []
    for l in range(len(boundaries)):
        for m in range(len(boundaries[0])-empty_space-1):
            if boundaries[l,m,n] == 0:
                # print(l,m,n)
                if l > l_max:
                    l_max = l
                if l < l_min:
                    l_min = l
                if m > m_max:
                    m_max = m
                if m < m_min:
                    m_min = m
                if n > n_max:
                    n_max = n
                if n < n_min:
                    n_min = n
                # emptys.append([l,m])
    source_node = [int((l_max+l_min)/2),int((m_max+m_min)/2),int((n_max+n_min)/2)]
    if boundaries[tuple(source_node)] == 0:
        print(source_node)
    else:
        print('source node in wall')
        quit()
    # emptys_a = np.asarray(emptys)
    # plt.scatter(emptys_a[:,0],emptys_a[:,1])
    # plt.xlim((0,SizeX+1))
    # plt.ylim((0,SizeZ+1))
    # plt.show()
    # quit()

    return empty_space, boundaries, Yvals, tuple(source_node)

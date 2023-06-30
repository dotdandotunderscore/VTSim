import numpy as np
import math

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

def make_boundaries(voxels, use_wang_impedance=False):
    ### voxels is a XxYxZ array of the simulation domain with a 1 in full voxels
    ### X is the width, Y is the length, Z is the height
    SizeX, SizeY, SizeZ = voxels.shape

    ### VT models, apply propagation space in front of lips
    empty_space =  math.floor(SizeY*0.2)
    npad = ((0, 0), (0, empty_space), (0, 0))
    boundaries = np.pad(voxels, pad_width=npad, mode='constant', constant_values=0)
    print('New Domain Shape:',boundaries.shape)
    print('New Domain Size:',boundaries.size)

    print('Walls before fill:',np.count_nonzero(boundaries))
    boundaries = flood_fill_set(boundaries, 1,1,1)
    print('Walls after fill:',np.count_nonzero(boundaries))
    print('Empty space:', boundaries.size - np.count_nonzero(boundaries))

    if use_wang_impedance:
        # Wang Impedance Values
        rho_air = 1.17 #kgm^-2
        c_air = 346.3 # ms^-1
        rho_wall = 1000 #kgm^-3
        c_wall = 1500 #ms^-1
        Z_air = rho_air*c_air
        Z_walls = rho_wall*c_wall #Pasm^-3
    else:
        # Arnela Impedance Values
        rho_air = 1.14 #kgm^-3
        c_air = 350 #ms^-1
        Z_air = rho_air*c_air #Pasm^-3
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

    boundaries[0,:,:] = 2
    boundaries[-1,:,:] = 2
    boundaries[:,0,:] = 2
    boundaries[:,-1,:] = 2
    boundaries[:,:,0] = 2
    boundaries[:,:,-1] = 2
    Yvals = np.array((Y_l,Y_m,Y_n))

    return boundaries, Yvals

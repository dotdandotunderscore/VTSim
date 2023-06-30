import numpy as np
import trimesh
import matplotlib.pyplot as plt

### Voxelises input STL model
### Reimplement of voxeltest.py on 14/11/2022

### Part of the VTSim Package

def voxelise_stl(input_path, voxelizer='subdivide'):
    mesh = trimesh.load_mesh(input_path)
    resolution = 1000 # in um
    volume = mesh.voxelized(pitch=resolution*1E-6, method=voxelizer)
    voxs = volume.matrix.astype(int)

    return voxs

# input = r'D:\Dan\Documents\Dan Woods PhD\Research Outputs\AcousticSimulation\Acoustic\DWM\Models\DMH-FOOD\DMH-FoodEnclosed_0130_130vox_1mm.stl'
# voxs = voxelise_stl(input)
# print(voxs.shape)

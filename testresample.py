import resample
import slam.utils as ut
import numpy as np
import numpy.matlib
import slam.generate_parametric_surfaces as sgps
import slam.io as sio
import slam.plot as splt
import slam.curvature as scurv
import slam.topology as stop
import slam.differential_geometry as sdg

# Load data :
mesh_file1 = './KKI2009_113/MR1/lh.sphere.reg.gii'
sphFV1 = sio.load_mesh(mesh_file1)
mesh_file2 = './KKI2009_113/MR2/lh.sphere.reg.gii'
sphFV2 = sio.load_mesh(mesh_file2)

SphFV2 = resample.resample(mesh_file1,mesh_file2,True)




"""
SLAM script
How to deal with geometric noise by using spectral representation ?
"""
import slam.utils as ut
import numpy as np
import slam.generate_parametric_surfaces as sgps
import slam.io as sio
import slam.plot as splt
import slam.curvature as scurv

# Load data
# FV1: surface1 + sphere + patch
mesh_file = '../KKI2009_113/MR1/lh.white.gii'
FV1 = sio.load_mesh(mesh_file)

mesh_file = '../KKI2009_113/MR1/lh.sphere.reg.gii'
sphFV1 = sio.load_mesh(mesh_file)

# Load texture
mesh_file = '../KKI2009_113/MR1/lh.white.bassin1.gii'
texture1 = sio.load_texture(mesh_file)

# FV2: surface2 + sphere
mesh_file = '../KKI2009_113/MR2/lh.white.gii'
FV2 = sio.load_mesh(mesh_file)

mesh_file = '../KKI2009_113/MR2/lh.sphere.reg.gii'
sphFV2 = sio.load_mesh(mesh_file)

# Imports :
import numpy as np
import numpy.matlib
import slam.io as sio
import pyvista as pv

import os
import slam.utils as ut
import numpy.matlib
import slam.generate_parametric_surfaces as sgps
import slam.plot as splt
import slam.curvature as scurv
import slam.topology as stop
import slam.differential_geometry as sdg
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


# Load data :
mesh_file1 = './KKI2009_113/MR1/lh.sphere.reg.gii'
sphFV1 = sio.load_mesh(mesh_file1)
mesh_file2 = './KKI2009_113/MR2/lh.sphere.reg.gii'
sphFV2 = sio.load_mesh(mesh_file2)
coord1 = sphFV1.vertices
coord2 = sphFV2.vertices
mesh_file3 = './KKI2009_113/MR2/lh.white.gii'
FV2 = sio.load_mesh(mesh_file3)
# Load Texture
mesh_file = './KKI2009_113/MR1/lh.white.bassin1.gii'
texture1 = sio.load_texture(mesh_file)
# Texture to array
texture1 = texture1.darray[0]


# Visualisation mesh1
visb_sc = splt.visbrain_plot(mesh=sphFV1, tex=texture1,caption='Surface 1 - ROI')
visb_sc.preview()

# Initialisation of var for texture
texture2 = np.zeros(len(coord2))

# Algo Interpolation
for i in range(len(coord2)):
    ind = np.argmin(np.linalg.norm(coord1 - sphFV2.vertices[i,:]))
    texture2[i] = texture1[ind]

# Saving texture in npy format
with open('texture2.npy', 'wb') as f:
    np.save(f,texture2)

# Visualisation mesh2
visb_sc = splt.visbrain_plot(mesh=sphFV2, tex=texture2,caption='Surface 2 - ROI')
visb_sc.preview()





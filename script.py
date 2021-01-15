"""
SLAM script
How to deal with geometric noise by using spectral representation ?
"""
import slam.utils as ut
import numpy as np
import numpy.matlib
import slam.generate_parametric_surfaces as sgps
import slam.io as sio
import slam.plot as splt
import slam.curvature as scurv



# Load data
# FV1: surface1 + sphere + patch
mesh_file = '../KKI2009_113/MR1/lh.white.gii'
FV1 = sio.load_mesh(mesh_file)

# Get transformation Matrix
M1 = FV1.principal_inertia_transform
mesh_file = '../KKI2009_113/MR1/lh.sphere.reg.gii'
sphFV1 = sio.load_mesh(mesh_file)

# Load texture
mesh_file = '../KKI2009_113/MR1/lh.white.bassin1.gii'
texture1  = sio.load_texture(mesh_file)
texture1 = texture1.darray[0]

# FV2: surface2 + sphere
# Get transformation Matrix
mesh_file = '../KKI2009_113/MR2/lh.white.gii'
FV2 = sio.load_mesh(mesh_file)
M2 = FV2.principal_inertia_transform

mesh_file = '../KKI2009_113/MR2/lh.sphere.reg.gii'
sphFV2 = sio.load_mesh(mesh_file)


# Transformation on FV2, inv(A1)*(A2*X2+T2-T1)
transFV2=FV2
transFV2.vertices = (FV2.vertices.dot(M2[0:3,0:3].T) + np.matlib.repmat((M2[0:3,3]-M1[0:3,3]).T , len(FV2.vertices),1)).dot(np.linalg.inv(M1[0:3,0:3]).T)

# TO DO resample 2nd surface

# Visualization of first surface
visb_sc = splt.visbrain_plot(mesh=FV1, tex=texture1,caption='Surface 1 - ROI')
visb_sc.preview()

# Calculate displacement between two surfaces
# transFV2 must be replaced with resampled surface
displacement= np.sqrt(np.array((transFV2.vertices-FV1.vertices[0:len(transFV2.vertices)])**2).sum(axis=1))

"""
SLAM script
How to deal with geometric noise by using spectral representation ?
"""

import os
import slam.utils as ut
import numpy as np
import numpy.matlib
import slam.generate_parametric_surfaces as sgps
import slam.io as sio
import slam.plot as splt
import slam.curvature as scurv
import slam.topology as stop
import slam.differential_geometry as sdg
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# This is a camera state for visbrain objects
# camera state must be set upon initialization
# and not a posteriori but cannot be passed as argument
# to splt.visbrain_plot. Could be ameliorated
CAM_STATE = dict(azimuth=-90,        # azimuth angle
                 elevation=0,     # elevation angle
                 scale_factor=180  # distance to the camera
                 )
S_KW = dict(camera_state=CAM_STATE)



def get_displacement_approx(FV1,FV2):
    """
    Returns approximate displacement between two meshes of diferrent
    sampling variables. This is an heuristic function developped for developement
    purposes only resampling is in order.(Naive approach n^2 complexity)
    Can also import displacement from file.
    """

    # Import from file if it already exists
    if os.path.exists('displacement.npy') and os.path.getsize('displacement.npy') > 0:
        with open('displacement.npy', 'rb') as f:
            displacement = np.load(f)

    # Calculate and save displacement if not
    else:
        # Allocating space for displacment matrix
        displacement = np.zeros(len(FV2.vertices))

        # Initialize counting variable
        cnt = 0

        # For each vertex in second mesh
        for i in FV2.vertices:

            # Calculate squared distance between all vertices in fist mesh and get minimal solution
            displacement[cnt]  = np.min(np.sqrt(np.array((FV1.vertices-i)**2)))

            #Augment iterator
            cnt +=1

        # Save result to file 
        # Note that the file must be created beforehand
        with open('displacement.npy', 'wb') as f:
             np.save(f,displacement)

    return displacement

# Load data
# FV1: surface1 + sphere + patch
mesh_file = '../KKI2009_113/MR1/lh.white.gii'
FV1 = sio.load_mesh(mesh_file)

# Get transformation Matrix
# The ransfromation matrix was wxtracted manually from the GIfTI files and strored in txt files
M1 = np.loadtxt('../KKI2009_113/MR1/M.txt')
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
M2 = np.loadtxt('../KKI2009_113/MR2/M.txt')

# Transformation on FV2, inv(A1)*(A2*X2+T2-T1)
transFV2=FV2
transFV2.vertices = (FV2.vertices.dot(M2[0:3,0:3].T) + np.matlib.repmat((M2[0:3,3]-M1[0:3,3]).T , len(FV2.vertices),1)).dot(np.linalg.inv(M1[0:3,0:3]).T)

# TO DO resample 2nd surface


# Visualization of first surface
visb_sc = splt.visbrain_plot(mesh=FV1, tex=texture1,caption='Surface 1 - ROI')
visb_sc.preview()

# Calculate displacement between two surfaces
# transFV2 must be replaced with resampled surface
displacement = get_displacement_approx(FV1,transFV2)

# Visualize displacement as a texture
visb_sc = splt.visbrain_plot(mesh=transFV2, tex=displacement,caption='Displacement between surfaces',
                             cblabel='displacement')
visb_sc.preview()

# Segment mesh based on texture
sub_meshes, sub_tex, sub_corresp = stop.cut_mesh(FV1,texture1)

# Segment txture respectively
ind = np.argwhere(texture1>0)
disp2 = displacement[ind]


# Visualize sub mesh with displacement texture
visb_sc = splt.visbrain_plot(mesh=sub_meshes[0],tex=disp2.flatten() ,caption='Displacement on ROI',
                             cblabel='displacement')

visb_sc.preview()

# Calculate laplacian eigenvectors for graphs

# Get laplacian matrix (performed 2x)
_,B1 = sdg.compute_mesh_laplacian(sub_meshes[0], lap_type='fem')
# Calculate eigenvectors
V1 = sdg.mesh_laplacian_eigenvectors(sub_meshes[0],nb_vectors=100)

# Get laplacian matrix (performed 2x)
_,B2 = sdg.compute_mesh_laplacian(FV1, lap_type='fem')
# Calculate eigenvectors
V2 = sdg.mesh_laplacian_eigenvectors(FV1)

# Get spectrum for meshes
spectrum1= np.dot(np.transpose(V1),B1*sub_meshes[0].vertices)
spectrum2=np.transpose(V2)*B2*np.transpose(FV1.vertices)

# Get spectrum of displacement
spectrum_displacement= np.dot(np.transpose(V1),B1*disp2)

# Visualize spectrums
plt.plot(np.log10(np.abs(spectrum1[0,:])),'b')
plt.plot(np.log10(np.abs(spectrum2[0,:])),'r')
plt.xlabel('Frequencies')
plt.ylabel('Amplitudes (log10(abs(.) )')
plt.show()
ax = 1
n = len(spectrum1[ax,:])
x = np.arange(n)
plt.plot(x,2*(spectrum1[ax,:]-spectrum2[ax,0:n])/(np.abs(spectrum1[ax,0:n])+np.abs(spectrum2[ax,0:n])))
plt.xlabel('Frequencies')
plt.ylabel('Relative errors')
plt.show()

# Perform Principal Component Analysis (PCA)
PFV = np.column_stack((np.asarray(transFV2.vertices),displacement.T))
pca = PCA(3)
X_new = pca.fit_transform(PFV)
scores = pca.score_samples(PFV)

plt.scatter(X_new[:,0],X_new[:,1])
plt.title(' Approximated Projection of the patch => dimensions of the quadric')
plt.show()

# Get length and width of mesh
L=max(X_new[:,0])-min(X_new[:,0])
w=max(X_new[:,1])-min(X_new[:,1])

# Empirically chosen value for height of mesh
h=10 

# Generate quadric surface
a=4*h/(w**2)
#Z=a*Y.^2

quadricFV = sgps.generate_quadric([1,a])


# Analyze spectrum
_,B3 = sdg.compute_mesh_laplacian(quadricFV, lap_type='fem')
V3 = sdg.mesh_laplacian_eigenvectors(quadricFV)

spectrum_quad =np.transpose(V3)*B3*np.transpose(quadricFV.vertices)

# Random displacement in the spectrum
randomFV=quadricFV

# Get lenth of vertices array
N=len(randomFV.vertices)

# Add more randomness to quadric surface
randomFV = quadricFV

# Generate random displacement based on displacement spectrum
randomD=np.dot(V3,np.transpose(spectrum_displacement[0:N-1]))
print(randomD.shape)
# Randomly  assign signs for positive value vector
# produces depreciation warning 
#randSign = np.sign(np.random.randint(-1,1+1,size=N))
randSign = np.sign(np.random.random_integers(-1,1,size=N))
randomD= np.multiply(randomD[:,1],randSign)
randomFV.vertices=randomFV.vertices+4*np.matlib.repmat(randomD,3,1).T*quadricFV.vertex_normals

# Visualize randomized quadric surface
visb_sc = splt.visbrain_plot(mesh=randomFV,caption='displacement',
                             cblabel='displacement')

visb_sc.preview()


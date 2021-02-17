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
import nibabel as nib
import trimesh.transformations as trans
from sklearn.decomposition import PCA
from lxml import etree
from visbrain.objects import SceneObj
from scipy.sparse.linalg import lgmres, eigsh
visc_bg = SceneObj(bgcolor='white', size=(1000, 1000))
solver_tolerance = 1e-6

# This is a camera state for visbrain objects
# camera state must be set upon initialization
# and not a posteriori but cannot be passed as argument
# to splt.visbrain_plot. Could be ameliorated
CAM_STATE = dict(azimuth=-90,        # azimuth angle
                 elevation=0,     # elevation angle
                 scale_factor=180  # distance to the camera
                 )
S_KW = dict(camera_state=CAM_STATE)


def my_mesh_laplacian_eigenvectors(mesh, nb_vectors=1):
    """
    compute the nb_vectors first non-null eigenvectors of the graph Laplacian
     of mesh
    :param mesh:
    :param nb_vectors:
    :return:
    """
    lap, lap_b = sdg.compute_mesh_laplacian(mesh)
    w, v = eigsh(lap.toarray(), nb_vectors + 1, M=lap_b.toarray(),
                 sigma=solver_tolerance,which='SM')
    return v[:, 1:], lap_b

def get_affine_from_gifti(filename):

    img = nib.load(filename)
    header = img.header
    print(img.CoordinateSystemTransformationMatrix)
    return data.affine

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
            print(displacement[1])
    # Calculate and save displacement if not
    else:
        # Allocating space for displacment matrix
        displacement = np.zeros(len(FV2.vertices))

        # Initialize counting variable
        cnt = 0
        total = len(FV2.vertices)
        # For each vertex in second mesh
        for i in range(total):
            if i%1000 ==0:
                print("finished",i/total*100)
            """
            k = np.sqrt(np.array((FV1.vertices-i)**2))
            n = np.zeros(len(k))
            cnt2  =  0 
            for j in k:
                n[cnt2]  = np.sum(j)
                cnt2+=1
            displacement[cnt] = np.min(n)
            """
            # Calculate squared distance between all vertices in fist mesh and get minimal solution
            #print(np.linalg.norm(np.sqrt(np.array((FV1.vertices-i)**2))))
            #print(np.min(np.linalg.norm(np.sqrt(np.array((FV1.vertices-i)**2)),axis  = 1)))
            #displacement[cnt]  = np.min(np.linalg.norm(np.sqrt(np.array((FV1.vertices-i)**2)),axis=0))
            displacement[i]  = np.min(np.linalg.norm(FV1.vertices-FV2.vertices[i,:],axis=0))
            #Augment iterator
            #cnt +=1

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
mesh_file = '../KKI2009_113/MR2/lh.sphere.reg.gii'
sphFV2 = sio.load_mesh(mesh_file)

M2 = np.loadtxt('../KKI2009_113/MR2/M.txt')

# Transformation on FV2, inv(A1)*(A2*X2+T2-T1)
transFV2=FV2

print(M2)
print(M2[0:3,0:3].T)
print(M1[0:3,0:3])
print(np.matlib.repmat((M2[0:3,3]-M1[0:3,3]).T , len(FV2.vertices),1))
#mat = trans.translation_matrix(FV2.vertices)


transFV2.vertices = (FV2.vertices.dot(M2[0:3,0:3].T) + np.matlib.repmat((M2[0:3,3]-M1[0:3,3]).T , len(FV2.vertices),1)).dot(np.linalg.inv(M1[0:3,0:3]).T)


# Visualization of first surface
visc_bg = SceneObj(bgcolor='white', size=(1000, 1000))
visb_sc = splt.visbrain_plot(mesh=FV1, tex=texture1,visb_sc = visc_bg,caption='Surface 1 - ROI')
visb_sc.preview()

# Calculate displacement between two surfaces
# transFV2 must be replaced with resampled surface
displacement = get_displacement_approx(FV1,FV2)



# Visualize displacement as a texture
visc_bg = SceneObj(bgcolor='white', size=(1000, 1000))
visb_sc = splt.visbrain_plot(mesh=transFV2, tex=displacement,visb_sc = visc_bg,caption='displacement between surfaces',
                             cblabel='displacement')
visb_sc.preview()

# Segment mesh based on texture

# Get texture for second mesh (has to be created with interpolation script)
with open('texture2.npy', 'rb') as f:
    texture2 = np.load(f)

sub_meshes, sub_tex, sub_corresp = stop.cut_mesh(FV1,texture1)
sub_meshes1, sub_tex1, sub_corresp1 = stop.cut_mesh(FV2,texture2)
test_mesh,sub_tex1, sub_corresp1 = stop.cut_mesh(sphFV2,texture2)


# Segment txture respectively
# sub_tex1 rmplacer
ind = np.argwhere(texture2>0)
disp2 = displacement[sub_corresp1[0]]
disp3 = disp2.flatten()
end = len(test_mesh[0].vertices)


# Visualize sub mesh with displacement texture
visc_bg = SceneObj(bgcolor='white', size=(1000, 1000))
visb_sc = splt.visbrain_plot(mesh=test_mesh[0],tex=disp3[0:end] ,visb_sc = visc_bg,caption='displacement',
                             cblabel='displacement')

visb_sc.preview()

red_tex = np.ones(len(sub_meshes[0].vertices))
visc_bg = SceneObj(bgcolor='white', size=(1000, 1000))
visb_sc = splt.visbrain_plot(mesh=sub_meshes[0],tex=red_tex ,visb_sc = visc_bg,caption='displacement',
                             cblabel='displacement')

visb_sc.preview()

# Calculate laplacian eigenvectors for graphs

# Get laplacian matrix (performed 2x)
n = len(sub_meshes[0].vertices)
V1,B1 =  my_mesh_laplacian_eigenvectors(test_mesh[0],nb_vectors=701)

# Get laplacian matrix (performed 2x)
V2,B2 =  my_mesh_laplacian_eigenvectors(sub_meshes1[0],nb_vectors=701)
# Get spectrum for meshes

spectrum1=np.dot(np.transpose(V1),B1*test_mesh[0].vertices)
spectrum2=np.dot(np.transpose(V2),B2*sub_meshes1[0].vertices)# np.transpose(V2)*B2*np.transpose(FV1.vertices)

# Get spectrum of displacement

spectrum_displacement=np.dot(np.transpose(V2)*B2,disp3[0:end])

# Visualize spectrums
plt.plot(np.log10(np.abs(spectrum_displacement)),'b')
#plt.plot(np.log10(np.abs(spectrum2[0,:])),'r')
plt.xlabel('Frequencies')
plt.ylabel('Amplitudes (log10(abs(.) )')
plt.show()


# Perform Principal Component Analysis (PCA)
PFV = np.column_stack((np.asarray(sub_meshes1[0].vertices),disp2.flatten()[0:end].T))
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
quadricFV = sgps.generate_quadric([1,a],nstep=[int(27),int(26)])

# Analyze quadric spectrum"
V3,B3 = my_mesh_laplacian_eigenvectors(quadricFV, nb_vectors=700)
spectrum_quad =np.dot(np.transpose(V3)*B3,quadricFV.vertices)

# Random displacement in the spectrum
randomFV=quadricFV

# Get lenth of vertices array
N=len(randomFV.vertices)

# Add more randomness to quadric surface
randomFV = quadricFV

# Generate random displacement based on displacement spectrum
randomD=np.dot(V3,spectrum_displacement[0:N-1])


# Randomly  assign signs for positive value vector
# produces depreciation warning 
randSign = np.sign(np.random.random_integers(-1,1,size=N))
randomD= np.multiply(randomD,randSign)

randomFV.vertices=randomFV.vertices+np.dot(4*randomD,quadricFV.vertex_normals)



# Visualize randomized quadric surface
visc_bg = SceneObj(bgcolor='white', size=(1000, 1000))
visb_sc = splt.visbrain_plot(mesh=randomFV,visb_sc = visc_bg,caption='displacement',
                             cblabel='displacement')

visb_sc.preview()

randomFV=quadricFV
randomFV.vertices = quadricFV.vertices

percentage=1
perturbation=np.multiply(np.random.randn(N-1),spectrum_displacement[0:N-1])*percentage

print("last push 1")
print(V3.shape)
print(spectrum_displacement[0:N-1].shape)
print(perturbation.shape)

randomD=np.dot(V3[0:N],(spectrum_displacement[0:N-1]+perturbation).T)
print("last push")
print(V3.shape)
#print((spectrum_displacement+perturbation.T).T.shape)
print(randomD.shape)

randomFV.vertices=randomFV.vertices+np.multiply(np.matlib.repmat(randomD,3,1).T,quadricFV.vertex_normals)*0.00001

visc_bg = SceneObj(bgcolor='white', size=(1000, 1000))
visb_sc = splt.visbrain_plot(mesh=randomFV,visb_sc = visc_bg,caption='displacement',
                             cblabel='displacement')

visb_sc.preview()

randomFV = sdg.laplacian_mesh_smoothing(randomFV, nb_iter=1, dt=0.1, volume_preservation=True)
#randomFV.vertices=randomFV.vertices+np.dot(randomD.T,quadricFV.vertex_normals)

# Show eigenvectors
plt.imshow(V3)
plt.show()

visc_bg = SceneObj(bgcolor='white', size=(1000, 1000))
visb_sc = splt.visbrain_plot(mesh=randomFV,visb_sc = visc_bg,caption='displacement',
                             cblabel='displacement')

#visb_sc.record_animation('animate_example.gif', n_pic=30)

visb_sc.preview()



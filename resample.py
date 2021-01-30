import numpy as np
import numpy.matlib
import slam.io as sio
import pyvista as pv


def resample(link1, link2, showResult=True):
    # get links or spheres ?
    # Load data : link to the directory of gii files
    mesh_file1 = link1
    sphFV1 = sio.load_mesh(mesh_file1)
    mesh_file2 = link2
    sphFV2 = sio.load_mesh(mesh_file2)

    # Creating pyvista suitable points and faces
    array_of_3_m1 = numpy.vstack([3] * len(sphFV1.faces))
    array_of_3_m2 = numpy.vstack([3] * len(sphFV2.faces))

    # Mesh 1 verticies & faces
    mesh1_vertices = np.array(sphFV1.vertices)
    mesh1_faces = np.hstack([array_of_3_m1, np.array(sphFV1.faces)]).reshape(-1)
    # Mesh 2 verticies & faces
    mesh2_vertices = numpy.array(sphFV2.vertices)
    mesh2_faces = np.hstack([array_of_3_m2, np.array(sphFV2.faces)]).reshape(-1)

    # Create pyvista mesh
    mesh1 = pv.PolyData(mesh1_vertices, mesh1_faces)
    mesh2 = pv.PolyData(mesh2_vertices, mesh2_faces)

    # Resampling : make mesh2 = mesh1 in terme of points faces?
    result = mesh1.sample(mesh2)

    # Show Results in plots
    if (showResult):
        p = pv.Plotter(shape=(1, 3))
        p.add_mesh(mesh1, show_edges=True, color='white')
        p.add_mesh(pv.PolyData(mesh1.points), color='red',
                   point_size=5, render_points_as_spheres=True)
        p.subplot(0, 1)
        p.add_mesh(result)
        p.subplot(0, 2)
        p.add_mesh(mesh2, show_edges=True, color='white')
        p.add_mesh(pv.PolyData(mesh2.points), color='red',
                   point_size=5, render_points_as_spheres=True)
        p.link_views()
        p.view_isometric()
        p.show()

    # return gii format
    sphFV2.vertices = result.points
    c = np.array([b[1:] for b in result.faces.reshape(int(len(result.faces) / 4), 4)]).flatten()
    sphFV2.faces = c.reshape(int(len(c) / 3), 3)
    return sphFV2
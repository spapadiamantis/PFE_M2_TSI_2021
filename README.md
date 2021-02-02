# PFE MASTER 2 TSI AMU 2020-2021
## Dealing with geometric noise by using spectral representations

This project was developed by Ismail Majbar and Sotiris Papadiamantis under the supervision of Julien Lefèvre as part of the final project for Signal and Image Processing master's degree course at Aix-Marseille University.

The work behind this project stems from the necessity to provide a solution to a widely-encountered practical problem in the domain of representation of brain surfaces stemming from MRI data. Cortical surfaces obtained from two MRI acquisitions of the same subject at different instants are not identical but closely related. Using two said surfaces as a departure point we aim is to provide a way to generate random surfaces that share statistical similarities. This process involves resampling meshes to the same sampling space and performing spectral analysis on them. We then syntesize a new spectrum.

#### Data

The cortical surfaces used where obtained with [Freesurfer](https://surfer.nmr.mgh.harvard.edu/) software. Those surfaces have more than 100k vertices which is too large to perform a full spectral analysis. Paches were extracted using the Surfpaint toolbox in Brainvisa/Anatomist. Meshes were stored in [GIfTI](https://surfer.nmr.mgh.harvard.edu/fswiki/GIfTI)

#### Dependencies

It is developped entirely in Python using the [Surface anaLysis And Modeling (Slam)](https://github.com/brain-slam/slam). Slam is an extension of [Trimesh](https://github.com/mikedh/trimesh) which focuses on the representation of neuroanatomical surfaces. We also used [pyvista](https://docs.pyvista.org/) for surface resampling and [sklearn](https://scikit-learn.org/stable/) for principal component analysis. Finally [numpy]() is used throught the script for scientific calculations. [Numpy](https://numpy.org/) is fully integrated with trimesh.


import numpy as np
import matplotlib.pyplot as plt
from leodis import Leodis
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
import sys
import os
import numpy as np
import sys

"""
Example of how to generate the input to leodis, run the code, and save the
output.

If method = "PPV" leodis requires an array in the following format:

x, y, I, sigI, vel, + any other variables req'd for the linking.
"""


datadirectory =  './'
datafilename =  datadirectory+'fits_final_pdbi.dat'

# Load in data
dataarr    = np.loadtxt(datafilename)
# Data is organised as follows: x, y, peak intensity, error on peak intensity,
# velocity, FWHM linewidth, rms noise
dataarr = np.array([dataarr[:,0],dataarr[:,1],dataarr[:,2],dataarr[:,3], dataarr[:,4], dataarr[:,6], dataarr[:,8]])
dataarr_leodis = np.array([dataarr[0,:],dataarr[1,:],dataarr[2,:],dataarr[3,:], dataarr[4,:], dataarr[5,:]])

# Basic information required for clustering
pixel_size = 1.969
min_radius = 3.332 # Ensures 9 pixels defines the smallest structure identified
min_height = 3.0*np.mean(dataarr[6,:]) # Clusters have to be at least this value above the merge level
velo_link = 0.14 # Velocity resolution of the data
dv_link = 0.28 # If you would also like to link using the LW as an additional criterion

# Generate the cluster_criteria
cluster_criteria = np.array([min_radius, velo_link, dv_link])
# Relax criterias
relax = np.array([1.5,0.5,1.0])
# Stopping criteria
stop = 3.

# Call leodis
L = Leodis.process( dataarr_leodis, cluster_criteria, method = "PPV", min_height = min_height, pixel_size = pixel_size, relax=relax, stop = stop, verbose=True, interactive = False, n_jobs=-1 )
L.save_to(datadirectory+'example.leodis')

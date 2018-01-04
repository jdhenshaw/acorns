import numpy as np
from acorns import Acorns
import sys
import os

"""
Example of how to generate the input to acorns, run the code, and save the
output.

If method = "PPV" acorns requires an array in the following format:

x, y, I, sigI, vel, + any other variables req'd for the linking.
"""

datadirectory =  './'
datafilename =  datadirectory+'fits_final_pdbi.dat'

# Load in data
dataarr    = np.loadtxt(datafilename)
# Data is organised as follows: x, y, peak intensity, error on peak intensity,
# velocity, FWHM linewidth, rms noise
dataarr = np.array(dataarr[:,np.array([0,1,2,3,4,6,8])]).T
dataarr_acorns = np.array(dataarr[np.array([0,1,2,3,4,5]),:])

# Basic information required for clustering
pixel_size = 1.969
min_radius = 3.332 # Ensures 9 pixels defines the smallest structure identified
min_height = 3.0*np.mean(dataarr[6,:]) # Clusters have to be at least this value above the merge level
velo_link = 0.14 # Velocity resolution of the data
dv_link = 0.28 # If you would also like to link using the LW as an additional criterion

# Generate the cluster_criteria
cluster_criteria = np.array([min_radius, velo_link, dv_link])
# Relax criterias
relax = np.array([3.0,2.0,0.5]) # for interactive set to 0.0 and set interactive = True when calling acorns
# Stopping criteria
stop = 3.

# Call acorns
A = Acorns.process( dataarr_acorns, cluster_criteria, method = "PPV", min_height = min_height, pixel_size = pixel_size, relax=relax, stop = stop )
A.save_to(datadirectory+'example.acorn')

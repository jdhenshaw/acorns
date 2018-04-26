import numpy as np
from acorns import Acorns
import sys
import os
from astropy.io import fits

"""
Example of how to generate the input to acorns, run the code, and save the
output.

If method = "PPV" acorns requires an array in the following format:

x, y, I, sigI, vel, + any other variables req'd for the linking.
"""

datadirectory =  './'
datafilename =  datadirectory+'G035_cont.fits'

# Load Continuum data
hdu   = fits.open(datafilename)
header = hdu[0].header
data  = hdu[0].data
hdu.close()
data = np.squeeze(data)

rmsnoise = 7.e-5

# Create the acorns table
x = np.arange(np.size(data[0,:]))
y = np.arange(np.size(data[:,0]))
xx,yy = np.meshgrid(x,y)
# Flatten 2D arrays
xx = xx.flatten(order='F')
yy = yy.flatten(order='F')
data = data.flatten(order='F')
noisearr = np.ones(len(xx))*rmsnoise

dataarr_acorns = np.array([xx,yy,data,noisearr])

# Basic information required for clustering
pixel_size = 1.0
min_radius = 3.3848/2. # Ensures 9 pixels defines the smallest structure identified
min_height = 3.0*noisearr[0] # Clusters have to be at least this value above the merge level

# Generate the cluster_criteria
cluster_criteria = np.array([min_radius])
# Relax criterias
relax = np.array([0]) # for interactive set to 0.0 and set interactive = True when calling leodis
# Stopping criteria
stop = 3.
# number of cores used in finding nearest neighbouts. all = -1
n_jobs = 1
verbose = True
interactive = False

# Call acorns
A = Acorns.process( dataarr_acorns, cluster_criteria, method = "PP", min_height = min_height, pixel_size = pixel_size, relax=relax, stop = stop, verbose=verbose, interactive = interactive, n_jobs=n_jobs )
A.save_to(datadirectory+'example_2d.acorn')

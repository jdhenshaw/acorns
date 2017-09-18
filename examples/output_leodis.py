import numpy as np
from leodis import Leodis
import sys
from astropy.table import Table

outputdirectoryfits = './leodis_output/output_fits/'
outputdirectoryascii = './leodis_output/output_ascii/'

directory = './'
filename = 'example.leodis'
L = Leodis.load_from(directory+filename)

datadirectory =  './'
datafilename =  datadirectory+'fits_final_pdbi.dat'

# Load in data
dataarr    = np.loadtxt(datafilename)
# Data is organised as follows: x, y, peak intensity, error on peak intensity,
# velocity, FWHM linewidth, rms noise
dataarr = np.array([dataarr[:,0],dataarr[:,1],dataarr[:,2],dataarr[:,3], dataarr[:,4], dataarr[:,6], dataarr[:,8]])
dataarr_leodis = np.array([dataarr[0,:],dataarr[1,:],dataarr[2,:],dataarr[3,:], dataarr[4,:], dataarr[5,:]])

dataarr_extended = np.transpose(np.loadtxt(datafilename))

leodis_cols = [0,1,2,3]
#headings = ['x', 'y', 'intensity', 'err intensity', 'velocity','FWHM']
headings = ['x', 'y', 'intensity', 'err intensity', 'velocity', 'err velocity', 'FWHM', 'err FWHM', 'rms', 'redchisq', 'resid']

for tree in L.forest:
    if L.forest[tree].trunk.leaf_cluster:
        outputfileascii = outputdirectoryascii+'leodis_nonhierarchical_tree_'+str(tree)+'.dat'
        outputfilefits = outputdirectoryfits+'leodis_nonhierarchical_tree_'+str(tree)+'.fits'
        L.forest[tree].trunk.output_cluster_table(dataarr_leodis, outputfileascii, format="ascii", extended=dataarr_extended, leodis_cols=leodis_cols)
        L.forest[tree].trunk.output_cluster_table(dataarr_leodis, outputfilefits, format="fits", extended=dataarr_extended, leodis_cols=leodis_cols, headings=headings)
        t = Table.read(outputfilefits)
        print(t)
        print("")
    else:
        outputfileascii = outputdirectoryascii+'leodis_hierarchical_tree_'+str(tree)+'.dat'
        outputfilefits = outputdirectoryfits+'leodis_hierarchical_tree_'+str(tree)+'.fits'
        L.forest[tree].trunk.output_cluster_table(dataarr_leodis, outputfileascii, format="ascii", extended=dataarr_extended, leodis_cols=leodis_cols)
        L.forest[tree].trunk.output_cluster_table(dataarr_leodis, outputfilefits, format="fits", extended=dataarr_extended, leodis_cols=leodis_cols, headings=headings)
        t = Table.read(outputfilefits)
        print(t)
        print("")

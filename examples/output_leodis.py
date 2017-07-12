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
dataarr = np.transpose(np.loadtxt(datafilename))

leodis_cols = [0,1,2,3]
#headings = ['x', 'y', 'intensity', 'err intensity', 'velocity','FWHM']
headings = ['x', 'y', 'intensity', 'err intensity', 'velocity', 'err velocity', 'FWHM', 'err FWHM', 'rms', 'redchisq', 'resid']

for tree in L.forest:
    if L.forest[tree].trunk.leaf_cluster:
        outputfileascii = outputdirectoryascii+'leodis_nonhierarchical_tree_'+str(tree)+'.dat'
        outputfilefits = outputdirectoryfits+'leodis_nonhierarchical_tree_'+str(tree)+'.fits'
        L.forest[tree].trunk.output_cluster_table(outputfileascii, format="ascii", extended=dataarr, leodis_cols=leodis_cols)
        L.forest[tree].trunk.output_cluster_table(outputfilefits, format="fits", extended=dataarr, leodis_cols=leodis_cols, headings=headings)
        t = Table.read(outputfilefits)
        print(t)
        print("")
    else:
        outputfileascii = outputdirectoryascii+'leodis_hierarchical_tree_'+str(tree)+'.dat'
        outputfilefits = outputdirectoryfits+'leodis_hierarchical_tree_'+str(tree)+'.fits'
        L.forest[tree].trunk.output_cluster_table(outputfileascii, format="ascii", extended=dataarr, leodis_cols=leodis_cols)
        L.forest[tree].trunk.output_cluster_table(outputfilefits, format="fits", extended=dataarr, leodis_cols=leodis_cols, headings=headings)
        t = Table.read(outputfilefits)
        print(t)
        print("")

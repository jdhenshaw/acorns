import numpy as np
from acorns import Acorns
import sys
from astropy.table import Table

outputdirectoryfits = './output_files/output_fits/'
outputdirectoryascii = './output_files/output_ascii/'

directory = './'
filename = 'example.acorn'
A = Acorns.load_from(directory+filename)

datadirectory =  './'
datafilename =  datadirectory+'fits_final_pdbi.dat'

dataarr = np.transpose(np.loadtxt(datafilename))

headings = ['x', 'y', 'intensity', 'err intensity', 'velocity', 'err velocity', 'FWHM', 'err FWHM', 'rms', 'redchisq', 'resid']

for tree in A.forest:
    if A.forest[tree].trunk.leaf_cluster:
        outputfileascii = outputdirectoryascii+'acorns_nonhierarchical_tree_'+str(tree)+'.dat'
        outputfilefits = outputdirectoryfits+'acorns_nonhierarchical_tree_'+str(tree)+'.fits'
        A.forest[tree].trunk.output_cluster_table(dataarr, outputfileascii, format="ascii", headings=headings )
        A.forest[tree].trunk.output_cluster_table(dataarr, outputfilefits, format="fits", headings=headings)
        t = Table.read(outputfilefits)
        print(t)
        print("")
    else:
        outputfileascii = outputdirectoryascii+'acorns_hierarchical_tree_'+str(tree)+'.dat'
        outputfilefits = outputdirectoryfits+'acorns_hierarchical_tree_'+str(tree)+'.fits'
        A.forest[tree].trunk.output_cluster_table(dataarr, outputfileascii, format="ascii", headings=headings)
        A.forest[tree].trunk.output_cluster_table(dataarr, outputfilefits, format="fits", headings=headings)
        t = Table.read(outputfilefits)
        print(t)
        print("")

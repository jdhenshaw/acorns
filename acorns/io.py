# Licensed under an MIT open source license - see LICENSE

"""
acorns - Agglomerative Clustering for ORgansing Nested Structures
Copyright (c) 2017 Jonathan D. Henshaw
CONTACT: j.d.henshaw@ljmu.ac.uk
"""

import numpy as np
import pickle
from astropy.io import fits
from astropy.table import Table
from astropy.table import Column
import sys
if sys.version_info.major >= 3:
    proto=3
else:
    proto=2
    
sys.setrecursionlimit(20000)

def reshape_cluster_array(self, data):
    """
    Reshape the cluster array so that it has the same dimensions as the input
    data array (rather than the unassigned array)
    """

    _cluster_arr = -1*np.ones([2, len(data[0,:])])
    _cluster_arr[0,:] = np.arange(len(data[0,:]))
    sortidx = np.argsort(self.cluster_arr[0,:])
    sortedcluster_arr = self.cluster_arr[:,sortidx]
    _cluster_arr[1,sortedcluster_arr[0,:]] = sortedcluster_arr[1,:]

    self.cluster_arr = None
    self.cluster_arr = _cluster_arr

    return self

def save(self, filename):
    """
    Saves the output file - requires pickle.
    """
    pickle.dump( self, open( filename, "wb" ), protocol=proto )

def load(filename):
    """
    Loads a previously computed file - requires pickle.
    """
    loadedfile = pickle.load( open(filename, "rb"))
    return loadedfile

def output_ascii(self, data, outputfile, headings=None):
    """
    Outputs an ascii table containing the information for each cluster.
    """

    table = make_table(self, data, headings=headings )
    table.write(outputfile, format='ascii', overwrite=True, delimiter='\t')

    return

def output_fits(self, data, outputfile, headings=None):
    """
    Outputs an astropy table containing the information for each cluster.
    """
    table = make_table(self, data, headings=headings )
    table.write(outputfile, format='fits', overwrite=True)

    return

def make_table(self, data, headings=None):
    """
    Generates an astropy table to hold the information
    """

    _members = data[:,self.cluster_members]
    table = Table(meta={'name': self.cluster_idx})

    if headings is None:
        for i in range(len(_members[:,0])):
            table[str(int(i))] = Column(_members[i,:])
    elif len(headings)==len(_members[:,0]):
        for i in range(len(_members[:,0])):
            table[headings[i]] = Column(_members[i,:])
    else:
        raise IOError("Please ensure each column has a heading. ")

    return table

def housekeeping(self):
    """
    Tidy up the output file
    """

    del self.cluster_criteria
    del self.max_dist
    del self.method
    del self.min_height
    del self.min_sep
    del self.minnpix_cluster
    del self.relax
    del self.unassigned_data_updated
    del self.unassigned_data

    return self

# Licensed under an MIT open source license - see LICENSE

"""
leodis - A heirarchical structure finding algorithm
Copyright (c) 2017 Jonathan D. Henshaw
CONTACT: j.d.henshaw@ljmu.ac.uk
"""

import numpy as np
import sys
import pickle
from astropy.io import fits
from astropy.table import Table
from astropy.table import Column

def reshape_leodis_array(self):
    """
    Reshape the leodis array so that it has the same dimensions as the input
    data array (rather than the unassigned array)
    """

    _leodis_arr = -1*np.ones([3, len(self.data[0,:])])
    for j in range(len(self.data[0,:])):
        idx = np.squeeze(np.where((self.leodis_arr[0,:] == self.data[0,j]) & \
                                  (self.leodis_arr[1,:] == self.data[1,j]) & \
                                  (self.leodis_arr[3,:] == self.data[2,j]) & \
                                  (self.leodis_arr[4,:] == self.data[3,j])))


        if np.size(idx) == 1:
            _leodis_arr[0:2,j] = self.data[0:2,j]
            _leodis_arr[2, j] = self.leodis_arr[2,idx]

        if np.size(idx) == 2:
            _leodis_arr[0:2,j] = self.data[0:2,j]
            _leodis_arr[2, j] = self.leodis_arr[2,idx[0]]

    self.leodis_arr = None
    self.leodis_arr = _leodis_arr

    return self

def save_leodis(self, filename):
    """
    Saves the output leodis file - requires pickle.
    """
    pickle.dump( self, open( filename, "wb" ) )

def load_leodis(filename):
    """
    Loads a previously computed leodis file - requires pickle.
    """
    leodis = pickle.load( open(filename, "rb"))
    return leodis

def output_ascii(self, outputfile, extended=None, leodis_cols=None, headings=None):
    OF = open(outputfile,'w')
    _members = self.cluster_members

    if extended is None:
        form = gen_format(_members)
        for i in range(len(_members[0,:])):
            vals = [_members[j,i] for j in range(len(_members[:,0]))]
            OF.write(form.format(*vals))
    else:
        if leodis_cols is not None:
            _members_extended = get_members_extended(_members, extended, leodis_cols)
            form = gen_format(extended)
            for i in range(len(_members_extended[0,:])):
                vals = [_members_extended[j,i] for j in range(len(_members_extended[:,0]))]
                OF.write(form.format(*vals))
        else:
            raise IOError("Please use leodis_cols to indicate which columns in the extended array correspond to those in the leodis data. ")
    OF.close()

    return

def output_fits(self, outputfile, extended=None, leodis_cols=None, headings=None):
    """
    Outputs an astropy table containing the information for each cluster.
    """
    table = make_table(self, extended=extended, leodis_cols=leodis_cols, headings=headings )
    table.write(outputfile, format='fits', overwrite=True)
    #print(table)
    #print("")

    return

def gen_format(clustermembers):
    """
    Generates the output format for printing.
    """
    numcols = len(clustermembers[:,0])
    strnewline = str(' \n')
    formindiv = "{:12.6f} "
    form = numcols*formindiv+strnewline

    return form

def get_members_extended(clustermembers, extended, leodis_cols):
    """
    Find the cluster members within a larger array.
    """
    idloc = np.zeros(len(extended[0,:]), dtype=bool)
    for i in range(len(clustermembers[0,:])):
        idx = np.where((extended[leodis_cols[0],:] == clustermembers[0,i]) & \
                       (extended[leodis_cols[1],:] == clustermembers[1,i]) & \
                       (extended[leodis_cols[2],:] == clustermembers[2,i]) & \
                       (extended[leodis_cols[3],:] == clustermembers[3,i]))
        if np.size(idx == 1):
            idloc[idx[0]] = True
    if np.any(idloc):
        clustermembersextended = np.transpose(np.asarray([extended[:,i] for i in range(len(extended[0,:])) if idloc[i]==True]))
    else:
        raise IOError("Cluster members could not be found within the supplied data (extended). Please check leodis_cols. ")

    return clustermembersextended

def make_table(self, extended=None, leodis_cols=None, headings=None):
    """
    Generates an astropy table to hold the information
    """

    _members = self.cluster_members
    table = Table(meta={'name': self.cluster_idx})

    if extended is None:
        if headings is None:
            for i in range(len(_members[:,0])):
                table[str(int(i))] = Column(_members[i,:])
        elif len(headings)==len(_members[:,0]):
            for i in range(len(_members[:,0])):
                table[headings[i]] = Column(_members[i,:])
        else:
            raise IOError("Please ensure each column has a heading. ")
    else:
        if leodis_cols is not None:
            _members_extended = get_members_extended(_members, extended, leodis_cols)

            if headings is None:
                for i in range(len(_members_extended[:,0])):
                    table[str(int(i))] = Column(_members_extended[i,:])
            elif len(headings)==len(_members_extended[:,0]):
                for i in range(len(_members_extended[:,0])):
                    table[headings[i]] = Column(_members_extended[i,:])
            else:
                raise IOError("Please ensure each column has a heading. ")
        else:
            raise IOError("Please use leodis_cols to indicate which columns in the extended array correspond to those in the leodis data. ")

    return table

def housekeeping(self):
    """
    Tidy up the output leodis file
    """

    del self.cluster_criteria
    del self.data
    del self.link
    del self.max_dist
    del self.method
    del self.min_height
    del self.min_sep
    del self.minnpix_cluster
    del self.relax
    del self.unassigned_data
    del self.unassigned_data_relax

    return self

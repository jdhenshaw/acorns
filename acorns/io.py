# Licensed under an MIT open source license - see LICENSE

import numpy as np
import os
import sys
import warnings
import pickle
from astropy.io import fits, ascii
from astropy.table import Table, Column
from astropy import units as u
import sys
if sys.version_info.major >= 3:
    proto=3
else:
    proto=2

sys.setrecursionlimit(20000)

if sys.version_info.major >= 3:
    proto=3
else:
    proto=2

def create_directory_structure(dir):
    """
    Make the output directory
    """
    from .colors import colors

    if not os.path.exists(dir):
        os.makedirs(dir)
        mkdirectory(os.path.join(dir, 'acorn'))
        mkdirectory(os.path.join(dir, 'tables'))
        mkdirectory(os.path.join(dir, 'images'))
        mkdirectory(os.path.join(dir, 'figures'))
        mkdirectory(os.path.join(dir, 'config_files'))
    else:
        pass

def mkdirectory(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
    else:
        pass

def append_keywords(config_file, dct, all_keywords=False, description=True):
    for key in dct.keys():
        if all_keywords:
            if description:
                config_file.append(
                    '\n\n# {}'.format(dct[key]['description']))
            config_file.append('\n{} = {}'.format(key, dct[key]['default']))
        else:
            if dct[key]['simple']:
                if description:
                    config_file.append(
                        '\n\n# {}'.format(dct[key]['description']))
                config_file.append('\n{} = {}'.format(key, dct[key]['default']))
    return config_file

def make_string(st):
    newstring="\'" + str(st) + "\'"
    return newstring

def generate_config_file(filename, datadirectory, outputdir, configdir, config_filename, description, dict=None):
    """
    Creates the configuration table for scousepy

    Parameters
    ----------
    filename : string
        output filename of the config file
    datadirectory : string
        directory containing the datacube
    outputdir : string
        output directory
    configdir : string
        directory containing the config files
    config_filename : string
        filename of the configuration file
    description : bool
        whether or not to include the description of each parameter

    Notes
    -----
    adapted from Gausspy+ methodology

    """
    from collections import OrderedDict
    filename_noext, file_extension = os.path.splitext(filename)

    config_file = str('# ACORNS config file\n\n')

    default = [
        ('datadirectory', {
            'default': make_string(datadirectory),
            'description': "location of the table containing the data you would like to cluster",
            'simple': True}),
        ('filename', {
            'default': make_string(filename),
            'description': "name of the table (including extension)",
            'simple': True}),
        ('filename_noext', {
            'default': make_string(filename_noext),
            'description': "name of the table (excluding extension)",
            'simple': True}),
        ('outputdirectory', {
            'default': make_string(outputdir),
            'description': "output directory for data products",
            'simple': True}),
        ('method', {
            'default': make_string('PPV'),
            'description': "clustering method (default is clustering in PPV, options include PPV, PP, PPP, and other)",
            'simple': True}),
        ('spatial', {
            'default': '[None]',
            'description': "column locations for spatial information, list one column for each axis ",
            'simple': False}),
        ('features', {
            'default': '[None]',
            'description': "column location(s) for other features(s) to cluster, list one column for each variable ",
            'simple': False}),
        ('metric', {
            'default': make_string('absolute'),
            'description': "Distance metric for clustering. Options include `absolute`. Note that if spatial clustering will be performed based on the euclidean distance. ",
            'simple': True}),
        ('distance_threshold', {
            'default': [None],
            'description': "The linkage distance threshold at or above which clusters will not be merged. Must have the same length as features. " +\
                            "If working with spatial data, must have length = 1+features. By default the spatial distance threshold should come first in the list.",
            'simple': False}),
        ('sort', {
            'default': '[None]',
            'description': "column location(s) for variable to sort by (e.g. peak intensity)",
            'simple': True}),
        ('min_npix', {
            'default': '0.0',
            'description': "minimum number of pixels to be considered a cluster",
            'simple': True}),
        ('min_height', {
            'default': '0.0',
            'description': "minimum height above the merge level to be considered a cluster",
            'simple': True}),
        ('stop', {
            'default': 'None',
            'description': "stopping criteria for clustering. column index for stopping criterion",
            'simple': False}),
        ('stop_value', {
            'default': '0.0',
            'description': "stopping criteria for clustering. value of the stopping criterion",
            'simple': True}),
        ('verbose', {
            'default': 'True',
            'description': "print messages to the terminal [True/False]",
            'simple': True}),
        ('autosave', {
            'default': 'True',
            'description': "autosave output [True/False]",
            'simple': True}),
        ('n_jobs', {
            'default': 3,
            'description': "number of workers for ball point query of kdtree",
            'simple': True}),
        ]

    dct_default = OrderedDict(default)
    config_file = []

    config_file.append('[DEFAULT]')
    config_file = append_keywords(config_file, dct_default,
                                  all_keywords=True,
                                  description=description)

    with open(os.path.join(configdir, config_filename), 'w') as file:
        for line in config_file:
            file.write(line)

def import_from_config(self, config_file, config_key='DEFAULT'):
    """
    Read in values from configuration table.

    Parameters
    ----------
    config_file : str
        Filepath to configuration file
    config_key : str
        Section of configuration file, whose parameters should be read in addition to 'DEFAULT'.

    Notes
    -----
    adapted from Gausspy+ methodology

    """
    import ast
    import configparser

    config = configparser.ConfigParser()
    config.read(config_file)

    for key, value in config[config_key].items():
        try:
            value=ast.literal_eval(value)
            setattr(self, key, value)
        except ValueError:
            raise Exception('Could not parse parameter {} from config file'.format(key))


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

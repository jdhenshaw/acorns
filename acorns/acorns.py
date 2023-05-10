# Licensed under an MIT open source license - see LICENSE

from __future__ import print_function
from __future__ import division

import numpy as np
import itertools
import sys
import time
import os
from .colors import colors
from scipy.spatial import cKDTree

# add Python 2 xrange compatibility, to be removed
# later when we switch to numpy loops
if sys.version_info.major >= 3:
    range = range
    proto=3
else:
    range = xrange
    proto=2

try:
    input = raw_input
except NameError:
    pass

class Acorns(object):
    def __init__(self, config=''):
        self.config=config
        self.datadirectory=None
        self.filename=None
        self.filename_noext=None
        self.outputdirectory=None
        self.method=None
        self.spatial=None
        self.features=None
        self.metric=None
        self.sort=None
        self.min_npix=None
        self.min_height=None
        self.stop=None
        self.stop_value=None
        self.verbose=None
        self.autosave=None
        self.njobs=3
        self.clusters={}
        self.forest={}

    def run_setup(filename, datadirectory, outputdir=None, acorn=None,
                  description=True, verbose=True):
        """
        Generates an acorns configuration file

        Parameters
        ----------
        filename : string
            Name of the file to be loaded (no extension)
        datadirectory: string
            Path to data location
        outputdir : string, optional
            Alternate output directory. Deflault is datadirectory
        acorn : int
            run version of acorns clustering - if this is not None, acorns
            will create a new acorn for different input parameters
        description : bool, optional
            whether or not a description of each parameter is included in the
            configuration file
        verbose : bool, optional
            verbose output to terminal
        """
        from .io import create_directory_structure
        from .io import generate_config_file
        from .verbose_output import print_to_terminal

        if outputdir is None:
            outputdir=datadirectory

        if acorn is not None:
            config_filename='acorns_'+str(int(acorn))+'.config'
        else:
            config_filename='acorns.config'

        filename_noext, file_extension = os.path.splitext(filename)

        acornsdir=os.path.join(outputdir, filename_noext)
        configdir=os.path.join(acornsdir+'/config_files')
        configpath=os.path.join(acornsdir+'/config_files', config_filename)

        if verbose:
            progress_bar = print_to_terminal(stage='init', step='init')

        if os.path.exists(configpath):
            if verbose:
                progress_bar = print_to_terminal(stage='init', step='configexists')
            return os.path.join(acornsdir+'/config_files', config_filename)
        else:
            if not os.path.exists(acornsdir):
                create_directory_structure(acornsdir)
                generate_config_file(filename, datadirectory, outputdir, configdir, config_filename, description)
                if verbose:
                    progress_bar = print_to_terminal(stage='init', step='makingconfig')
            else:
                configpath=None
                print('')
                print(colors.fg._yellow_+"Warning: output directory exists but does not contain a config file. "+colors._endc_)
                print('')

        return configpath

    @staticmethod
    def process(config):
        from .io import import_from_config
        # initialise acorns
        self = Acorns(config=config)
        # import key words from config file
        import_from_config(self, config)

        start = time.time()

        # read in the data, sort it, index it, and add linking columns
        data, headings = prepare_data(self)
        # determine the length of the data set we are going to cluster
        size = n_unclustered_pts(data, headings)

        # create a kdtree if spatial queries are req'd
        if not None in self.spatial:
            kdtree = generate_kdtree(self, data, headings)

        # clustering while loop: will terminate when the number of clustered
        # points no longer changes
        count=0
        clusteridx=0
        while True:

            # get the data point to cluster
            datapt = data[count]
            # create distance table
            dataqueried = find_links(self, datapt, data, headings, kdtree)

            # update counter for moving through table
            count+=1
            if count==len(data['link']): count=0
            # stopping criteria
            old_size, size = size, n_unclustered_pts(data, headings)
            if size == old_size: break

        #print(data)
        return self

def prepare_data(self):
    """
    Prepare the data for clustering. It is assumed that the data is passed in
    ascii format.

    """
    from astropy.io import ascii
    from astropy.table import Table
    from astropy.table import Column, Row

    # read table information
    data = ascii.read(os.path.join(self.datadirectory, self.filename))
    headings = list(data.columns)
    # create new columns
    indexcolumn = Column(np.arange(len(data[headings[0]])), name='index')
    stopcolumn = Column(np.zeros_like(data[headings[0]], dtype='bool'), name='stop')
    linkcolumn = Column(np.zeros_like(data[headings[0]], dtype='bool'), name='link')
    clustercolumn = Column(np.full_like(data[headings[0]], np.nan, dtype=np.double), name='clusteridx')
    # add new columns
    data.add_column(indexcolumn, index=-1)
    data.add_column(stopcolumn, index=-1)
    data.add_column(linkcolumn, index=-1)
    data.add_column(clustercolumn, index=-1)

    # stopping criteria
    if self.stop is not None:
        data['stop'][data[headings[self.stop]]<self.stop_value]=True

    # sort the array
    if not None in self.sort:
        data.sort([headings[h] for h in self.sort])
        data.reverse()

    headings = list(data.columns)

    return data, headings

def n_unclustered_pts(data, headings):
    """
    determine the number of unclustered data points

    """
    return len(data['link'])-np.count_nonzero(data['link'])

def find_links(self, datapt, data, headings, kdtree):
    """
    Create a table containing the distances to all points

    """

    dataq=data
    # first query the spatial info if required
    if not None in self.spatial:
        # calculate euclidean distance between spatial coordinates
        rows = query_kdtree(self, datapt, kdtree, headings)
        dataq=data[rows]

    # next query the other features
    dataq = query_table(self, datapt, dataq, headings)
    # now query if there are already any established links
    rows = [row for row in range(len(dataq[headings[0]])) if dataq['link'][row]]
    # filter data table
    dataq = dataq[rows]
    return dataq

def generate_kdtree(self, data, headings):
    """
    Generates a KDTree to be queried for nearest neighbour searches
    """
    return cKDTree(np.asarray([data[headings[int(c)]] for c in self.spatial]).T)

def query_kdtree(self, datapt, kdtree, headings):
    """
    Performs query of KD Tree
    """
    rows = kdtree.query_ball_point(np.asarray([datapt[headings[int(c)]] for c in self.spatial]).T, self.distance_threshold[0], eps = 0, workers=3)
    rows.remove(0)
    return np.sort(rows)

def query_table(self, datapt, dataq, headings):
    """
    Performs query of the table
    """
    # first create a table that will contain the distances between points
    dist_tab = distance_table(self, datapt, dataq, headings)

    # if spatial information is included modify the distance_threshold list
    distance_threshold = self.distance_threshold[1::] if not None in self.spatial else self.distance_threshold

    # set row values to true/false based on distance metric
    for c, col in enumerate(list(dist_tab.columns)):
        dist_tab[col]=[True if dist_tab[row][c]<distance_threshold[c] else False for row in range(len(dist_tab[col]))]

    rows = [row for row in range(len(dist_tab[col])) if np.all(np.asarray(list(dist_tab[row])))]

    return dataq[rows]

def distance_table(self, datapt, dataq, headings):
    """
    Create a table of differences
    """
    from astropy.table import Table
    from astropy.table import Column, Row
    from scipy.spatial import distance

    # create an empty table
    dist_table=Table()

    # cycle through the features
    for feature in self.features:
        # select the correct column
        datapt_feature = datapt[headings[feature]]
        dataq_feature = dataq[headings[feature]]
        # compute distances
        if self.metric=='absolute':
            dist = [distance.minkowski(datapt_feature, dataq_feature[i], 1) for i in range(len(dataq_feature))]
        # add distances to table
        dist_table.add_column(Column(dist))

    return dist_table

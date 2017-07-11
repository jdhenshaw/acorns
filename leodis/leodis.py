# Licensed under an MIT open source license - see LICENSE

"""
leodis - A heirarchical structure finding algorithm
Copyright (c) 2017 Jonathan D. Henshaw
CONTACT: j.d.henshaw@ljmu.ac.uk
"""

import numpy as np
import itertools
import sys
import leodis_io
import time
from progressbar import AnimatedProgressBar
from scipy.spatial import distance
from scipy.spatial import cKDTree
from .cluster_definition import Cluster
from .cluster_definition import merge_clusters
from .cluster_definition import merge_data
from .cluster_definition import form_a_branch
from .tree_definition import Tree
from math import log10, floor

class Leodis(object):

    def __init__(self):
        self.leodis_arr = None
        self.clusters = None
        self.forest = None
        self.data = None
        self.cluster_idx = None
        self.method = None
        self.relax = None
        self.unassigned_data = None
        self.unassigned_data_updated = None
        self.unassigned_data_relax = None
        self.unassigned_array = None
        self.cluster_criteria = None

    @staticmethod
    def process(data, cluster_criteria, method = "PP", \
                min_height = 0, pixel_size = 0, \
                relax = 0, stop = 0, \
                verbose = False ):

        """

        Description
        -----------

        Agglomerative hierarchical clustering of data. Designed specifically for
        astronomical position-position-velocity (PPV) data (also works with PP
        or PPP data).

        Parameters
        ----------

        Examples
        --------

        datadirectory = '/my_directory/'
        datafilename = datadirectory+'myfile.dat'

        dataarr = np.loadtxt(datafilename)

        # For PPV data - construct the array in the following way: x, y,
        intensity, error intensity, velocity, ++ any other variables

        dataarr_leodis = np.array([dataarr[:,0],dataarr[:,1],
                                   dataarr[:,2],dataarr[:,3],dataarr[:,4]])

        pixel_size = 1.0
        min_radius = (3.3848*pixel_size)/2.
        min_height = 3.0*np.mean(dataarr_leodis[3,:])
        stop = 3.
        velo_link = 0.4

        # Construct the clustering criteria array and the relax array (if you
        # would like to relax the clustering criteria)
        cluster_criteria = np.array([min_radius, velo_link])
        relax = np.array([1.5, 0.5])

        L = Leodis.process( dataarr_leodis, cluster_criteria, method = "PPV", \
                            min_height = min_height, pixel_size = pixel_size, \
                            relax=relax, stop = stop, verbose=True)

        directory = '/my_output_directory/'
        L.save_to(directory+'my_output_file.leodis')

        Notes
        -----

        """

#==============================================================================#
        """
        Initial prep of key variables
        """

        self = Leodis()
        start = time.time()
        # User input information
        self.data = data
        self.cluster_criteria = cluster_criteria

        if np.size(relax) == 1:
            self.relax = relax if (relax != 0) else -1.0
            relaxcond = True if (relax != 0) else False
        else:
            self.relax = relax
            relaxcond = True

        if method == "PP":
            self.method = 0
        if method == "PPV":
            self.method = 1
        if method == "PPP":
            self.method = 2
        method = str(method)

        # Generate some important information:
        self.minnpix_cluster = get_minnpix(self, pixel_size, self.cluster_criteria[0])
        self.min_height = min_height
        self.max_dist = get_maxdist(self, pixel_size)
        self.cluster_criteria[0] = self.max_dist
        self.min_sep = 2.*self.cluster_criteria[0]

        # Prime the leodis information:
        # leodis_arr will be updated with the indices of new clusters
        self.leodis_arr = gen_leodis_arr(self, stop)
        self.clusters = {}
        self.forest = {}

        # Generate the unassigned data array
        self = find_unassigned_data(self, stop)
        unassigned_array_length = len(self.unassigned_data[0,:])

        count= 0.0
        if verbose:
            progress_bar = print_to_terminal(self, count, \
                                             unassigned_array_length, method)

#==============================================================================#
        """
        Main controlling routine for leodis
        """

        if self.method==2:
            coordinates = self.unassigned_data[0:3,:]
        else:
            coordinates = self.unassigned_data[0:2,:]
        tree = cKDTree(np.transpose(coordinates))

        # Cycle through the unassigned array
        starthierarchy = time.time()
        for i in xrange(0, unassigned_array_length):

            if verbose and (count % 1 == 0):
                progress_bar + 1
                progress_bar.show_progress()

            # Extract the current data point
            data_point = np.array(self.unassigned_data[:,i])

            # Every data point begins as a new cluster
            self.cluster_idx = i
            bud_cluster = Cluster(data_point, idx=self.cluster_idx, leodis=self)

            # Calculate distances between all data points
            link = get_links(self, i, tree)

            # Find clusters that are closely associated with the current data
            # point
            linked_clusters = find_linked_clusters(self, i, bud_cluster, link)

            """

            Notes
            -----

            Now try and merge this cluster with surrounding linked_clusters.
            From this point on there are three options for that data_point:

            1. If no linked clusters are found - add the bud cluster to the
               cluster dictionary.
            2. If a single linked cluster is found - merge the two.
            3. If multiple linked clusters are found, check the validity of each
               cluster and either merge non-independent clusters or form a
               branch.

            This philosophy follows that of agglomerative hierarchical
            clustering techniques. The basic principle is discussed here:
            http://scikit-learn.org/stable/modules/clustering.html under
            "2.3.6. Hierarchical Clustering".

            A single link measure is used to connect clusters. The strategy is
            adapted from the general methods of:

            astrodendro:
            https://github.com/dendrograms/astrodendro
            Copyright (c) 2013 Thomas P. Robitaille, Chris Beaumont, Braden
                               MacDonald, and Erik Rosolowsky
            quickclump:
            https://github.com/vojtech-sidorin/quickclump
            Copyright (c) 2016 Vojtech Sidorin

            When linking using the "PPV" methodology, single link measures may
            be insufficient and additional connectivity constraints are applied.
            Specifically - it is imposed that no two spectral features extracted
            from the same location can be merged into the same cluster.

            Additionally, an additional linking strategy is implemented which
            takes into account of the variance in the properties of the linked
            clusters (specifically those selected by the user). This is only
            implemented when trying to resolve ambiguities and is used as a way
            of establishing the "strongest" links when multiple spectral
            features have been detected.

            """

            if not linked_clusters:
                self = add_to_cluster_dictionary(self, bud_cluster)
            elif len(linked_clusters) == 1:
                self = merge_into_cluster(self, linked_clusters[0], bud_cluster)
            else:
                self = resolve_ambiguity(self, linked_clusters, bud_cluster)

        if verbose:
            progress_bar.progress = 100
            progress_bar.show_progress()
            print ''
            print ''

        endhierarchy = time.time()-starthierarchy

        #sys.exit()
#==============================================================================#
        """
        # Secondary controlling routine for leodis

        """

        # Remove insignificant clusters from the clusters dictionary and update
        # the unassigned array
        self, cluster_list, cluster_indices = update_clusters(self)

        if relaxcond==True:
            """

            Notes
            -----

            At this stage the main hierarchy has been established. If
            the 'relax' key word has been implemented by the user - the
            linking constraints will be relaxed and leodis will make a
            second pass at linking some of the remaining unassigned data.

            """

            startrelax = time.time()

            cluster_criteria_original = self.cluster_criteria

            self.cluster_criteria = get_relaxed_cluster_criteria(self, cluster_criteria_original)
            self.unassigned_data_relax = self.unassigned_data_updated
            unassigned_array_length = len(self.unassigned_data_relax[0,:])

            count=0.0
            if verbose:
                progress_bar = print_to_terminal(self, count, \
                                                 unassigned_array_length,\
                                                 method, re=True)

            # Now run the linking again with the relaxed constraints
            for i in range(0, unassigned_array_length):

                if verbose and (count % 1 == 0):
                    progress_bar + 1
                    progress_bar.show_progress()

                # Extract the current data point
                data_point = np.array(self.unassigned_data_relax[:,i])
                current_idx = get_current_index(self, i)

                # Every data point starts as a new cluster
                self.cluster_idx = current_idx
                bud_cluster = Cluster(data_point, idx=self.cluster_idx, leodis=self)

                # Calculate distances between all data points
                link = get_links(self, i, tree, re=True)

                # Find clusters that are closely associated with the current
                # data point
                linked_clusters = find_linked_clusters(self, i, bud_cluster, link, re = True)

                if not linked_clusters:
                    self = add_to_cluster_dictionary(self, bud_cluster)
                elif len(linked_clusters) == 1:
                    self = merge_into_cluster(self, linked_clusters[0], bud_cluster, re = True)
                else:
                    self = resolve_ambiguity(self, linked_clusters, bud_cluster, re = True)

            if verbose:
                progress_bar.progress = 100  # Done
                progress_bar.show_progress()
                print ''
                print ''

            endrelax = time.time()-startrelax

            # Update clusters
            self, cluster_list, cluster_indices = update_clusters(self)

#==============================================================================#
        """
        Tidy everything up for output

        """

        self = leodis_io.reshape_leodis_array(self)
        self.unassigned_array = self.data[:,np.squeeze(np.where(self.leodis_arr[2,:]==-1))]
        self = get_forest(self)

        end = time.time()-start

        if verbose:
            print('leodis took {0:0.1f} seconds for completion.').format(end)
            print('Primary clustering took {0:0.1f} seconds for completion.').format(endhierarchy)
            if relaxcond==True:
                print('Secondary clustering took {0:0.1f} seconds for completion.').format(endrelax)
            print ''
            print('leodis found a total of {0} clusters').format(len(self.clusters))
            print ''
        return self

    def save_to(self, filename):
        """
        Saves the output leodis file
        """
        from .leodis_io import save_leodis
        return save_leodis(self, filename)

    @staticmethod
    def load_from(filename):
        """
        Loads a previously computed leodis file
        """
        from .leodis_io import load_leodis
        return load_leodis(filename)


#==============================================================================#
# Methods
#==============================================================================#

def get_minnpix(self, pixel_size, radius):
    """
    Return the minimum number of pixels that will define a cluster

    """

    npix = latticepoints(radius,pixel_size)
    return npix

def latticepoints(circle_radius, pixel_size):

    """
    Returns the number of 'pixel_size' sized lattice points in a circular area
    with radius 'circle_radius'.

    Notes
    -----

    Modified this to include points at the edges. Toggle comment line to return
    to original.

    http://stackoverflow.com/questions/32835091/number-of-lattice-points-in-a-circle

    """

    numlatticepoints = 0
    npixels = int(circle_radius/pixel_size)
    for i in range(-npixels, npixels+1, 1):
        for j in range(-npixels, npixels+1, 1):
            if ((i*pixel_size)**2 + (j*pixel_size)**2) <= (np.sqrt(2.*float(npixels*pixel_size)**2))**2:
            #if ((m*pixel_size)**2 + (n*pixel_size)**2) <= npixels**2:
                numlatticepoints = numlatticepoints + 1

    return numlatticepoints

def get_maxdist(self, pixel_size):
    """
    Returns the Euclidean distance between centre of a cluster with npixels of
    size `pixel_size` to the outermost pixel

    """

    total_area = self.minnpix_cluster*pixel_size**2.

    radius = ((np.sqrt(total_area)/2.))
    if radius > 1.0:
        radius = int(radius)
    else:
        radius = round_to_1(radius)
    dist = np.sqrt(2.*float(radius)**2.)
    dist = dist+(0.05*dist)

    return dist

def round_to_1(val):
    """
    To round to 1 sig fig
    """
    return round(val, -int(floor(log10(abs(val)))))

def gen_leodis_arr(self, stop):
    """
    Returns the empty leodis array

    """

    keep = ((self.data[2,:] > stop*self.data[3,:]))
    leodis_arr = -np.ones((5,len(self.data[0,:])))
    leodis_arr = leodis_arr[:,keep]
    leodis_arr[0,:] = self.data[0, keep]
    leodis_arr[1,:] = self.data[1, keep]
    leodis_arr[3,:] = self.data[2, keep]
    leodis_arr[4,:] = self.data[3, keep]

    return leodis_arr

def find_unassigned_data(self, stop):
    """
    Return the unassigned_data

    """

    keep = ((self.data[2,:] > stop*self.data[3,:]))
    self.unassigned_data = self.data[:,keep]
    sortidx = np.argsort(self.unassigned_data[2,:])[::-1]
    self.unassigned_data = self.unassigned_data[:,sortidx]
    self.leodis_arr = self.leodis_arr[:,sortidx]

    return self

def get_links(self, index, tree, re=False):
    """
    Find the distances between the data point and all other data points
    Physical separations calculated using euclidean distance.
    For all other variables the absolute difference is calculated.

    Notes
    -----
    This code generates a array 'link' which checks the 'distance'
    between all data points in n dimensions against the clustering criteria
    imposed by the user.

    If re == True then the code finds links based on the relaxed linking
    criteria

    Tried to do this initially outside of the main loop but ran into memory
    problems when trying to link ~100000 data points. This could probably be
    streamlined somewhat. May be cumbersome depending on number of data points.

    """

    if re == False:
        coords = np.array([self.unassigned_data[0:2,index]])
    else:
        coords = np.array([self.unassigned_data_relax[0:2,index]])

    sep = None
    if self.method == 0:
        idx = tree.query_ball_point(coords, self.cluster_criteria[0], eps = 0,  n_jobs=1)
        idx = np.array(idx[0])
        idx = idx[np.where(idx != 0)]
        link = np.ones(len(idx), dtype=bool)
        if re == False:
            if len(self.cluster_criteria) != 1:
                for i in xrange(len(self.cluster_criteria)-1):
                    val = self.unassigned_data[4+i,index]
                    sep = np.abs(val - unassigned_data[4+i,idx])
                    link = (sep <= self.cluster_criteria[1+i]) & (link==True)
                    sep = None
            link = idx[np.where(np.array(link)==True)]
        else:
            if len(self.cluster_criteria) != 1:
                for i in xrange(len(self.cluster_criteria)-1):
                    val_relax = self.unassigned_data_relax[4+i,index]
                    sep = np.abs(val_relax - self.unassigned_data[4+i,idx])
                    link = (sep <= self.cluster_criteria[1+i]) & (link==True)
                    sep = None
            link = idx[np.where(np.array(link)==True)]

    if self.method == 1:
        idx = tree.query_ball_point(coords, self.cluster_criteria[0], eps = 0, n_jobs = 1)
        idx = np.array(idx[0])
        idx = idx[np.where(idx != 0)]
        link = np.ones(len(idx), dtype=bool)
        if re == False:
            velocity = self.unassigned_data[4,index]
            sep = np.abs(velocity - self.unassigned_data[4,idx])
            link = (sep <= self.cluster_criteria[1])
            sep = None

            if len(self.cluster_criteria) != 2:
                for i in xrange(len(self.cluster_criteria)-2):
                    val = self.unassigned_data[5+i,index]
                    sep = np.abs(val - self.unassigned_data[5+i,idx])
                    link = (sep <= self.cluster_criteria[2+i]) & (link==True)
                    sep = None
            link = idx[np.where(np.array(link)==True)]
        else:
            velocity_relax = self.unassigned_data_relax[4,index]
            sep = np.abs(velocity_relax - self.unassigned_data[4,idx])
            link = (sep <= self.cluster_criteria[1])
            sep = None
            if len(self.cluster_criteria) != 2:
                for i in xrange(len(self.cluster_criteria)-2):
                    val_relax = self.unassigned_data_relax[5+i,index]
                    sep = np.abs(val_relax - self.unassigned_data[5+i,idx])
                    link = (sep <= self.cluster_criteria[2+i]) & (link==True)
                    sep = None
            link = idx[np.where(np.array(link)==True)]

    if self.method == 2:
        coords = np.array([self.unassigned_data[0:3,index]])
        idx = tree.query_ball_point(coords, self.cluster_criteria[0], eps = 0, n_jobs=1)
        idx = np.array(idx[0])
        idx = idx[np.where(idx != 0)]
        link = np.ones(len(idx), dtype=bool)
        if re == False:
            if len(self.cluster_criteria) != 1:
                for i in xrange(len(self.cluster_criteria)-1):
                    val = self.unassigned_data[5+i,index]
                    sep = np.abs(val - unassigned_data[5+i,idx])
                    link = (sep <= self.cluster_criteria[1+i]) & (link==True)
                    sep = None
            link = idx[np.where(np.array(link)==True)]
        else:
            if len(self.cluster_criteria) != 1:
                for i in xrange(len(self.cluster_criteria)-1):
                    val_relax = self.unassigned_data_relax[5+i,index]
                    sep = np.abs(val_relax - unassigned_data[5+i,idx])
                    link = (sep <= self.cluster_criteria[1+i]) & (link==True)
                    sep = None
            link = idx[np.where(np.array(link)==True)]

    return link

def find_linked_clusters(self, index, cluster, linked_indices, re = False):
    """
    Return all clusters that are linked to the data_point which is currently
    under consideration.

    Notes
    -----

    If re == False, always identify the largest common ancestor of the
    linked_clusters.

    However, if re == True then we want to try and slot the cluster into the
    correct level of the previously established hierarchy.

    """

    if np.size(linked_indices) != 0.0:
        linked_clusters = [self.leodis_arr[2,ID] for ID in linked_indices \
                           if self.leodis_arr[2,ID] != -1]
    else:
        linked_clusters = []

    if len(linked_clusters) != 0:
        if re == False:
            # Identify largest common ancestor of the linked clusters - the
            # antecessor.
            linked_clusters = [self.clusters[ID].antecessor for ID in linked_clusters]
            linked_clusters = remdup_preserve_order(linked_clusters)
        else:
            # Identify the correct level in the hierarchy to add the cluster.
            antecessors = [self.clusters[ID].antecessor for ID in linked_clusters]
            antecessors = remdup_preserve_order(antecessors)
            antecessors = sorted(antecessors, key=get_cluster_idx, reverse=True)

            linked_clusters = [self.clusters[ID] for ID in linked_clusters]
            linked_clusters = remdup_preserve_order(linked_clusters)
            if len(antecessors)==1:
                linked_clusters = find_linked_clusters_single_antecessor(self, cluster, linked_clusters)
            else:
                linked_clusters = find_linked_clusters_multiple_antecessors(self, cluster, linked_clusters, antecessors)

        if self.method == 1:
            # If method == PPV then we also want to prevent multiple components
            # located at the same location from being linked together.
            if re == False:
                linked_clusters = multi_component_check(self, cluster, linked_clusters)
            else:
                linked_clusters = multi_component_check(self, cluster, linked_clusters, re = re )

    linked_clusters = sorted(linked_clusters, key=get_cluster_idx, reverse=True)

    return linked_clusters

def get_cluster_idx(_cluster):
    """
    returns cluster idx for sorting
    """

    return _cluster.cluster_idx

def find_linked_clusters_single_antecessor(self, cluster, linked_clusters):
    """
    Identifies where to slot a cluster into an already established hierarchy.

    Notes
    -----

    Often, a cluster can be linked to multiple clusters that belong to the same
    hierarchy. If a cluster can be linked to a particular hierarchy, we want to
    slot it into the correct level. This is based on the min/max values of the
    cluster and the corresponding values of the different levels in the
    hierarchy.

    This method will always return a single (or no) cluster to link to.

    """

    cluster_floor = cluster.statistics[0][0]
    cluster_ceiling = cluster.statistics[0][1]

    # Establish the min/max values of all linked clusters and arrange in
    # descending order
    floor = [link.merge_level for link in linked_clusters]
    ceiling = [link.descendants[0].merge_level if link.branch_cluster else np.inf for link in linked_clusters ]
    sort_idx = np.argsort(floor)[::-1]
    linked_clusters = [linked_clusters[idx] for idx in sort_idx ]
    floor = [floor[idx] for idx in sort_idx]
    ceiling = [ceiling[idx] for idx in sort_idx]

    # Create a boolean list of the same length as the linked_clusters list which
    # will indicate if the cluster can be slotted into a level in an already
    # established hierarchy.
    slot = []
    for j in range(len(linked_clusters)):
        # In the first instance - attempt to slot the cluster into the level of
        # the immediately adjactent cluster (that which is in linked_clusters).
        if linked_clusters[j].antecedent == None:
            # If there is no antecedent - we only care about the cluster not
            # touching the ceiling.

            if (cluster_ceiling <= ceiling[j]):
                slot.append(True)
            else:
                slot.append(False)
        else:

            if (cluster_floor >= floor[j]) & (cluster_ceiling <= ceiling[j]):
                slot.append(True)
            else:
                slot.append(False)

            # If it cannot be slotted in at that level in the hierarchy, descend
            # the hierarchy and see whether or not it can be linked to any of
            # the immediate antecedents of that cluster.
            if slot[j] == False:
                _linked_cluster = linked_clusters[j]
                while _linked_cluster.antecedent is not None:
                    _linked_cluster = _linked_cluster.antecedent
                    if _linked_cluster.antecedent == None:
                        _floor = None
                        _ceiling = _linked_cluster.descendants[0].merge_level
                        if (cluster_floor >= _floor) & (cluster_ceiling <= _ceiling):
                            slot[j] = True
                            linked_clusters[j] = _linked_cluster
                    else:
                        _ceiling = _linked_cluster.descendants[0].merge_level
                        if (cluster_ceiling <= _ceiling):
                            slot[j] = True
                            linked_clusters[j] = _linked_cluster

    # Now establish which cluster to return
    if np.all(slot):
        linked_clusters = [linked_clusters[0]]
    elif np.any(slot):
        idx = np.squeeze(np.where(np.array(slot) == True))
        if np.size(idx)==1:
            linked_clusters = [linked_clusters[idx]]
        else:
            if self.method == 1:
                linked_clusters = [linked_clusters[i] for i in idx]

                var = []
                for link in linked_clusters:
                    var = get_var(self, cluster, link, var)
                keepidx = np.squeeze(np.where(np.asarray(var) == min(np.asarray(var))))
                if keepidx.size != 1:
                    keepidx = keepidx[0]
                linked_clusters = [linked_clusters[keepidx]]
            else:
                linked_clusters = [linked_clusters[idx[0]]]
    else:
        linked_clusters = []

    return linked_clusters

def find_linked_clusters_multiple_antecessors(self, cluster, linked_clusters, antecessors):
    """
    Finds the correct slot if multiple antecessors identified.

    Notes
    -----

    If the linked_clusters belong to multiple antecessors, we want to establish
    the correct level to slot the cluster into with each tree.

    """

    # First generate a list of all linked clusters belonging to each hierarchy
    # then feed them to find_linked_clusters_single_antecessor one at a time
    num_antecessors = len(antecessors)
    clustlst = gen_clustlst(num_antecessors, linked_clusters, antecessors)

    _linked_clusters = []
    for j in range(num_antecessors):
        _linked_clusters_ = find_linked_clusters_single_antecessor(self, cluster, clustlst[j])
        _linked_clusters.append(_linked_clusters_)

    # This will return lists within a list. Flatten this.
    linked_clusters = [item for sublist in _linked_clusters for item in sublist]
    # Now check to see if any of the linked clusters are bud clusters.
    linked_clusters = bud_check(self, cluster, linked_clusters)

    return linked_clusters

def gen_clustlst(number_of_antecessors, linked_clusters, antecessors):
    """
    Returns a list of clusters
    """
    clustlst = []
    for j in range(number_of_antecessors):
        lst = []
        for k in range(len(linked_clusters)):
            if linked_clusters[k].antecessor == antecessors[j]:
                lst.append(linked_clusters[k])
        clustlst.append(lst)

    return clustlst

def bud_check(self, cluster, linked_clusters):
    """
    Returns linked clusters after checking for buds

    Notes
    -----

    When trying to slot clusters into the correct level of an already
    established hierarchy, bud clusters become a bit of an issue. Although we
    may get the cluster correct - any accompanying fledgling clusters could be
    linked at the wrong level of the hierarchy (see resolve_ambiguity). This
    method establishes:

    1) if there are any bud clusters linked to the cluster
    2) and if there are, it checks that all data points within the bud clusters
       can be linked to the same level in an established hierarchy as the
       cluster itself.

    If the above two statements are true then the cluster and the bud cluster
    will be merged - ensuring that they are linked to the same level in the
    hierarchy at a later stage. If the above statements are false then we
    simply ignore any linked buds and focus instead on already established
    clusters.

    """

    # Find any linked buds
    linked_buds = find_linked_buds(self, linked_clusters, cluster)

    # Remove these from linked_clusters
    for bud_cluster in linked_buds:
        linked_clusters.remove(bud_cluster)

    # If a cluster is linked to multiple buds, we may be able to merge these
    # buds and link all of them.
    if len(linked_buds) > 1:
        linked_bud = linked_buds.pop()
        # Before we can merge buds we need to check there are not instances
        # where we may link multiple spectral components located at the same
        # position.
        if self.method == 1:
            do_not_merge = check_components(self, linked_bud, linked_buds)
        else:
            do_not_merge = [False]
        if np.any(do_not_merge):
            # If there are any cases multiple components - focus on a single
            # bud only.
            linked_buds = [linked_bud]
        else:
            # Merge all buds.
            for bud_cluster in linked_buds:
                self.clusters.pop(bud_cluster.cluster_idx)
                self = update_leodis_array(self, linked_bud, -1)
                self = merge_into_cluster(self, linked_bud, bud_cluster)
            linked_buds = [linked_bud]

    # Now we want to work out what to return.
    antecessors = [link.antecessor for link in linked_clusters]
    antecessors = remdup_preserve_order(antecessors)
    antecessors = sorted(antecessors, key=get_cluster_idx, reverse=True)
    num_antecessors = len(antecessors)

    if not linked_clusters:
        # If there are no linked_clusters - return only the buds.
        linked_clusters = linked_buds
    elif not linked_buds:
        # If there are no linked_buds - return the linked clusters.
        linked_clusters = linked_clusters
    elif (len(linked_buds)==1) & (len(linked_clusters)>=1):
        # If there are both linked clusters and linked buds, we need to
        # establish if the bud can also be linked to the same level in the
        # hierarchy as the cluster (because they will all end up being merged).

        clustlst = gen_clustlst(num_antecessors, linked_clusters, antecessors)
        _linked_clusters = []
        for j in range(num_antecessors):
            # Find out which level in each identified hierarchy the bud can be
            # linked to.
            _linked_clusters_ = find_linked_clusters_single_antecessor(self, linked_buds[0], clustlst[j])
            _linked_clusters.append(_linked_clusters_)
        _linked_clusters = [item for sublist in _linked_clusters for item in sublist]

        _linked_clusters_ = linked_clusters
        _bud_linked_clusters_ = _linked_clusters

        # Check to see that the clusters the cluster is linked to and the
        # clusters the bud is linked to are identical.
        if set(_linked_clusters_) == set(_bud_linked_clusters_):
            linked_bud = linked_buds.pop()

            # Now we want to check to see if there are any cases of multiple
            # components between: 1) the cluster and the bud; 2) the bud and the
            # linked_clusters.
            if self.method == 1:
                check1 = check_components(self, cluster, [linked_bud])
                check2 = check_components(self, linked_bud, linked_clusters)
            else:
                check1 = [False]
                check2 = [False]

            # If both of these conditions are false - merge the cluster and bud.
            if not np.any([check1, check2]):
                self.clusters.pop(linked_bud.cluster_idx)
                self = update_leodis_array(self, linked_bud, -1)
                self = merge_into_cluster(self, cluster, linked_bud)
            linked_clusters = linked_clusters
        else:
            # If this isn't the case - forget about the bud and just return the
            # linked_clusters.
            linked_clusters = linked_clusters

    return linked_clusters

def remdup_preserve_order(lst):
    """
    Removes duplicates from a list but maintains the order see:
    https://www.peterbe.com/plog/uniqifiers-benchmark

    """
    val = set()
    val_add = val.add
    return [x for x in lst if not (x in val or val_add(x))]

def check_components(self, _cluster, _linked_clusters):
    """
    Checks to see if a position in one cluster can be found in another cluster.

    Notes
    -----

    Returns a boolean list of equivalent length to the number of linked clusters

    If any of the list values are true - you don't want to merge those two
    clusters.

    """

    do_not_merge = []
    for _link in _linked_clusters:
        boolval = []
        for j in range(_cluster.number_of_members):
            # Check all cluster components against those belonging to another
            # cluster
            multiple_components = (_cluster.cluster_members[0,j] == _link.antecessor.cluster_members[0,:]) & \
                                  (_cluster.cluster_members[1,j] == _link.antecessor.cluster_members[1,:])
            if np.any(multiple_components):
                boolval.append(True)
            else:
                boolval.append(False)
        if np.any(boolval):
            do_not_merge.append(True)
        else:
            do_not_merge.append(False)

    return do_not_merge

def multi_component_check(self, cluster, linked_clusters, re = False):
    """
    Additional connectivity constraint for method: PPV

    Prevents multiple components located at the same position being linked to
    the same cluster in the hierarchy

    Notes
    -----

    1. Check the position of the current data_point against the members of all
       the linked_clusters. Remove any linked clusters with members located at
       the same position.

    2. Check all linked clusters against each other. This process depends on how
       many linked_clusters there are:

       a. For a single pair of linked_clusters - check one against the other. If
          they cannot be merged together then select one to keep and one to
          remove.

          Keep the cluster that the data_point is most closely related to. This
          decision is based on how many standard deviations the data_point's
          properties lie from the mean of the cluster properties (all linking
          criteria are considered).

       b. If there are multiple pairs of linked_clusters - select the strongest
          link from each pair. Creating a new list of clusters. Then check to
          see if pairs within the new list can be merged. Continue to do this
          until all (or none) of the linked_clusters are able to be merged
          together.

    """

    do_not_merge = check_components(self, cluster, linked_clusters)
    if np.any(do_not_merge):
        IDx_do_not_merge = (np.where(np.asarray(do_not_merge) == True))
        linked_clusters = [k for l, k in enumerate(linked_clusters) if l not in IDx_do_not_merge[0]]

    if len(linked_clusters) > 1:
        var = []
        for link in linked_clusters:
            var = get_var(self, cluster, link, var)
        # First - get rid of any clusters for which the data_point's properties
        # sit > 3 sigma away from the mean.
        linked_clusters, var = remove_outliers(self, linked_clusters, var)

    if len(linked_clusters) > 1:

        pairs, do_not_merge = get_pairs(self, linked_clusters)

        if np.any(do_not_merge):

            IDx_do_not_merge = (np.where(np.asarray(do_not_merge) == True))

            if len(IDx_do_not_merge[0]) == 1:

                var = []
                for _cluster in linked_clusters:
                    var = get_var(self, cluster, link, var)
                keepidx = np.squeeze(np.where(np.asarray(var) == min(np.asarray(var))))
                if keepidx.size != 1:
                    keepidx = keepidx[0]
                linked_clusters = [linked_clusters[keepidx]]

            else:
                keep_clusters = linked_clusters
                Continue = False
                while not Continue:
                    if len(keep_clusters) > 1:
                        pairs, do_not_merge = get_pairs(self, keep_clusters)
                        if np.any(do_not_merge):
                            keep_clusters = []

                            for j in range(len(pairs)):

                                var = []
                                for link in pairs[j]:
                                    var = get_var(self, cluster, link, var)
                                keepidx = np.squeeze(np.where(np.asarray(var) == min(np.asarray(var))))
                                if keepidx.size != 1:
                                    keepidx = keepidx[0]
                                keep_clusters.append(pairs[j][keepidx])
                            keep_clusters = list(set(keep_clusters))

                        else:
                            Continue = True

                    elif len(keep_clusters)==1:
                        Continue = True

                linked_clusters = keep_clusters

    return linked_clusters

def get_pairs(self, linked_clusters):
    """
    Return pairs of linked clusters and establish which can be merged and which
    cannot

    """

    _linked_clusters = [_cluster.antecessor for _cluster in linked_clusters]
    pairs = [pair for pair in itertools.combinations(_linked_clusters, r=2)]
    do_not_merge = [False for pair in itertools.combinations(_linked_clusters, r=2)]
    paircount = 0
    for pair in itertools.combinations(_linked_clusters, r=2):
        cluster1 = pair[0]
        cluster2 = pair[1]
        cluster1coords = cluster1.cluster_members[0:2,:]
        cluster2coords = cluster2.cluster_members[0:2,:]
        sep = distance.cdist(np.transpose(cluster1coords), np.transpose(cluster2coords), 'euclidean')
        multiple_components = (sep == 0.0)
        if np.any(multiple_components):
            do_not_merge[paircount] = True
        paircount += 1

    return pairs, do_not_merge

def get_var(self, cluster, link, var):
    """
    Return pooled sig difference between data point and all linked_clusters

    Notes
    -----

    Is this correct? I'm summing the number of standard deviations in
    quaderature. Could maybe add a weighting to less important linking
    parameters?

    """

    pooled_var = 0.0
    for j in range(4, len(cluster.cluster_members[:,0])):
        if link.statistics[j-3][4] != 0.0:
            pooled_var += (abs(cluster.statistics[j-3][3]-link.statistics[j-3][2])/link.statistics[j-3][4])**2.
        else:
            pooled_var += 0.0
    pooled_var = np.sqrt(pooled_var)
    var.append(pooled_var)

    return var

def remove_outliers(self, linked_clusters, var):
    """
    Check to see whether the properties of the cluster under consideration are
    within 3 sigma of those of the linked_clusters

    """

    sig = 3.0
    threesig = ( np.asarray(var) > sig )
    remidx = np.squeeze(np.where(np.asarray(var) > sig ))

    keep_clusters = []
    keep_var = []
    for j in range(len(threesig)):
        if (threesig[j] == False):
            keep_clusters.append(linked_clusters[j])
            keep_var.append(var[j])

    linked_clusters = keep_clusters
    var = keep_var

    return linked_clusters, var

def add_to_cluster_dictionary(self, cluster):
    """
    Add a cluster to the cluster dictionary. Update the leodis array.

    """

    self.clusters[self.cluster_idx] = cluster
    self = update_leodis_array(self, cluster, cluster.cluster_idx)

    return self

def merge_into_cluster(self, linked_cluster, cluster, re = False):
    """
    Merge two clusters together. Update the leodis array.

    """

    if re == False:
        linked_cluster = merge_clusters(linked_cluster, cluster)
    else:
        _linked_cluster = linked_cluster
        while _linked_cluster.antecedent is not None:
            _linked_cluster = _linked_cluster.antecedent
            _linked_cluster = merge_clusters(_linked_cluster, cluster, branching=True)
        linked_cluster = merge_clusters(linked_cluster, cluster)
    self = update_leodis_array(self, cluster, linked_cluster.cluster_idx)

    return self

def resolve_ambiguity(self, linked_clusters, cluster, re = False):
    """
    Strategy employed if more than one linked cluster has been identified.

    Notes
    -----

    1. Check whether or not the linked clusters are independent according to the
       clustering criteria.

    2. What happens next depends on how many independent clusters there are:

        a) If there are no independent clusters - merge the cluster into one of
           the non-independent clusters.
        b) If there is a single independent cluster - merge the cluster into
           this one.
        c) If there are multiple independent clusters - form a branch between
           the clusters and a new level in the hierarchy.

    """

    # Establish whether or not the data point is in fact linked to any 'buds'.
    linked_buds = find_linked_buds(self, linked_clusters, cluster)

    # Remove buds from linked clusters.
    for bud_cluster in linked_buds:
        linked_clusters.remove(bud_cluster)

    if not linked_clusters:
        linked_cluster = linked_buds.pop()
        self = merge_into_cluster(self, linked_cluster, cluster, re = re)
    elif len(linked_clusters) == 1:
        linked_cluster = linked_clusters[0]
        self = merge_into_cluster(self, linked_cluster, cluster, re = re)
    else:
        self, linked_cluster = branching(self, linked_clusters, cluster, re = re)

    # Now merge any remaining buds into linked cluster
    for bud_cluster in linked_buds:
        self.clusters.pop(bud_cluster.cluster_idx)
        self = merge_into_cluster(self, linked_cluster, bud_cluster, re = re)

    return self

def update_leodis_array(self, cluster, value):
    """
    Updates the leodis_arr.

    """
    for j in range(0, len(cluster.cluster_indices)):
        self.leodis_arr[2, cluster.cluster_indices[j]] = value

    return self


def branching(self, linked_clusters, cluster, re = False):
    """
    Methodology for creating a new branch.
    """

    if re == False:
        # Form the branch
        self.clusters[cluster.cluster_idx] = cluster
        self = update_leodis_array(self, cluster, cluster.cluster_idx)
        linked_cluster = form_a_branch(cluster, descendants = linked_clusters)
    else:
        # If the conditions have been relaxed, we need to ensure we don't
        # create any redundant branches.
        _linked_clusters = [_cluster.antecessor for _cluster in linked_clusters]
        _linked_clusters = remove_branch(self, _linked_clusters, cluster)
        # Slot the cluster in at the correct level in the hierarchy.
        # Find the closest matching cluster
        var = []
        for link in linked_clusters:
            var = get_var(self, cluster, link, var)
        keepidx = np.squeeze(np.where(np.asarray(var) == min(np.asarray(var))))
        if keepidx.size != 1:
            keepidx = keepidx[0]
        _linked_cluster_ = linked_clusters[keepidx]
        _cluster_ = cluster
        _linked_cluster = _linked_cluster_
        # Merge downwards until loedis reaches the antecessor.
        while _linked_cluster.antecedent is not None:
            _linked_cluster = merge_clusters(_linked_cluster, _cluster_, branching=True)
            _linked_cluster = _linked_cluster.antecedent

        # Form the branch
        self.clusters[cluster.cluster_idx] = cluster
        self = update_leodis_array(self, cluster, cluster.cluster_idx)
        linked_cluster = form_a_branch(cluster, descendants = _linked_clusters)

    return self, linked_cluster

def remove_branch(self, linked_clusters, cluster):
    """
    Method used to check to see if a redundant branch is about to be created.

    Notes
    -----

    This is a problem related to the fact that we are relaxing the conditions
    for linking. Because the descent in intensity has been disrupted, we require
    that new data points are slotted in at the correct level in the hierarchy.
    However, this becomes complicated when we attempt to form a new branch and
    you can end up making redundant branches.

    If a data point has a value which is greater than the merge level of the
    descendants then the descendants will inherit the minimum value belonging to
    the two descendants as the merge level. If this newly formed cluster then
    form a branch with another cluster you can find a situation whereby multiple
    structures have the same merge level but are at different levels in the
    hierarchy - these branches are redundant and we want to get rid of them.

    """

    # Find all the branches in the current list of linked clusters
    _branches = [branch for branch in linked_clusters if branch.branch_cluster]
    _merge_levels_descendants = []
    _rembranch = []
    # Extract the merge levels of their descendants
    if len(_branches) >= 1.0:
        _merge_levels_descendants = [branch.descendants[0].merge_level for branch in _branches]
        _rembranch = [False for branch in _branches]

    # Identify the potential merge level of the linked_clusters
    _merge_levels = []
    for _cluster in linked_clusters:
        _merge_levels.append(_cluster.statistics[0][0])
    _merge_levels.append(cluster.statistics[0][0])
    _potential_merge_level = np.min(_merge_levels)

    # If the potential merge level of the linked_clusters is equal to the merge
    # level of any of the descendant branches' descendants then the intermediate
    # branch is effectively redundant.
    if len(_branches) >= 1.0:
        for i in range(len(_branches)):
            if _potential_merge_level == _merge_levels_descendants[i]:
                _rembranch[i] = True

    _rembranch_ = []
    for i in range(len(linked_clusters)):
        if linked_clusters[i].leaf_cluster == True:
            _rembranch_.append(False)
        else:
            idx = np.squeeze(np.where(linked_clusters[i]==np.asarray(_branches)))
            if np.size(idx) != 0.0:
                if _rembranch[idx] == True:
                    _rembranch_.append(True)
                else:
                    _rembranch_.append(False)

    _rembranch = _rembranch_
    # If there are any instances where _rembranch is true - this branch should
    # be removed and replaced with the branch descendants in linked_clusters.
    if np.any(_rembranch):
        _linked_clusters = [[item] for item in linked_clusters]
        for i in range(len(_rembranch)):
            if _rembranch[i] == True:
                self = correct_branching(self, _linked_clusters[i][0], cluster)
                _linked_clusters[i] = [descendant for descendant in linked_clusters[i].descendants]
            else:
                _linked_clusters[i] = [linked_clusters[i]]
        _linked_clusters = [item for sublist in _linked_clusters for item in sublist]
        linked_clusters = _linked_clusters

    return linked_clusters

def correct_branching(self, rembranch, cluster):
    """
    Removes redundant branch and merges data with cluster.

    Notes
    -----

    Here, because we are removing an already established branch from the
    hierarchy, what we want to do is find all the data points that are unique to
    that branch - and merge it with another cluster. All of these data points
    have already been slotted into the correct level in their respective
    hierarchies so all we have to do is merge them with cluster.

    """

    # Generate two sets - one containing all of the cluster_indices in rembranch
    # and another containing the cluster_indices in rembranch's descendants.
    set_rembranch = set(rembranch.cluster_indices)
    set_rembranch_descendants = set(rembranch.descendants[0].cluster_indices) | set(rembranch.descendants[1].cluster_indices)
    # The difference between these sets are the data points that are unique to
    # the branch we are trying to get rid of - merge these with cluster, after
    # which we can delete the branch.
    branch_indices = list(set_rembranch-set_rembranch_descendants)
    if np.size(branch_indices) != 0.0:
        for i in range(len(branch_indices)):
            idx = np.squeeze(np.where(rembranch.cluster_indices == branch_indices[i]))
            if np.size(idx)==1:
                cluster = merge_data(cluster, rembranch.cluster_members[:,idx])
            else:
                for j in range(np.size(idx)):
                    cluster = merge_data(cluster, rembranch.cluster_members[:,idx[j]])
    # Remove the branch and set corresponding data in leodis_arr to -1 (these
    # will be updated with the new branch indices during the merger).
    self.clusters.pop(rembranch.cluster_idx)
    idx = np.squeeze(np.where(self.leodis_arr[2,:] == rembranch.cluster_idx))
    if np.size(idx) != 0.0:
        if np.size(idx) == 1.0:
            self.leodis_arr[2, idx] = -1
        else:
            for j in range(np.size(idx)):
                self.leodis_arr[2, idx[j]] = -1

    return self

def find_linked_buds(self, linked_clusters, cluster):
    """
    Check whether or not the currently linked clusters satisfy the conditions to
    be classified as a leaf cluster

    """

    # Find any buds
    linked_buds = [check_cluster for check_cluster in linked_clusters \
                   if check_cluster.leaf_cluster and check_cluster.antecedent is None
                   and not independent_leaf_cluster(self, check_cluster, \
                                            linked_clusters, cluster)]

    return linked_buds

def independent_leaf_cluster(self, check_cluster, linked_clusters, cluster):
    """
    Method to determine whether or not a leaf cluster can be classed as an
    independent cluster

    """

    independence_check = [False, False, False]

    # Check the number of members in the cluster
    if check_cluster.number_of_members >= self.minnpix_cluster:
        independence_check[0] = True

    # Check to see if the peak of the cluster is at least min_height above the
    # current data point
    if ((check_cluster.statistics[0][1]-cluster.statistics[0][1]) >= self.min_height):
        independence_check[1] = True

    sep = []
    sep = [np.linalg.norm(check_cluster.peak_location-linked_cluster.peak_location) \
           for linked_cluster in linked_clusters]
    idx_too_close = np.squeeze(np.where(np.asarray(np.asarray(sep) == 0.0) == True))
    sep = [k for l, k in enumerate(sep) if l not in idx_too_close]

    if (all(np.asarray(sep) >= self.min_sep) is True):
        independence_check[2] = True

    # Are all conditions satisfied?
    independence_check = all(independence_check)

    return independence_check

def get_current_index(self, index):
    """
    Returns the index of the current data point from the original unassigned
    array
    """

    if self.method != 1:
        current_idx = np.where((self.unassigned_data[0,:]==self.unassigned_data_relax[0,index]) & \
                               (self.unassigned_data[1,:]==self.unassigned_data_relax[1,index]) & \
                               (self.unassigned_data[2,:]==self.unassigned_data_relax[2,index]) & \
                               (self.unassigned_data[3,:]==self.unassigned_data_relax[3,index]) & \
                               (self.unassigned_data[4,:]==self.unassigned_data_relax[4,index]))
    else:
        current_idx = np.where((self.unassigned_data[0,:]==self.unassigned_data_relax[0,index]) & \
                               (self.unassigned_data[1,:]==self.unassigned_data_relax[1,index]) & \
                               (self.unassigned_data[2,:]==self.unassigned_data_relax[2,index]) & \
                               (self.unassigned_data[3,:]==self.unassigned_data_relax[3,index]))

    current_idx = current_idx[0][0]

    return current_idx

def update_clusters(self):
    """
    Remove any clusters with number_of_members < self.minnpix_cluster, update_clusters
    the leodis_arr, and generate new unassigned_data from the bud clusters

    """

    self.unassigned_data_updated = None
    cluster_list = []
    cluster_indices = []
    bud_list = []
    bud_indices = []

    for cluster_index, cluster in self.clusters.iteritems():
        cluster_list.append(cluster)
        cluster_indices.append(cluster_index)
        if cluster.number_of_members < self.minnpix_cluster:
            bud_list.append(cluster)
            bud_indices.append(cluster_index)

    unassigned_data_updated = []
    for i in range(len(bud_indices)):
        self.clusters.pop(bud_indices[i])
        idx = np.squeeze(np.where(self.leodis_arr[2,:] == bud_indices[i]))
        self.leodis_arr[2,idx]=-1.0
        unassigned_data_updated.extend(np.transpose(bud_list[i].cluster_members))

    unassigned_data_updated = {(tuple(e)) for e in unassigned_data_updated}
    unassigned_data_updated = [list(x) for x in {(tuple(e)) for e in unassigned_data_updated}]
    unassigned_data_updated = np.transpose(unassigned_data_updated)

    sortidx = np.argsort(unassigned_data_updated[2,:])[::-1]
    self.unassigned_data_updated = unassigned_data_updated[:,sortidx]

    return self, cluster_list, cluster_indices

def get_relaxed_cluster_criteria(self, cluster_criteria_original):
    """
    Create new clustering criteria.

    """

    cluster_criteria = None
    # Get new clustering criteria
    if np.size(self.relax) == 1:
        cluster_criteria_relax = cluster_criteria_original+(cluster_criteria_original*self.relax)
        cluster_criteria = cluster_criteria_relax
    else:
        cluster_criteria_relax = cluster_criteria_original
        for j in range(np.size(self.relax)):
            cluster_criteria_relax[j] = cluster_criteria_original[j]+(cluster_criteria_original[j]*self.relax[j])
        cluster_criteria = cluster_criteria_relax

    return cluster_criteria

def check_components_test(self, _cluster, _linked_clusters):
    """
    Checks to see if a position in one cluster can be found in another cluster.
    Returns a boolean list of equivalent length to the number of linked clusters

    If any of the list values are true - you don't want to merge those two
    clusters.
    """

    do_not_merge = []
    for _link in _linked_clusters:
        boolval = []
        for j in range(_cluster.number_of_members):
            multiple_components = (_cluster.cluster_members[0,j] == _link.antecessor.cluster_members[0,:]) & \
                                  (_cluster.cluster_members[1,j] == _link.antecessor.cluster_members[1,:]) & \
                                  (_cluster.cluster_members[2,j] == _link.antecessor.cluster_members[2,:]) & \
                                  (_cluster.cluster_members[4,j] == _link.antecessor.cluster_members[4,:])
            if np.any(multiple_components):
                boolval.append(True)
            else:
                boolval.append(False)
        if np.any(boolval):
            do_not_merge.append(True)
        else:
            do_not_merge.append(False)

    return do_not_merge

def get_forest(self):

    _antecessors = []
    for key, cluster in self.clusters.iteritems():
        if cluster.leaf_cluster == True:
            _antecessors.append(cluster.antecessor)
    _antecessors = remdup_preserve_order(_antecessors)
    _antecessors = sorted(_antecessors, key=get_cluster_idx, reverse=True)

    _tree_idx = 0
    for antecessor in _antecessors:
        #print antecessor
        #check = check_components_test(self, antecessor, _antecessors)
        #print check
        tree = Tree(antecessor, idx = _tree_idx, leodis=self)
        self.forest[_tree_idx] = tree
        _tree_idx += 1

    #print self.forest[1].tree_members
    return self

def print_to_terminal(self, count, unassigned_array_length, method, re=False):
    """
    Prints some information to the terminal if verbose == True
    """

    if (re==False):

        print ''
        print 'Beginning analysis...'
        print ''
        print("leodis will look for clusters within {}% of the data: ".format((100 * unassigned_array_length / self.data[0,:].size)))
        print ''
        print("Method = {}".format(method))
        print("Max. Euclidean distance between linked data = {}".format(np.around(self.max_dist, decimals = 2)))
        print("Min. # of data points in a cluster = {}".format(np.around(self.minnpix_cluster, decimals = 0)))
        print("Min. height above the merge level = {}".format(np.around(self.min_height, decimals = 2)))
        print("Min. separation between clusters = {}".format(np.around(self.min_sep, decimals = 2)))

        if self.method == 0:
            if len(self.cluster_criteria) != 1:
                print ''
                print "Additional criteria: "
                for j in range(4, len(self.data[:,0])):
                    print("Max. absolute difference between column {} data = {}".format(j, np.around(self.cluster_criteria[j-3], decimals = 2)))
        if self.method == 1:
            print("Max. absolute velocity difference between linked data = {}".format(np.around(self.cluster_criteria[1], decimals = 2)))
            print ''
            if len(self.cluster_criteria) > 2:
                print "Additional criteria: "
                for j in range(5, len(self.data[:,0])):
                    print("Max. absolute difference between column {} data = {}".format(j, np.around(self.cluster_criteria[j-3], decimals = 2)))
        if self.method == 2:
            if len(self.cluster_criteria) != 1:
                print ''
                print "Additional criteria: "
                for j in range(4, len(self.data[:,0])):
                    print("Max. absolute difference between column {} data = {}".format(j, np.around(self.cluster_criteria[j-3], decimals = 2)))

        print 'Primary clustering...'
        print ''
        progress_bar = AnimatedProgressBar(end=unassigned_array_length, width=50, \
                                           fill='=', blank='.')

    else:
        relaxval = (100 * self.relax)
        if np.size(self.relax) == 1:
            print("Relaxing the linking constraints by {}%...".format(int(relaxval)))
            print ''
            print 'Secondary clustering...'
            print ''
        else:
            for j in range(np.size(self.relax)):
                print("Relaxing the linking constraint {} by {}%...".format(j+1, int(relaxval[j])))
            print ''
            print 'Secondary clustering...'
            print ''
        progress_bar = AnimatedProgressBar(end=unassigned_array_length, width=50, \
                                           fill='=', blank='.')

    return progress_bar

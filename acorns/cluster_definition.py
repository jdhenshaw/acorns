# Licensed under an MIT open source license - see LICENSE

"""
acorns - Agglomerative Clustering for ORgansing Nested Structures
Copyright (c) 2017 Jonathan D. Henshaw
CONTACT: j.d.henshaw@ljmu.ac.uk
"""

import numpy as np


class Cluster(object):

    def __init__(self, data_point, data_idx, idx=None, acorns=None):

        """

        Description
        -----------

        This is how clusters are defined and characterised.

        Parameters
        ----------

        Examples
        --------

        Notes
        -----

        """

        self._acorns      = acorns
        self._antecessor  = self
        self._antecedent  = None
        self._siblings    = None
        self._descendants = []
        self._merge_level = None
        self._cluster_idx  = idx

        # cluster point properties
        # data indices
        self.cluster_members = np.array([data_idx], dtype='i4')
        # cluster indices
        self.cluster_indices = np.array([idx], dtype='i4')
        # peak location
        self._peak_location = np.array([data_point[0],data_point[1]])
        # Set up a dictionary of important information. See statistics below
        self._statistics = {}
        #self._statistics[0] = [data_point[2],data_point[2],data_point[2],data_point[2],data_point[2]]
        self._statistics[0] = [data_point[2],data_point[2],data_point[2],data_point[2],0.0]

        # Add remaining attributes to stats dict
        for j in range(4, len(data_point)):
            #self._statistics[j-3] = [data_point[j],data_point[j],data_point[j],data_point[j],data_point[j]]
            self._statistics[j-3] = [data_point[j],data_point[j],data_point[j],data_point[j],0.0]


    @property
    def cluster_idx(self):
        """
        Returns cluster index
        """

        return self._cluster_idx

    @property
    def merge_level(self):
        """
        Returns the level at which this cluster because a branch

        """
        return self._merge_level

    @property
    def statistics(self):
        """
        Return dictionary of important statistics.

        Notes
        -----

        Each entry to the dictionary relates to a single observable quantity.
        Each entry is a list which contains the following information:

        min, max, mean, median, stddev

        """

        return self._statistics

    @property
    def peak_location(self):
        """
        Return the location of the peak value

        """

        return self._peak_location

    @property
    def number_of_members(self):
        """
        Return the number of cluster members.

        """

        return np.size(self.cluster_members)

    @property
    def leaf_cluster(self):
        """
        Return true if the present structure is a leaf (it has no descendants).

        """

        return not self.descendants

    @property
    def branch_cluster(self):
        """
        Return true if the present structure is a branch (it is not a leaf
        cluster).

        """

        return not self.leaf_cluster

    @property
    def descendants(self):
        """
        Return antecessor

        """

        return self._descendants

    @property
    def antecessor(self):
        """
        Return antecessor

        """

        return self._antecessor

    @property
    def antecedent(self):
        """
        Return antecessor

        """

        return self._antecedent

    @property
    def siblings(self):
        """
        Return siblings

        """

        return self._siblings

    def output_cluster_table(self, data, outputfile, format=None, headings=None ):
        """
        Generates an output table for a given cluster.

        Notes
        -----

        Default format is an ascii file with the same information as the acorns
        input array. However, if an array is provided using keyword "extended" then
        a table with additional information can be supplied (headings can also
        be provided with this keyword).

        If format "fits" is supplied output cluster table will produce a astropy
        table containing the results.

        """

        if format=="ascii":
            from .io import output_ascii
            return output_ascii(self, data, outputfile, headings=headings)
        elif format=="fits":
            from .io import output_fits
            return output_fits(self, data, outputfile, headings=headings)
        else:
            raise IOError("Please enter a valid output format (ascii, fits)")

    def reset_antecedent(self):

        self._antecedent = None

        return self

    def reset_antecessor(self):

        self._antecessor= None

        return self

    def __repr__(self):
        """
        Return a nice printable format for the object. This format will indicate
        if the current structure is a leaf_cluster or a branch_cluster, and give
        the cluster index.

        """

        if self.leaf_cluster:
            return "<< acorns cluster; type=leaf_cluster; cluster_index={0}; number_of_members={1} >>".format(self.cluster_idx, self.number_of_members)
        else:
            return "<< acorns cluster; type=branch_cluster; cluster_index={0}; number_of_members={1} >>".format(self.cluster_idx, self.number_of_members)

def _set_antecedent(self, descendants):
    """
    Set antecedent property of descendents to current branch.

    Notes
    -----

    We want the clusters to know who they are related to. An antecedent is the
    immediate parent of a cluster. So when branching set the antecedent property
    of all descendants in the current branch to the branch itself. Also set the
    antecedent value of the current branch to 'None' to indicate that it doens't
    have a parent.

    """

    for descendant in descendants:
        descendant._antecedent = self

    return self._antecedent

def _set_antecessor(self, descendants):
    """
    Set reference to largest related cluster.

    Notes
    -----

    We want the clusters to know who they are related to. The antecessor is the
    largest structure in the current family. Every time a new branch is formed
    the branch becomes the antecessor. However, we must descend the family tree
    and assign the antecessor property of all descendants (branch clusters or
    leaf clusters) to the current branch.

    """

    # Create a temporary list of descendants that will be updated
    new_descendants = descendants

    # Cycle through descendants looking for new descendants
    while (len(new_descendants) !=0 ):
        descendant_list = []
        # Loop over descendants
        for descendant in new_descendants:
            # Set the antecessor property to the current cluster level
            descendant._antecessor = self
            # Check to see if the current descendant has any descendants
            if (len(descendant.descendants) !=0 ):
                # If there are, add these to the descendant_list
                descendant_list.extend(descendant.descendants)
        # Once search for descendants has finished begin a new search based
        # on the descendant_list
        new_descendants = descendant_list

    return self._antecessor

def _set_siblings(self, descendants):
    """
    Returns siblings
    """

    for descendant in descendants:
        descendant._siblings = [cluster for cluster in descendants if cluster != descendant]

    return self._siblings

def _set_merge_level(self, descendants):
    """
    Sets the merge level of the leaves

    """

    self._merge_level = None

    mergevals = []
    for descendant in descendants:
        mergevals.append(descendant.statistics[0][0])

    for descendant in descendants:
        descendant._merge_level = np.min(np.asarray(mergevals))

    mergevals = None

    return self._merge_level

def merge_clusters(self, merge_cluster, data, branching = False):
    """
    Add descendant data points to a new branch

    """

    if branching == False:
        merge_cluster._cluster_idx  = self._cluster_idx

    # Merge cluster into the linked cluster
    self.cluster_members = np.concatenate(
            [self.cluster_members, merge_cluster.cluster_members])
    self.cluster_indices = np.concatenate([self.cluster_indices, \
                                      merge_cluster.cluster_indices])

    cluster_data = data[:, self.cluster_members]

    # Update the cluster statistics
    self._statistics[0] = [np.min(cluster_data[2,:]), \
                           np.max(cluster_data[2,:]),\
                           np.mean(cluster_data[2,:]),\
                           np.median(cluster_data[2,:]),\
                           np.std(cluster_data[2,:])]

    # Repeat for all quantities under consideration
    for j in range(4, len(cluster_data[:,0])):
        self._statistics[j-3] = [np.min(cluster_data[j,:]), \
                                 np.max(cluster_data[j,:]),\
                                 np.mean(cluster_data[j,:]),\
                                 np.median(cluster_data[j,:]),\
                                 np.std(cluster_data[j,:])]
    # Update the peak location
    peak_idx = np.argmax(cluster_data[2,:])
    self._peak_location = np.array([cluster_data[0,peak_idx], cluster_data[1,peak_idx]])

    return self

def merge_data(self, data_idx, un_idx, data):
    """
    Add data points to a cluster and update cluster properties

    """

    self.cluster_members = np.concatenate(
            [self.cluster_members, np.array([data_idx])])
    self.cluster_indices = np.concatenate([self.cluster_indices, np.array([un_idx])])

    cluster_data = data[:, self.cluster_members]

    # Update the cluster statistics
    self._statistics[0] = [np.min(cluster_data[2,:]), \
                           np.max(cluster_data[2,:]),\
                           np.mean(cluster_data[2,:]),\
                           np.median(cluster_data[2,:]),\
                           np.std(cluster_data[2,:])]

    # Repeat for all quantities under consideration
    for j in range(4, len(cluster_data[:,0])):
        self._statistics[j-3] = [np.min(cluster_data[j,:]), \
                                 np.max(cluster_data[j,:]),\
                                 np.mean(cluster_data[j,:]),\
                                 np.median(cluster_data[j,:]),\
                                 np.std(cluster_data[j,:])]

    # Update the peak location
    peak_idx = np.argmax(cluster_data[2,:])
    self._peak_location = np.array([cluster_data[0,peak_idx], cluster_data[1,peak_idx]])

    return self

def form_a_branch(self, data, descendants = []):
    """
    Convert the current cluster into a branch and update the information
    """

    self._descendants = descendants
    self._antecedent  = _set_antecedent(self, descendants)
    self._antecessor  = _set_antecessor(self, descendants)
    self._siblings    = _set_siblings(self, descendants)
    self._merge_level = _set_merge_level(self, descendants)

    # Merge descendants into branch and update the important information
    for descendant in descendants:
        self = merge_clusters(self, descendant, data, branching = True)

    return self

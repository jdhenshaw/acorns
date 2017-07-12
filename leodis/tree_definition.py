# Licensed under an MIT open source license - see LICENSE

"""
leodis - A heirarchical structure finding algorithm
Copyright (c) 2017 Jonathan D. Henshaw
CONTACT: j.d.henshaw@ljmu.ac.uk
"""

import numpy as np
import sys

class Tree(object):

    def __init__(self, _antecessor, idx=None, leodis=None):

        """

        Description
        -----------

        This is how individual trees are defined and characterised. Output can
        be plotted as a dendrogram

        Parameters
        ----------

        Examples
        --------

        Notes
        -----


        """

        self._leodis = leodis
        self._tree_idx = idx

        self._number_of_tree_members = None
        self._tree_members_idx = None
        self._tree_members = None
        self._trunk = None
        self._branches = None
        self._leaves = None
        self._sorted_leaves = None
        self._crown_width = None
        self._cluster_vertices = [[], []]
        self._horizontals = [[], []]

        self = _get_members(self, _antecessor)
        self = _sort_members(self)
        self = _dendrogram_positions(self)

    @property
    def tree_idx(self):
        """
        Returns tree index

        """

        return self._tree_idx

    @property
    def tree_members_idx(self):
        """
        Returns the indices of the tree members
        """

        return self._tree_members_idx

    @property
    def tree_members(self):
        """
        Returns tree members
        """

        return self._tree_members

    @property
    def number_of_tree_members(self):
        """
        Returns the number of tree members
        """

        return int(self._number_of_tree_members)

    @property
    def trunk(self):
        """
        Returns trunk

        """

        return self._trunk

    @property
    def branches(self):
        """
        Returns branches

        """

        return self._branches

    @property
    def leaves(self):
        """
        Returns leaves

        """

        return self._leaves

    @property
    def crown_width(self):
        """
        Returns maximum width of dendrogram crown

        """

        return len(self._leaves)

    @property
    def cluster_vertices(self):
        """
        Returns cluster vertices for plotting
        """

        return self._cluster_vertices

    @property
    def horizontals(self):
        """
        Returns cluster vertices for plotting
        """

        return self._horizontals

    def __repr__(self):
        """
        Return a nice printable format for the object. This format will indicate
        if the current structure is a leaf_cluster or a branch_cluster, and give
        the cluster index.

        """

        if self.trunk.leaf_cluster:
            return "<< leodis tree; type=leaf_cluster; tree_index={0}; number_of_tree_members={1} >>".format(self.tree_idx, self.number_of_tree_members)
        else:
            return "<< leodis tree; type=branch_cluster; tree_index={0}; number_of_tree_members={1} >>".format(self.tree_idx, self.number_of_tree_members)

def _get_members(self, _antecessor):
    """
    Returns information on the tree members
    """

    self._trunk = _antecessor
    self._branches = []
    self._leaves = []

    # Create a temporary list of descendants that will be updated
    new_descendants = _antecessor.descendants

    self._number_of_tree_members = 1.0
    self._number_of_leaves = 0.0
    self._number_of_branches = 0.0
    self._tree_members = [_antecessor]
    self._tree_members_idx = [_antecessor.cluster_idx]
    # Cycle through descendants looking for new descendants
    while (len(new_descendants) !=0 ):
        descendant_list = []
        # Loop over descendants
        for _descendant in new_descendants:
            self._number_of_tree_members+=1.0
            self._tree_members_idx.append(_descendant.cluster_idx)
            if _descendant.leaf_cluster:
                self._number_of_leaves += 1.0
                self._tree_members.append(_descendant)
                self._leaves.append(_descendant)
            else:
                self._number_of_branches += 1.0
                self._tree_members.append(_descendant)
                self._branches.append(_descendant)
            # Check to see if the current descendant has any descendants
            if (len(_descendant.descendants) !=0 ):
                # If there are, add these to the descendant_list
                descendant_list.extend(_descendant.descendants)
        # Once search for descendants has finished begin a new search based
        # on the descendant_list
        new_descendants = descendant_list

    if (_antecessor.leaf_cluster==True):
        self._number_of_leaves = 1.0
        self._leaves.append(_antecessor)

    return self

def _sort_members(self):
    """
    Sorts the tree members.

    Notes
    -----

    Here we want to sort the leaf clusters such that we can plot them as a
    dendrogram. The method starts with the brightest leaf in a tree and then
    descends the hierarchy checking for leaf siblings along the way.

    """

    # Initial sorting
    leaf_flux = [leaf.statistics[0][1] for leaf in self.leaves]
    sort_idx = np.argsort(np.asarray(leaf_flux))
    _leaves = np.array(self.leaves)
    _sorted_peak_leaves = list(_leaves[sort_idx])
    number_of_leaves = len(self.leaves)

    self._sorted_leaves = []
    while len(_sorted_peak_leaves) != 0.0:
        # Start with the brightest
        cluster = _sorted_peak_leaves.pop()
        self._sorted_leaves.append(cluster)
        # Now descend
        while cluster.antecedent is not None:
            siblings = cluster.siblings
            for sibling in siblings:
                # If the sibling is a leaf add it to the sorted list
                if sibling.leaf_cluster:
                    found_sibling = (np.asarray(_sorted_peak_leaves) == sibling)
                    if np.any(found_sibling):
                        idx = np.squeeze(np.where(found_sibling == True))
                        if np.size(idx) == 1:
                            self._sorted_leaves.append(_sorted_peak_leaves[idx])
                            _sorted_peak_leaves.pop(idx)
                        else:
                            for j in range(np.size(idx)):
                                self._sorted_leaves.append(_sorted_peak_leaves[idx[j]])
                                _sorted_peak_leaves.pop(idx[j])
                # If however, the sibling is a branch we need to ascend that
                # branch to get the order correct
                else:
                    new_clusters = [sibling]
                    while len(new_clusters) != 0:
                        cluster_list = []

                        for _cluster in new_clusters:
                            # Search upwards for descendants - if descendants
                            # are found - these will form the basis for new
                            # searches
                            if len(_cluster.descendants) != 0:
                                for _descendant in _cluster.descendants:
                                    cluster_list.append(_descendant)
                            # If leaves are identified then add them to the
                            # sorted list
                            else:
                                found_leaf = (np.asarray(_sorted_peak_leaves) == _cluster)
                                if np.any(found_leaf):
                                    idx = np.squeeze(np.where(found_leaf == True))
                                    self._sorted_leaves.append(_sorted_peak_leaves[idx])
                                    _sorted_peak_leaves.pop(idx)
                        new_clusters = cluster_list
            cluster = cluster.antecedent

    return self

def _dendrogram_positions(self):

    # Make lines for plotting a dendrogram
    # x locations of the leaf clusters
    x_loc = -1.*np.ones(len(self.tree_members))
    x_loc = list(x_loc)
    for j in range(len(self.tree_members)):
        idx = np.where(np.asarray(self._sorted_leaves) == self.tree_members[j])
        if np.size(idx) != 0.0:
            x_loc[j] = idx[0][0]+1.0

    # Find x_locations of clusters
    while np.any(np.asarray(x_loc) == -1.0):
        for i in range(len(x_loc)):
            if x_loc[i] != -1.0:
                _sibling_pos = [x_loc[i]]
                if self.tree_members[i].siblings is not None:
                    for sibling in self.tree_members[i].siblings:
                        found_sibling = (np.asarray(self.tree_members) == sibling)
                        idx = np.where(found_sibling == True)
                        for j in range(np.size(idx)):
                            _sibling_pos.append(x_loc[idx[0][j]])
                    if not np.any(np.asarray(_sibling_pos) == -1.0):
                        x_loc_add = np.mean(_sibling_pos)
                        idx = np.squeeze(np.where(np.asarray(self.tree_members) == self.tree_members[i].antecedent))
                        x_loc[idx] = x_loc_add

    # Generate lines for dendrogram
    for i in range(len(self.tree_members)):
        self._cluster_vertices[0].append(np.array([x_loc[i], x_loc[i]]))
        if self.tree_members[i] == self.trunk:
            if len(self.trunk.descendants) != 0.0:
                self._cluster_vertices[1].append(np.array([self.trunk.statistics[0][0], self.trunk.descendants[0].merge_level]))
                # find the descendants positions in x_loc
                x_loc_descendants = []
                for descendant in self.tree_members[i].descendants:
                    found_descendant = (np.asarray(self.tree_members) == descendant)
                    idx = np.where(found_descendant == True)
                    for j in range(np.size(idx)):
                        x_loc_descendants.append(x_loc[idx[0][j]])
                range_x = np.ptp(x_loc_descendants)
                self._horizontals[0].append(np.array([np.min(np.asarray(x_loc_descendants)) ,np.min(np.asarray(x_loc_descendants))+range_x]))
                self._horizontals[1].append(np.array([self.trunk.descendants[0].merge_level, self.trunk.descendants[0].merge_level]))
            else:
                self._cluster_vertices[1].append(np.array([self.trunk.statistics[0][0], self.trunk.statistics[0][1]]))
                self._horizontals[0].append(np.array([0.0,0.0]))
                self._horizontals[1].append(np.array([0.0,0.0]))
        elif self.tree_members[i].leaf_cluster == True:
            self._cluster_vertices[1].append(np.array([self.tree_members[i].merge_level, self.tree_members[i].statistics[0][1]]))
            self._horizontals[0].append(np.array([0.0,0.0]))
            self._horizontals[1].append(np.array([0.0,0.0]))
        else:
            self._cluster_vertices[1].append(np.array([self.tree_members[i].merge_level, self.tree_members[i].descendants[0].merge_level]))
            # find the descendants positions in x_loc
            x_loc_descendants = []
            for descendant in self.tree_members[i].descendants:
                found_descendant = (np.asarray(self.tree_members) == descendant)
                idx = np.where(found_descendant == True)
                for j in range(np.size(idx)):
                    x_loc_descendants.append(x_loc[idx[0][j]])
            range_x = np.ptp(x_loc_descendants)
            self._horizontals[0].append(np.array([np.min(np.asarray(x_loc_descendants)) ,np.min(np.asarray(x_loc_descendants))+range_x]))
            self._horizontals[1].append(np.array([self.tree_members[i].descendants[0].merge_level, self.tree_members[i].descendants[0].merge_level]))

    #print(self.horizontals)
    #sys.exit()

    return self

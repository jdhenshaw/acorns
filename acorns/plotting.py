# Licensed under an MIT open source license - see LICENSE

"""
acorns - Agglomerative Clustering for ORgansing Nested Structures
Copyright (c) 2017 Jonathan D. Henshaw
CONTACT: j.d.henshaw@ljmu.ac.uk
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors

def plot_scatter(self):

    fig = plt.figure(figsize=( 8.0, 8.0))

    if self.method == 0:
        ax = fig.add_subplot(111)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.scatter(self.data[0,:], self.data[1,:], marker='o', s=2., c='black',linewidth=0., alpha=0.2)

        count=0.0
        _antecessors = []
        for cluster in self.clusters:
            if self.clusters[cluster] == self.clusters[cluster].antecessor:
                _antecessors.append(self.clusters[cluster].antecessor)
                count+=1.0

        colour=iter(cm.rainbow(np.linspace(0,1,count)))
        for ant in _antecessors:
            c=next(colour)
            if ant.leaf_cluster:
                ax.scatter(ant.cluster_members[0,:], ant.cluster_members[1,:], \
                           marker='o', s=3., c='black',linewidth=0, alpha=0.7)
                ax.scatter(ant.cluster_members[0,:], ant.cluster_members[1,:], ant.cluster_members[4,:], \
                           marker='o', s=10., c='None', edgecolors = c ,alpha=0.9, depthshade=False, linewidth = 0.8)
            else:
                ax.scatter(ant.cluster_members[0,:], ant.cluster_members[1,:], \
                           marker='o', s=3., c='black',linewidth=0, alpha=0.7)
                ax.scatter(ant.cluster_members[0,:], ant.cluster_members[1,:], \
                           marker='o', s=10., c='None', edgecolors = c ,alpha=0.9, depthshade=False, linewidth = 0.8)

        plt.draw()
        plt.show()

    if self.method == 1:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.scatter(self.data[0,:], self.data[1,:], self.data[4,:], marker='o', s=2., c='black',linewidth=0., alpha=0.2)

        count=0.0
        _antecessors = []
        for cluster in self.clusters:
            if self.clusters[cluster] == self.clusters[cluster].antecessor:
                _antecessors.append(self.clusters[cluster].antecessor)
                count+=1.0

        colour=iter(cm.rainbow(np.linspace(0,1,count)))
        for ant in _antecessors:
            c=next(colour)
            if ant.leaf_cluster:
                ax.scatter(ant.cluster_members[0,:], ant.cluster_members[1,:], ant.cluster_members[4,:], \
                           marker='o', s=3., c='black',linewidth=0, alpha=0.7)
                ax.scatter(ant.cluster_members[0,:], ant.cluster_members[1,:], ant.cluster_members[4,:], \
                           marker='o', s=10., c='None', edgecolors = c ,alpha=0.9, depthshade=False, linewidth = 0.8)
            else:
                ax.scatter(ant.cluster_members[0,:], ant.cluster_members[1,:], ant.cluster_members[4,:], \
                           marker='o', s=3., c='black',linewidth=0, alpha=0.7)
                ax.scatter(ant.cluster_members[0,:], ant.cluster_members[1,:], ant.cluster_members[4,:], \
                           marker='o', s=10., c='None', edgecolors = c ,alpha=0.9, depthshade=False, linewidth = 0.8)

        plt.draw()
        plt.show()

    if self.method==2:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.scatter(self.data[0,:], self.data[1,:], self.data[2,:], marker='o', s=2., c='black',linewidth=0., alpha=0.2)

        count=0.0
        _antecessors = []
        for cluster in self.clusters:
            if self.clusters[cluster] == self.clusters[cluster].antecessor:
                _antecessors.append(self.clusters[cluster].antecessor)
                count+=1.0

        colour=iter(cm.rainbow(np.linspace(0,1,count)))
        for ant in _antecessors:
            c=next(colour)
            if ant.leaf_cluster:
                ax.scatter(ant.cluster_members[0,:], ant.cluster_members[1,:], ant.cluster_members[2,:], \
                           marker='o', s=3., c='black',linewidth=0, alpha=0.7)
                ax.scatter(ant.cluster_members[0,:], ant.cluster_members[1,:], ant.cluster_members[2,:], \
                           marker='o', s=10., c='None', edgecolors = c ,alpha=0.9, depthshade=False, linewidth = 0.8)
            else:
                ax.scatter(ant.cluster_members[0,:], ant.cluster_members[1,:], ant.cluster_members[2,:], \
                           marker='o', s=3., c='black',linewidth=0, alpha=0.7)
                ax.scatter(ant.cluster_members[0,:], ant.cluster_members[1,:], ant.cluster_members[2,:], \
                           marker='o', s=10., c='None', edgecolors = c ,alpha=0.9, depthshade=False, linewidth = 0.8)

        plt.draw()
        plt.show()

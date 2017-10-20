import numpy as np
import matplotlib.pyplot as plt
from leodis import Leodis
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
import sys


datadirectory =  './'
datafilename =  datadirectory+'fits_final_pdbi.dat'

# Load in data
dataarr    = np.loadtxt(datafilename)
# Data is organised as follows: x, y, peak intensity, error on peak intensity,
# velocity, FWHM linewidth, rms noise
dataarr = np.array([dataarr[:,0],dataarr[:,1],dataarr[:,2],dataarr[:,3], dataarr[:,4], dataarr[:,6], dataarr[:,8]])
dataarr_leodis = np.array([dataarr[0,:],dataarr[1,:],dataarr[2,:],dataarr[3,:], dataarr[4,:], dataarr[5,:]])

# The input to leodis therefore should be x,y,i,sig_i,vel

filename = 'example.leodis'
L = Leodis.load_from(datadirectory+filename)

fig   = plt.figure(figsize=( 8.0, 8.0))
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim([41,49])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Velocity  [km/s]')

ax.scatter(dataarr_leodis[0,:], dataarr_leodis[1,:], dataarr_leodis[4,:], marker='o', s=2., c='black',linewidth=0., alpha=0.2)

# Generate a new colour for each trunk
n = len(L.forest)
colour=iter(cm.rainbow(np.linspace(0,1,n)))
#colour=iter(cm.rainbow(np.linspace(0,1,5)))

# Toggle comments to plot just hierarchical structures
for tree in L.forest:
    c=next(colour)
    if L.forest[tree].trunk.leaf_cluster:
        #pass
        #c=next(colour)
        ax.scatter(dataarr_leodis[0, L.forest[tree].trunk.cluster_members], dataarr_leodis[1,L.forest[tree].trunk.cluster_members], dataarr_leodis[4,L.forest[tree].trunk.cluster_members], \
                   marker='o', s=3., c='black',linewidth=0, alpha=0.7)
        ax.scatter(dataarr_leodis[0, L.forest[tree].trunk.cluster_members], dataarr_leodis[1,L.forest[tree].trunk.cluster_members], dataarr_leodis[4,L.forest[tree].trunk.cluster_members], \
                   marker='o', s=10., c='None', edgecolors = c ,alpha=0.9, depthshade=False, linewidth = 0.8)
    else:
        #c=next(colour)
        #pass
        n = len(L.forest[tree].leaves)
        col=iter(cm.rainbow(np.linspace(0,1,n)))
        ax.scatter(dataarr_leodis[0, L.forest[tree].trunk.cluster_members], dataarr_leodis[1,L.forest[tree].trunk.cluster_members], dataarr_leodis[4,L.forest[tree].trunk.cluster_members], \
                   marker='o', s=3., c='black',linewidth=0, alpha=0.7)
        ax.scatter(dataarr_leodis[0, L.forest[tree].trunk.cluster_members], dataarr_leodis[1,L.forest[tree].trunk.cluster_members], dataarr_leodis[4,L.forest[tree].trunk.cluster_members], \
                   marker='o', s=10., c='None', edgecolors = c ,alpha=0.9, depthshade=False, linewidth = 0.8)
        #for leaf in L.forest[tree].leaves:
        #    c=next(col)
            #pass
            #ax.scatter(leaf.cluster_members[0,:], leaf.cluster_members[1,:], leaf.cluster_members[4,:], \
            #           marker='o', s=10., c=c, edgecolors = 'k',alpha=1.0, depthshade=False, linewidth = 0.1)

ax.azim = 180
ax.elev = 0

plt.show()

fig   = plt.figure(figsize=( 8.0, 8.0))
ax = fig.add_subplot(111)
ax.set_xlim([-1,50])
ax.set_ylim([-0.1,3])
ind = 0

# Generate a new colour for each trunk
n = len(L.forest)
colour=iter(cm.rainbow(np.linspace(0,1,n)))
count = 0.0
for tree in L.forest:
    c=next(colour)
    if L.forest[tree].trunk.leaf_cluster:
        pass#c=next(colour)
    else:
        pass#c=next(colour)
    for j in range(len(L.forest[tree].tree_members)):
        if L.forest[tree].tree_members[j] == L.forest[tree].trunk:
            ax.plot(L.forest[tree].cluster_vertices[0][j]+count, np.array([(np.mean(dataarr_leodis[3,:])), L.forest[tree].cluster_vertices[1][j][0]]), 'k:')

        if L.forest[tree].trunk.leaf_cluster:
            ax.plot(L.forest[tree].cluster_vertices[0][j]+count, L.forest[tree].cluster_vertices[1][j], c=c)
            ax.plot(L.forest[tree].horizontals[0][j]+count, L.forest[tree].horizontals[1][j], c=c)
        else:
            ax.plot(L.forest[tree].cluster_vertices[0][j]+count, L.forest[tree].cluster_vertices[1][j], c=c)
            ax.plot(L.forest[tree].horizontals[0][j]+count, L.forest[tree].horizontals[1][j], c=c)
    count+=len(L.forest[tree].leaves)


plt.show()

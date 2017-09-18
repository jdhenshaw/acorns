import numpy as np
import matplotlib.pyplot as plt
from leodis import Leodis
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
from astropy.io import fits
import sys

datadirectory =  './'
datafilename =  datadirectory+'G035_cont.fits'

# Load Continuum data
hdu   = fits.open(datafilename)
header = hdu[0].header
data  = hdu[0].data
hdu.close()
data = np.squeeze(data)

rmsnoise = 7.e-5

dataarr = np.zeros((4,len(data[1,:])*len(data[:,0])))
noisearr = rmsnoise*np.ones(len(data[1,:])*len(data[:,0]))

for i in range(len(data[1,:])):
    for j in range(len(data[:,0])):
        dataarr[0,j+i*len(data[:,0])]=i
        dataarr[1,j+i*len(data[:,0])]=j
        dataarr[2,j+i*len(data[:,0])]=data[j,i]
        dataarr[3,j+i*len(data[:,0])]=noisearr[j+i*len(data[:,0])]

dataarr_leodis = dataarr
# The input to leodis therefore should be x,y,i,sig_i,vel

filename = 'example_2d.leodis'
L = Leodis.load_from(datadirectory+filename)

fig   = plt.figure(figsize=( 8.0, 8.0))
ax = fig.add_subplot(111)
ax.set_xlabel('X')
ax.set_ylabel('Y')

ax.scatter(dataarr_leodis[0,:], dataarr_leodis[1,:], marker='o', s=2., c='black',linewidth=0., alpha=0.2)

# Generate a new colour for each trunk
n = len(L.forest)
colour=iter(cm.rainbow(np.linspace(0,1,n)))
#colour=iter(cm.rainbow(np.linspace(0,1,3)))

# Toggle comments to plot just hierarchical structures
for tree in L.forest:
    c=next(colour)
    if L.forest[tree].trunk.leaf_cluster:
        #pass
        #c=next(colour)
        ax.scatter(dataarr_leodis[0, L.forest[tree].trunk.cluster_members], dataarr_leodis[1,L.forest[tree].trunk.cluster_members], \
                   marker='o', s=3., c='black',linewidth=0, alpha=0.7)
        ax.scatter(dataarr_leodis[0, L.forest[tree].trunk.cluster_members], dataarr_leodis[1,L.forest[tree].trunk.cluster_members], \
                   marker='o', s=10., c='None', edgecolors = c ,alpha=0.9, linewidth = 0.8)
    else:
        #c=next(colour)
        #pass
        ax.scatter(dataarr_leodis[0, L.forest[tree].trunk.cluster_members], dataarr_leodis[1,L.forest[tree].trunk.cluster_members], \
                   marker='o', s=3., c='black',linewidth=0, alpha=0.7)
        ax.scatter(dataarr_leodis[0, L.forest[tree].trunk.cluster_members], dataarr_leodis[1,L.forest[tree].trunk.cluster_members], \
                   marker='o', s=10., c='None', edgecolors = c ,alpha=0.9, linewidth = 0.8)
        n = len(L.forest[tree].leaves)
        col=iter(cm.viridis(np.linspace(0,1,n)))
        for leaf in L.forest[tree].leaves:
            c=next(col)
            #pass
            ax.scatter(dataarr_leodis[0,leaf.cluster_members], dataarr_leodis[1,leaf.cluster_members], \
                       marker='o', s=10., c=c, edgecolors = 'k',alpha=1.0, linewidth = 0.1)

ax.azim = 180
ax.elev = 0

plt.show()

fig   = plt.figure(figsize=( 8.0, 8.0))
ax = fig.add_subplot(111)
ax.set_xlim([-1,40])
ax.set_ylim([-0.0001,0.0015])
ind = 0

# Generate a new colour for each trunk
n = len(L.forest)
colour=iter(cm.rainbow(np.linspace(0,1,n)))
count = 0.0
for tree in L.forest:
    if L.forest[tree].trunk.leaf_cluster:
        c=next(colour)
    else:
        c=next(colour)
    for j in range(len(L.forest[tree].tree_members)):
        if L.forest[tree].tree_members[j] == L.forest[tree].trunk:
            ax.plot(L.forest[tree].cluster_vertices[0][j]+count, np.array([(np.mean(dataarr_leodis[3,:])), L.forest[tree].cluster_vertices[1][j][0]]), 'k:')

        if L.forest[tree].trunk.leaf_cluster:
            ax.plot(L.forest[tree].cluster_vertices[0][j]+count, L.forest[tree].cluster_vertices[1][j], c=c)
            ax.plot(L.forest[tree].horizontals[0][j]+count, L.forest[tree].horizontals[1][j], c=c)
        else:
            ax.plot(L.forest[tree].cluster_vertices[0][j]+count, L.forest[tree].cluster_vertices[1][j], c=c)
            ax.plot(L.forest[tree].horizontals[0][j]+count, L.forest[tree].horizontals[1][j], c=c)
            num = len(L.forest[tree].leaves)
            col=iter(cm.viridis(np.linspace(0,1,num)))
            for i in range(len(L.forest[tree].tree_members)):
                if len(L.forest[tree].tree_members[i].descendants) == 0:
                    cc=next(col)
                    ax.plot(L.forest[tree].cluster_vertices[0][i]+count, L.forest[tree].cluster_vertices[1][i], c=cc,lw=2)

    count+=len(L.forest[tree].leaves)


plt.show()

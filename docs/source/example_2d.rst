Example - ``acorns`` clustering in 2D
=====================================

Here is a simple example for running ``acorns`` in 2D, e.g. dust continuum data.
This isn't `really` what ``acorns`` was designed for, but it can do it anyway,
so here's how...

The data used for this example can be downloaded
`here <https://github.com/jdhenshaw/acorns/blob/master/examples/G035_cont.fits>`_.
For reference, this is a 3.2mm dust continuum image taken using the Plateau de Bure
Interferometer (now NOEMA). You can find more details about the data in
`Henshaw et al. (2016b) <http://adsabs.harvard.edu/abs/2016MNRAS.463..146H>`_.

Start by loading in the data::

  import numpy as np
  from acorns import Acorns
  import sys
  import os
  from astropy.io import fits

  datadirectory =  './'
  datafilename =  datadirectory+'G035_cont.fits'

  # Load Continuum data
  hdu   = fits.open(datafilename)
  header = hdu[0].header
  data  = hdu[0].data
  hdu.close()
  data = np.squeeze(data)

  # Noise level of the continuum
  rmsnoise = 7.e-5

``acorns`` works by reading in a table. Because we have a 2D image here, in the
case of our continuum image, we first of all just want to flatten this::

  # Create the acorns table
  x = np.arange(np.size(data[0,:]))
  y = np.arange(np.size(data[:,0]))
  xx,yy = np.meshgrid(x,y)
  # Flatten 2D arrays
  xx = xx.flatten(order='F')
  yy = yy.flatten(order='F')
  data = data.flatten(order='F')
  # Create an equivlent length noise array
  noisearr = np.ones(len(xx))*rmsnoise
  # This is the array that will be fed to acorns
  dataarr_acorns = np.array([xx,yy,data,noisearr])

Now set up the input parameters::

  # Basic information required for clustering
  pixel_size = 1.0
  min_radius = 1.7 # Ensures 9 pixels defines the smallest structure identified
  min_height = 3.0*noisearr[0] # Clusters have to be at least this value above the merge level

  # Generate the cluster_criteria - In this instance we are only going to use
  # a single linking length - i.e. the distance between two pixels.
  cluster_criteria = np.array([min_radius])
  # Relax criteria - In the case of continuum data we don't need to relax the
  # linking constraints as all pixels are adjacent.
  relax = np.array([0])
  # Stopping criteria - determines when the algorithm will stop - this is a
  # multiple of the rms value
  stop = 3.

Now call ``acorns``::

  # Call acorns
  A = Acorns.process( dataarr_acorns, cluster_criteria, method = "PP", \
                      min_height = min_height, pixel_size = pixel_size, \
                      relax=relax, stop = stop, verbose=True )
  A.save_to(datadirectory+'example_2d.acorn')

If verbose=True, then the output in the terminal should look like this::

  Beginning analysis...

  acorns will look for clusters within 1.66015625% of the data:

  Method = PP
  Max. Euclidean distance between linked data = 1.48492
  Min. number of data points in a cluster = 9
  Min. height above the merge level = 0.0
  Min. separation between clusters = 2.96985
  Primary clustering...

  [==================================================>] 100%

  Making second pass...

  [==================================================>] 100%

  Generating forest...

  [==================================================>] 100%

  acorns took 2.0 seconds for completion.
  Primary clustering took 2.0 seconds for completion.

  acorns found a total of 48 clusters.

  A total of 2176 data points were used in the search.
  A total of 2038.0 data points were assigned to clusters.
  A total of 138 data points remain unassigned to clusters.

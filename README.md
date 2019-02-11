<p align="center">
<img src="docs/source/ACORNS_LOGO.jpg"  alt="" width = "450" />
</p>

About
=====

Agglomerative Clustering for ORganising Nested Structures

Installing ``acorns``
=======================

Requirements
------------

* [Python](http://www.python.org>) 2.6 or later (Python 3.x is supported)
* [Numpy](http://www.numpy.org) 1.13.3 or later
* [Scipy](http://www.scipy.org/) 1.0.1 or later
* [Matplotlib](http://matplotlib.org/) 2.2.2 or later
* [Astropy](http://www.astropy.org>) 3.0.2 or later

Installation
------------

(Available soon - stick to developer version for now - see below)

To install the latest stable release, you can type::

    pip install acorns

or you can download the latest tar file from
[PyPI](https://pypi.python.org/pypi/acorns) and install it using::

    python setup.py install

Developer version
-----------------

If you want to install the latest developer version, you
can do so using github::

    git clone https://github.com/jdhenshaw/acorns.git
    cd acorns
    python setup.py install

You may need to add the ``--user`` option to the last line if you do not have
root access.

Reporting issues and getting help
=================================

Please help to improve this package by reporting issues via [GitHub]
(https://github.com/jdhenshaw/acorns/issues). Alternatively, you can get in
touch [here](mailto:jonathan.d.henshaw@gmail.com).

Developers
==========

This package was developed by:

* Jonathan Henshaw

Contributors include:

* Vlas Sokolov
* Adam Ginsburg

Citing ``acorns``
===================

If you make use of this package in a publication, please consider the following
acknowledgement...

```
@ARTICLE{henshaw19,
   author = {{Henshaw}, J.~D. and {Ginsburg}, A. and {Haworth}, T.~J. and
	{Longmore}, S.~N. and {Kruijssen}, J.~M.~D. and {Mills}, E.~A.~C. and
	{Sokolov}, V. and {Walker}, D.~L. and {Barnes}, A.~T. and {Contreras}, Y. and
	{Bally}, J. and {Battersby}, C. and {Beuther}, H. and {Butterfield}, N. and
	{Dale}, J.~E. and {Henning}, T. and {Jackson}, J.~M. and {Kauffmann}, J. and
	{Pillai}, T. and {Ragan}, S. and {Riener}, M. and {Zhang}, Q.
	},
    title = "{'The Brick' is not a brick: A comprehensive study of the structure and dynamics of the Central Molecular Zone cloud G0.253+0.016}",
  journal = {arXiv e-prints},
archivePrefix = "arXiv",
   eprint = {1902.02793},
 keywords = {Astrophysics - Astrophysics of Galaxies},
     year = 2019,
    month = feb,
   adsurl = {http://adsabs.harvard.edu/abs/2019arXiv190202793H},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

```

Please also consider acknowledgements to the required packages in your work.

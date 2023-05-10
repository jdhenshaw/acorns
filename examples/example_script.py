from acorns import Acorns

datadir='./'
outputdir='./'
filename='fits_final_pdbi.dat' # note we remove the fits extension

config_file=Acorns.run_setup(filename, datadir, outputdir=outputdir)
acorn=Acorns.process(config_file)

from acorns import Acorns

datadir='./'
outputdir='./'
filename='fits_final_pdbi.dat' 

acorn=Acorns.run_setup(filename, datadir, outputdir=outputdir)
acorn=Acorns.initialize(acorn)

print(acorn.cluster_config.cluster_method)
#acorn=Acorns.process(config_file)

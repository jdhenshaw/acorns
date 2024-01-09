# This code is licensed under the MIT License.
# For more details, see the LICENSE file.

import numpy as np
import os

class Setup:
    """
    Class to setup the acorns project

    """
    def __init__(self):
        # Initialize any necessary attributes here
        self.filename = None
        self.datadir = None
        self.outputdir = None
        self.datatype = None
        self.verbose = True
        self.autosave = True
        self.n_jobs = 3
        self.min_npoints = 0.0
        self.min_height = 0.0
        self.stop_value = 0.0
        self.cluster_method = None
        self.cluster_spatial_idx = None
        self.cluster_spatial_metric = None
        self.cluster_spatial_thresh = None
        self.cluster_feature_idx = None
        self.cluster_feature_metric = None
        self.cluster_feature_thresh = None
        self.cluster_stopcrit_idx = None
        self.cluster_sortcrit_idx = None
        self.reverse = False

    def determine_input(self, filename):
        """
        Determine the input datatype

        Parameters
        ----------
        filename : string
            Name of the file to be loaded
        
        Returns
        -------
        datatype : string
            Type of input data. Can be 'fitsimage', 'fitscube', or 'ascii'
        
        """
        # Import necessary modules
        from astropy.io import fits, ascii
        
        # Initialize datatype variable
        datatype = None
        
        try:
            # Try opening the file with fits module
            data = fits.open(filename)
            data = data[0].data.squeeze()
            # Check the shape of the data
            if len(data.shape) == 2:
                datatype = 'fitsimage'
            elif len(data.shape) == 3:
                datatype = 'fitscube'
            else:
                raise ValueError('Unsupported FITS file format.')
        
        except:
            try:
                # If opening with fits module fails, try opening with ascii module
                data = ascii.read(filename)
                datatype = 'ascii'
            
            except:
                raise ValueError('Unsupported file format.')
        
        # Return the determined datatype
        return datatype

    def create_directory_structure(self, filename, datadir, outputdir=None, verbose=True):
        """
        Make the output directory

        Parameters
        ----------
        filename : string
            Name of the file to be loaded
        datadir: string
            Path to data location
        outputdir : string, optional
            Alternate output directory. Default is datadir
        verbose : bool, optional
            verbose output to terminal

        Returns
        -------
        acornsdir : string
            Path to the acorns directory
        
        """        
        # Determine the filename without extension
        filename_noext = os.path.splitext(filename)[0]
        
        # Determine the output directory
        if outputdir is None:
            outputdir = datadir
        
        # Determine the acorns directory
        acornsdir=os.path.join(outputdir, filename_noext)

        # Create the directory structure
        if not os.path.exists(acornsdir):
            if verbose:
                print('Creating Acorns directory structure.')
            self.mkdirectory(acornsdir, verbose=verbose)
            self.mkdirectory(os.path.join(acornsdir, 'acorn'), verbose=verbose)
            self.mkdirectory(os.path.join(acornsdir, 'tables'), verbose=verbose)
            self.mkdirectory(os.path.join(acornsdir, 'images'), verbose=verbose)
            self.mkdirectory(os.path.join(acornsdir, 'figures'), verbose=verbose)
            self.mkdirectory(os.path.join(acornsdir, 'config_files'), verbose=verbose)
        else:
            if verbose:
                print('Acorns directory already exists. Skipping directory creation.')
        
        # Return the output directory
        return acornsdir

    def mkdirectory(self, dir, verbose=True):
        """
        Make a directory if it does not exist

        Parameters
        ----------
        dir : string
            Path to the directory to be created
        
        
        """
        if not os.path.exists(dir):
            os.mkdir(dir)
        else:
            if verbose:
                print('Directory already exists. Skipping directory creation.')
    
    def mkstring(self, st):
        """
        Make a string

        Parameters
        ----------
        st : string
            String to be converted to a string

        Returns
        -------
        newstring : string
            String with single quotes around it
        
        """
        newstring="\'" + str(st) + "\'"
        return newstring

    def append_keywords(self, config_file, dct, all_keywords=False, description=True):
        """
        Append keywords to the config file

        Parameters
        ----------
        config_file : list
            List containing the lines of the config file
        dct : OrderedDict
            Dictionary containing the keywords to be appended
        all_keywords : bool, optional
            Append all keywords to the config file. Default is False
        description : bool, optional
            Append the description of the keyword to the config file. Default is True
        
        Returns
        -------
        config_file : list
            List containing the lines of the config file
        
        """
        for key in dct.keys():
            if all_keywords:
                if description:
                    config_file.append(
                        '\n\n# {}'.format(dct[key]['description']))
                config_file.append('\n{} = {}'.format(key, dct[key]['default']))
            else:
                if dct[key]['simple']:
                    if description:
                        config_file.append(
                            '\n\n# {}'.format(dct[key]['description']))
                    config_file.append('\n{} = {}'.format(key, dct[key]['default']))
        return config_file

    def create_config_file(self, filename, datadir, outputdir=None, datatype=None, verbose=True):
        """
        Creates the configuration table for acorns

        Parameters
        ----------
        filename : string
            Name of the file to be loaded
        datadir : string
            Path to data location
        outputdir : string, optional
            Alternate output directory. Default is datadir
        datatype : string, optional
            Type of input data. Can be 'fitsimage', 'fitscube', or 'ascii'. Default is None
        verbose : bool, optional
            verbose output to terminal

        Returns
        -------
        config_file : string
            Path to the config file

        """

        from collections import OrderedDict

        # Determine the filename without extension
        filename_noext, file_extension = os.path.splitext(filename)

        # config file location
        configdir = os.path.join(outputdir, 'config_files')
        config_filename = 'acorns.config'

        # check to see if config file already exists
        if os.path.exists(os.path.join(configdir, config_filename)):
            if verbose:
                print('Config file already exists.')
            return os.path.join(configdir, config_filename)
        else:
            if verbose:
                print('Creating config file.')

            # create config file with default settings
            config_header = str('# ACORNS config file\n\n')

            # The following keywords are data independent and will be including in all config files
            default = [
                ('filename', {
                    'default': self.mkstring(filename),
                    'description': "name of the table (including extension)",
                    'simple': True}),
                ('datadirectory', {
                    'default': self.mkstring(datadir),
                    'description': "location of the table containing the data you would like to cluster",
                    'simple': True}),
                ('outputdirectory', {
                    'default': self.mkstring(outputdir),
                    'description': "output directory for data products",
                    'simple': True}),
                ('datatype', {
                    'default': self.mkstring(datatype),
                    'description': "data type: fits, ascii",
                    'simple': True}),
                 ('verbose', {
                    'default': 'True',
                    'description': "print messages to the terminal [True/False]",
                    'simple': True}),
                ('autosave', {
                    'default': 'True',
                    'description': "autosave output [True/False]",
                    'simple': True}),
                ('n_jobs', {
                    'default': 3,
                    'description': "number of workers for ball point query of kdtree",
                    'simple': True}),
                ('min_npoints', {
                    'default': '0.0',
                    'description': "minimum number of data points to be considered a cluster",
                    'simple': True}),
                ('min_height', {
                    'default': '0.0',
                    'description': "minimum height above the merge level to be considered a cluster",
                    'simple': True}),
                ('stop_value', {
                    'default': '0.0',
                    'description': "stopping criteria for clustering. value of the stopping criterion",
                    'simple': True}),
                ]

            # Get data dependent default configs
            methoddefs = self.create_method_defaults(datatype)

            method_defaults=[
                ('cluster_method', {
                    'default': methoddefs['cluster_method'],
                    'description': "clustering method. depends on the data type. for a fits image the default is PP. for a fits cube the default is PPV. for an ascii file the default is tab.",
                    'simple': True}),
                ('cluster_spatial_idx', {
                    'default': methoddefs['cluster_spatial_idx'],
                    'description': "column locations for spatial information, list one column for each axis, order [x,y] - for fits image "+\
                                    "default is [1,0]",
                    'simple': False}),
                ('cluster_spatial_metric', {
                    'default': methoddefs['cluster_spatial_metric'],
                    'description': "Distance metric for clustering of spatial information. Default is the euclidean distance. ",
                    'simple': True}),
                ('cluster_spatial_thresh', {
                    'default': methoddefs['cluster_spatial_thresh'],
                    'description': "Merging criteria for clusters. Defines maximum distance at which two points can be linked. ",
                    'simple': True}),
                ('cluster_feature_idx', {
                    'default': methoddefs['cluster_feature_idx'],
                    'description': "column locations for clustering information, list one column for each feature to cluster. "+\
                                   "for a fits cube the default is [0] (velocity axis) ",
                    'simple': False}),
                ('cluster_feature_metric', {
                    'default': methoddefs['cluster_feature_metric'],
                    'description': "Distance metrics for clustering of features. should be a list of length = nfeatures e.g. ['absolute', 'absolute']. ",
                    'simple': False}),
                ('cluster_feature_thresh', {
                    'default': methoddefs['cluster_feature_thresh'],
                    'description': "Merging criteria for clusters. Defines maximum distance at which two points can be linked. should be a list of length = nfeatures e.g. [0.1, 0.1]. ",
                    'simple': False}),
                ('cluster_stopcrit_idx', {
                    'default': methoddefs['cluster_stopcrit_idx'],
                    'description': "stopping criteria for clustering. column index for stopping criterion. for fits images and cubes use `data'. ",
                    'simple': False}),
                ('cluster_sortcrit_idx', {
                    'default': methoddefs['cluster_sortcrit_idx'],
                    'description': "column location for variable to sort by. Default would be to set this as the data column. "+\
                                    "For a fits file the default is `data' i.e. the contents of the array. For an ascii file the default is 'None'. "+\
                                    "However, this could be set to e.g. the column index of peak intensity, mass, density, etc",
                    'simple': True}),
                ('reverse', {
                    'default': methoddefs['reverse'],
                    'description': "whether or not to reverse sort cluster_sortcrit_idx",
                    'simple': True}),
                ]

            # Create the ordered dictionary
            dct_default = OrderedDict(default)
            dct_method = OrderedDict(method_defaults)

            # Create the config file
            config_file = []
            config_file.append('[DEFAULT]')
            config_file = self.append_keywords(config_file, dct_default,
                                          all_keywords=True,
                                          description=True)
            config_file = self.append_keywords(config_file, dct_method,
                                          all_keywords=True,
                                          description=True)

            # Write the config file
            with open(os.path.join(configdir, config_filename), 'w') as file:
                for line in config_file:
                    file.write(line)

            return os.path.join(configdir, config_filename)


    def create_method_defaults(self, datatype):
        """
        Create the method defaults

        Parameters
        ----------
        datatype : string
            Type of input data. Can be 'fitsimage', 'fitscube', or 'ascii'
        
        Returns
        -------
        methoddefs : dictionary
            Dictionary containing the method defaults

        """
        methoddefs = {}
        
        if datatype=='fitsimage':
            # assumption is that clustering will be done in PP space
            methoddefs['cluster_method'] = self.mkstring('PP')
            methoddefs['cluster_spatial_idx'] = '[1,0]'
            methoddefs['cluster_spatial_metric']=self.mkstring('euclidean')
            methoddefs['cluster_spatial_thresh']='0.0'
            methoddefs['cluster_feature_idx'] = '[None]'
            methoddefs['cluster_feature_metric'] = '[None]'
            methoddefs['cluster_feature_thresh'] = '[None]'
            methoddefs['cluster_stopcrit_idx'] = 'None'
            methoddefs['cluster_sortcrit_idx'] = self.mkstring('data')
            methoddefs['reverse'] = 'False'
        elif datatype=='fitscube':
            # assumption is that clustering will be done in PP space
            methoddefs['cluster_method'] = self.mkstring('PPV')
            methoddefs['cluster_spatial_idx'] = '[2,1]'
            methoddefs['cluster_spatial_metric']=self.mkstring('euclidean')
            methoddefs['cluster_spatial_thresh']='0.0'
            methoddefs['cluster_feature_idx'] = '[0]'
            methoddefs['cluster_feature_metric'] = "['absolute']"
            methoddefs['cluster_feature_thresh'] = '[None]'
            methoddefs['cluster_stopcrit_idx'] = 'None'
            methoddefs['cluster_sortcrit_idx'] = self.mkstring('data')
            methoddefs['reverse'] = 'False'
        elif datatype=='ascii':
            # no assumptions are made re clustering method
            methoddefs['cluster_method'] = self.mkstring('tab')
            methoddefs['cluster_spatial_idx'] = '[None]'
            methoddefs['cluster_spatial_metric']=self.mkstring('euclidean')
            methoddefs['cluster_spatial_thresh']='0.0'
            methoddefs['cluster_feature_idx'] = '[None]'
            methoddefs['cluster_feature_metric'] = '[None]'
            methoddefs['cluster_feature_thresh'] = '[None]'
            methoddefs['cluster_stopcrit_idx'] = 'None'
            methoddefs['cluster_sortcrit_idx'] = 'None'
            methoddefs['reverse'] = 'False'

        return methoddefs
    
    def import_from_config(self, config_filename, config_key='DEFAULT'):
        """
        Read in values from configuration table.

        Parameters
        ----------
        config_file : str
            Filepath to configuration file
        config_key : str
            Section of configuration file, whose parameters should be read in addition to 'DEFAULT'.

        Notes
        -----
        adapted from Gausspy+ methodology

        """
        import ast
        import configparser

        config = configparser.ConfigParser()
        config.read(config_filename)

        for key, value in config[config_key].items():
            try:
                value=ast.literal_eval(value)
                setattr(self, key, value)
            except ValueError:
                raise Exception('Could not parse parameter {} from config file'.format(key))
            
        return self


    def initialize_data(self): 
        """
        Initialize the data into a format that can be used by the clustering algorithm

        """
        # Import necessary modules
        from astropy.io import fits, ascii
        from astropy.table import Table, Column

        # initialize the data table depending on the clustering method
        if self.datatype=='fitsimage':
            data = fits.open(self.filename)[0].data.squeeze()
            xpix, ypix = np.meshgrid(np.arange(data.shape[self.cluster_spatial_idx[0]]), np.arange(data.shape[self.cluster_spatial_idx[1]]))
            data_acorns_arr = np.vstack((xpix.flatten(), ypix.flatten(), data.flatten())).T
            # now make a table with appropriate column names
            data_acorns = Table(data_acorns_arr, names=['sfeature1', 'sfeature2', 'data'])

        elif self.datatype=='fitscube':
            data = fits.open(self.filename)[0].data.squeeze()
            xpix, ypix, vpix = np.meshgrid(np.arange(data.shape[self.cluster_spatial_idx[0]]), np.arange(data.shape[self.cluster_spatial_idx[1]]), np.arange(data.shape[self.cluster_feature_idx[0]]))
            data_acorns_arr = np.vstack((xpix.flatten(), ypix.flatten(), vpix.flatten(), data.flatten())).T
            # now make a table with appropriate column names
            data_acorns = Table(data_acorns_arr, names=['sfeature1', 'sfeature2', 'feature1', 'data'])

        elif self.datatype=='ascii':
            data = ascii.read(self.filename)
            if self.cluster_spatial_idx[0] is not None:
                # if clustering spatial information we are going to need at least two dimensions so test for this
                if len(self.cluster_spatial_idx)>=2:
                    x, y = Column(data.columns[self.cluster_spatial_idx[0]], name='sfeature1'), Column(data.columns[self.cluster_spatial_idx[1]], name='sfeature2')
                    # create an empty table
                    data_acorns=Table()
                    # append columns x, y to data_acorns
                    data_acorns.add_column(x)
                    data_acorns.add_column(y)

                    if len(self.cluster_spatial_idx)>2:
                        z = Column(data.columns[self.cluster_spatial_idx[2]], name='sfeature3')
                        data_acorns.add_column(z)
                else:
                    raise ValueError('Unsupported data format. If clustering spatial information, the data must have at least two dimensions.')
                
            if self.cluster_feature_idx[0] is not None:
                mycolumns=[Column(data.columns[self.cluster_feature_idx[i]], name='feature'+str(self.cluster_feature_idx[i])) for i in range(len(self.cluster_feature_idx))]
                for col in mycolumns:
                    data_acorns.add_column(col)

            if self.cluster_sortcrit_idx is not None:
                data_acorns.add_column(Column(data.columns[self.cluster_sortcrit_idx], name='data'))
            else:
                data_acorns.add_column(Column(np.ones_like(data.columns[0]), name='data'))

        # now we have our basic data table lets clean it
        # First lets remove rows with NaNs in the data column
        mask = np.isnan(data_acorns['data'])
        data_acorns.remove_rows(mask)

        # apply the stopping criteria by masking the data if required
        if self.cluster_stopcrit_idx is not None:
            if self.cluster_method=='tab':
                stop = Column(data.columns[self.cluster_stopcrit_idx], name='stop')
                mask = (stop < self.stop_value)
                data_acorns.remove_rows(mask)
            else:
                mask = (data_acorns['data'] < self.stop_value)
                data_acorns.remove_rows(mask)


        # sort the data if required
        if self.cluster_sortcrit_idx is not None:
            if self.reverse:
                # Sort the table by the 'data' column
                data_acorns.sort('data')
            else:
                # default is to sort ascending so we need to reverse the reverse
                data_acorns.sort('data')
                data_acorns.reverse()
        
        return data_acorns

# # Instantiate the Setup class and run the setup process
# if __name__ == "__main__":
#     setup = Setup()
#     setup.setup_project()
#     setup.install_dependencies()
#     setup.run()

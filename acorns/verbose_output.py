# Licensed under an MIT open source license - see LICENSE

from .colors import *
from tqdm import tqdm

def print_to_terminal(stage='', step='', length=None, var=None, t1=None, t2=None):
    """
    Keeping all the noisy stuff in one place.
    """
    if stage=='init':
        if step=='init':
            print('')
            print('---------------------')
            print(colors.fg._lightblue_+'initialising acorns'+colors._endc_)
            print('---------------------')
            print('')
            progress_bar=[]
        if step=='configexists':
            print(colors.fg._lightgreen_+"acorns config file already exists. Returning filepath. "+colors._endc_)
            print('')
            progress_bar=[]
        if step=='makingconfig':
            print(colors.fg._lightgreen_+"config file created "+colors._endc_)
            print('')
            progress_bar=[]

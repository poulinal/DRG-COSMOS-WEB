# AP 2026

from abc import abstractmethod
from typing import Callable

import numpy as np
from matplotlib import pylab as plt
from astropy.io import ascii, fits
from astropy.table import Table

from astropy import units as u

class CatalogHandler():
    def __init__(self, path):
        self.catalog_path : str = path
        
        self.hdu : fits.HDUList = None
        self.hdr : fits.Header = None
        self.cat_photom : dict[str,Table] = {}
        self.cat_lephare : dict[str,Table] = {}
        self.cat_cigale : dict[str,Table] = {}
        self.cat_bd : dict[str,Table] = {}
        
        self.condition_clean : dict[np.ndarray] = {}
        
    
    @abstractmethod
    def load_catalog(self):
        pass

    @abstractmethod
    def get_filter_cut(self, filter_func : Callable | np.ndarray, filtername : str = "default_filter", filter_to_take_from : str = "original"):
        pass
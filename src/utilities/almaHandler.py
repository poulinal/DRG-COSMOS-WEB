# AP 2026

from typing import Callable

import numpy as np
from matplotlib import pylab as plt
from astropy.io import ascii, fits
from astropy.table import Table

from astropy import units as u
from utilities.catalogHandler import CatalogHandler

class AlmaHandler(CatalogHandler):
    def __init__(self, path):
        super().__init__(path)
        
    def load_catalog(self):
        # try:
        import os
        if not os.path.exists(self.catalog_path):
            raise FileNotFoundError(f"Catalog file not found at {self.catalog_path}")
        
        with fits.open(self.catalog_path) as hdu:
            self.hdu = hdu
            print(f" HDU Info: {hdu.info()}")
            self.hdr = hdu[1].header
            self.cat_photom['original'] = Table(hdu[1].data)
        print("Catalog loaded successfully.")
        
    def get_filter_cut(self, filter_func : Callable | np.ndarray, filtername : str = "default_filter", filter_to_take_from : str = "original"):
        """
        Method to apply a filter cut to the photometry catalog and store the resulting filtered catalog under a new name.

        Args:
            filter_func (Callable | np.ndarray): Either a boolean numpy array of the same length as the photometry catalog, where True values indicate which entries to keep, or a callable function that takes the photometry catalog as input and returns such a boolean array. Callable function should have the signature filter_func(catalog) -> np.ndarray[bool].
            filtername (str, optional): The name under which to store the filtered catalog. Defaults to "default_filter".
            filter_to_take_from (str, optional): The name of the catalog from which to take the filter. Defaults to "original".

        Raises:
            ValueError: _filter_to_take_from_ not found in photometry catalog.
            ValueError: _filtername_ already exists in photometry catalog.
            ValueError: _filter_func_ is a numpy array but has incorrect length.
            ValueError: _filter_func_ is a callable but returns an array of incorrect length.
            ValueError: _filter_func_ is neither a numpy array nor a callable function.

        Returns:
            np.ndarray[bool]: The boolean mask used for filtering the catalog.
        """
        if filter_to_take_from not in self.cat_photom:
            raise ValueError(f"Filter to take from '{filter_to_take_from}' not found in photometry catalog. Available options: {list(self.cat_photom.keys())}")
        if filtername in self.cat_photom:
            print(f"Filter '{filtername}' already exists in photometry catalog. Overwriting it.")
        
        if isinstance(filter_func, np.ndarray):
            if filter_func.shape[0] != len(self.cat_photom[filter_to_take_from]):
                raise ValueError(f"Filter function array length {filter_func.shape[0]} does not match number of entries in photometry catalog {len(self.cat_photom[filter_to_take_from])}.")
            self.cat_photom[filtername] = self.cat_photom[filter_to_take_from][filter_func]
            mask = filter_func
        elif callable(filter_func):
            filter_mask = filter_func(self.cat_photom[filter_to_take_from])
            if filter_mask.shape[0] != len(self.cat_photom[filter_to_take_from]):
                raise ValueError(f"Filter function output length {filter_mask.shape[0]} does not match number of entries in photometry catalog {len(self.cat_photom[filter_to_take_from])}.")
            self.cat_photom[filtername] = self.cat_photom[filter_to_take_from][filter_mask]
            mask = filter_mask
        else:
            raise ValueError(f"Filter function must be either a numpy array or a callable function. Got {type(filter_func)}.")
        
        return mask
    
    def get_cat_photom(self, filtername : str = "original"):
        if filtername not in self.cat_photom:
            raise ValueError(f"Filter name '{filtername}' not found in photometry catalog. Available options: {list(self.cat_photom.keys())}")
        return self.cat_photom[filtername]
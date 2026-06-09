# AP 2026

from typing import Callable

import numpy as np
from matplotlib import pylab as plt
from astropy.io import ascii, fits
from astropy.table import Table

from astropy import units as u
from utilities.catalogHandler import CatalogHandler

class CosmosWebHandler(CatalogHandler):
    
    def __init__(self, path, catalog_type='master'):
        
        self.hdu : fits.HDUList = None
        self.hdr : fits.Header = None
        self.cat_photom : dict[str,Table] = {}
        self.cat_lephare : dict[str,Table] = {}
        self.cat_cigale : dict[str,Table] = {}
        self.cat_bd : dict[str,Table] = {}
        
        self.condition_clean : dict[np.ndarray] = {}
        
        if catalog_type != 'photometry' and catalog_type != 'lephare' and catalog_type != 'cigale' and catalog_type != 'bd' and catalog_type != 'master':
            raise NotImplementedError(f"Catalog type '{catalog_type}' not supported. Supported types: 'photometry', 'lephare', 'cigale', 'bd', 'master'.")
        self.catalog_type = catalog_type
        
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
            self.cat_photom['original'] = Table(hdu[1].data) if self.catalog_type in ['photometry', 'master'] else None
            self.cat_lephare['original'] = Table(hdu[2].data) if self.catalog_type in ['lephare', 'master'] else None
            self.cat_cigale['original'] = Table(hdu[4].data) if self.catalog_type in ['cigale', 'master'] else None
            self.cat_bd['original'] = Table(hdu[6].data) if self.catalog_type in ['bd', 'master'] else None
        print("Catalog loaded successfully.")

        self.purity_cut()
        print("Purity cut applied successfully.")
        
        self.miri_cut()
        print("MIRI cut applied successfully.")
            # print("Data loaded successfully.")
        # except Exception as e:
        #     print(f"Error loading data: {e}")
        #     raise
        
    def purity_cut(self):
        if self.cat_photom is None or self.cat_lephare is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        # if self.catalog_type == 'lephare':
        self.condition_clean['condition_clean'] = np.logical_and.reduce((
            self.cat_lephare['original']['type']==0, # Select only galaxies
            self.cat_photom['original']['warn_flag']==0, # No warning flag
            np.abs(self.cat_photom['original']['mag_model_f444w'])<30, # Remove very faint objects
            # self.cat_photom['original']['flag_star_hsc']==0, # Remove objects in HSC star mask area # For this project, we will keep since we don't care about the photometry
        ))
        mask = self.condition_clean['condition_clean']
        print(f"Purity cut: {np.sum(mask)} out of {len(self.cat_lephare['original'])} objects remain. Fraction: {np.sum(mask)/len(self.cat_lephare['original']):.2%}")

    
        self.cat_lephare['condition_clean'] = self.cat_lephare['original'][mask]
        self.cat_photom['condition_clean'] = self.cat_photom['original'][mask]
        self.cat_cigale['condition_clean'] = self.cat_cigale['original'][mask]
        self.cat_bd['condition_clean'] = self.cat_bd['original'][mask]

    def miri_cut(self):
        condition_clean_miri = np.logical_and(self.condition_clean['condition_clean'], self.cat_photom['original']['flux_model_f770w']>0)
        self.condition_clean['condition_clean_miri'] = condition_clean_miri

        print(f"MIRI cut: {np.sum(condition_clean_miri)} out of {len(self.cat_lephare['condition_clean'])} objects remain. Fraction: {np.sum(condition_clean_miri)/len(self.cat_lephare['condition_clean']):.2%}")

        self.cat_lephare['condition_clean_miri'] = self.cat_lephare['original'][condition_clean_miri]
        self.cat_photom['condition_clean_miri'] = self.cat_photom['original'][condition_clean_miri]
        self.cat_cigale['condition_clean_miri'] = self.cat_cigale['original'][condition_clean_miri]
        self.cat_bd['condition_clean_miri'] = self.cat_bd['original'][condition_clean_miri]

    def get_filter_cut(self, filter_func : Callable | np.ndarray, filtername : str = "default_filter", filter_to_take_from : str = "condition_clean_miri"):
        """
        Apply an additional filter to the already-loaded catalogs.

        `filter_func` may be either:
          - a callable with signature (cat_photom, cat_lephare, cat_cigale, cat_bd)
            that returns a boolean mask
          - or a boolean array-like mask directly (matching the current catalog lengths)

        The method updates the stored tables in-place (like `purity_cut`) and
        returns the boolean mask that was applied.
        """
        if self.cat_photom is None or self.cat_lephare is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if callable(filter_func):
            mask = filter_func(
                self.cat_photom[filter_to_take_from],
                self.cat_lephare[filter_to_take_from],
                self.cat_cigale[filter_to_take_from],
                self.cat_bd[filter_to_take_from],
            )
        else:
            mask = np.asarray(filter_func)

        mask = np.asarray(mask, dtype=bool)

        print(f"Filter cut: {np.sum(mask)} out of {len(self.cat_lephare[filter_to_take_from])} objects remain. Fraction: {np.sum(mask)/len(self.cat_lephare[filter_to_take_from]):.2%}")

        self.cat_lephare[filtername] = self.cat_lephare[filter_to_take_from][mask]
        self.cat_photom[filtername] = self.cat_photom[filter_to_take_from][mask]
        self.cat_cigale[filtername] = self.cat_cigale[filter_to_take_from][mask]
        self.cat_bd[filtername] = self.cat_bd[filter_to_take_from][mask]

        self.condition_clean[filtername] = mask

        return mask
    
    def get_cat_lephare(self, filtername : str = "original"):
        return self.cat_lephare[filtername]
    
    def get_cat_photom(self, filtername : str = "original"):
        return self.cat_photom[filtername]
    
    def get_cat_cigale(self, filtername : str = "original"):
        return self.cat_cigale[filtername]
    
    def get_cat_bd(self, filtername : str = "original"):
        return self.cat_bd[filtername]
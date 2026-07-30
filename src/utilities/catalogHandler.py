# AP 2026

from abc import abstractmethod
# from tkinter.font import names
from typing import Callable

import numpy as np
from matplotlib import pylab as plt
from astropy.io import ascii, fits
from astropy.table import Table

from astropy import units as u

from utilities.fitFileHandler import FitFileHandler
from utilities.bandEnum import BandEnum
from utilities.catalogAvailableEnum import CatalogAvailableEnum

class CatalogHandler(FitFileHandler):
    def __init__(self, path):
        super().__init__(path)
        self.catalog_path : str = path
        self.type : CatalogAvailableEnum = None
        
        self.hdu : fits.HDUList = None
        self.hdr : fits.Header = None
        self.cat_photom : dict[str,Table] = {}
        self.cat_lephare : dict[str,Table] = {}
        self.cat_cigale : dict[str,Table] = {}
        self.cat_bd : dict[str,Table] = {}
        
        self.condition_clean : dict[np.ndarray] = {}

    @abstractmethod
    def get_filter_cut(self, filter_func : Callable | np.ndarray, filtername : str = "default_filter", filter_to_take_from : str = "original"):
        pass

    @abstractmethod
    def load_catalog(self):
        self.load_fit_file()
    
    @staticmethod
    def get_dict_filtername(catalog : dict[str, Table], filtername : str = "original"):
        # print(f"catalog: {catalog}, catalog type: {type(catalog)}")
        all_filternames_in_catalog = list(catalog.keys())
        if filtername in all_filternames_in_catalog:
            return catalog[filtername]
        else:
            raise KeyError(f"Filter name '{filtername}' not found in catalog. Available filter names: {all_filternames_in_catalog}")
    
    @abstractmethod
    def get_photometry_catalog(self, filtername : str = "original"):
        pass
    
    @abstractmethod
    def get_size_catalog(self, filtername : str = "original"):
        pass
    
    @abstractmethod
    def get_photoz_catalog(self, filtername : str = "original"):
        pass
    
    @abstractmethod
    def get_filter_catalog_name_convention(self, band : BandEnum = BandEnum.F444W, aperture : int = 2):
        pass
    
    @abstractmethod
    def get_magnitude_catalog_name_convention(self, band : BandEnum = BandEnum.F444W, aperture : int = 2):
        pass

    def _normalize_band(self, band):
        """Normalize `band` to a BandEnum instance.

        Accepts a `BandEnum` or a string (either the enum value like 'f770w' or
        the member name like 'F770W'). Raises TypeError/ValueError on bad input.
        """
        if isinstance(band, BandEnum):
            return band
        if isinstance(band, str):
            s = band.lower()
            # try matching stored value (e.g. 'f770w')
            try:
                return BandEnum.get_band_from_string(s)
            except ValueError:
                # try matching by member name (e.g. 'F770W')
                try:
                    return BandEnum[band]
                except Exception:
                    raise ValueError(f"Invalid band string: {band}")
        raise TypeError(f"band must be a BandEnum or str, got {type(band)}")
    
    @abstractmethod
    def get_filter_flux(self, filtername : str = "original", band : BandEnum = BandEnum.F444W, aperture : int = 2):
        """
        Gets the flux for a given filter in the photometry catalog. Returns a tuple of the list of flux for the filter and the list of error of the flux for the filter.

        Args:
            filtername (str, optional): _description_. Defaults to "original".
            band (BandEnum, optional): _description_. Defaults to BandEnum.F444W.
            aperture (int, optional): _description_. Defaults to 2.

        Returns:
            _type_: _description_
        """
        pass
        
    @abstractmethod
    def get_filter_magnitude(self, filtername : str = "original", band : BandEnum = BandEnum.F444W, aperture : int = 2):
        """
        Gets the magnitude for a given filter in the photometry catalog. Returns a tuple of the list of magnitude for the filter and the list of error of the magnitude for the filter.

        Args:
            filtername (str, optional): _description_. Defaults to "original".
            band (BandEnum, optional): _description_. Defaults to BandEnum.F444W.
            aperture (int, optional): _description_. Defaults to 2.

        Returns:
            _type_: _description_
        """
        pass
    
    @abstractmethod
    def get_z_redshift(self, filtername : str = "original"):
        pass
    
    @abstractmethod
    def get_radius_col_name(self):
        pass
    
    def get_radius(self, filtername : str = "original"):
        """
            Gets the radius for a given filter in the photometry catalog. Returns a tuple of the list of radius for the filter and the list of error of the radius for the filter. Returns in arcseconds.

        Args:
            filtername (str, optional): _description_. Defaults to "original".
        """
        radius_col, radius_err_col = self.get_radius_col_name()
        print(self.get_size_catalog(filtername))
        if radius_err_col is None:
            return self.get_size_catalog(filtername)[radius_col], None
        else:
            return self.get_size_catalog(filtername)[radius_col], self.get_size_catalog(filtername)[radius_err_col]
    
    @abstractmethod
    def get_id_col_name(self):
        pass
    
    def get_ids(self, filtername : str = "original"):
        return self.get_photometry_catalog(filtername)[self.get_id_col_name()]
    
    @abstractmethod
    def get_ra_col_name(self):
        pass
    
    @abstractmethod
    def get_dec_col_name(self):
        pass

    
    
    
    
    
    
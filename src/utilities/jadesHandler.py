# AP 2026

from typing import Callable

import numpy as np
from matplotlib import pylab as plt
from astropy.io import ascii, fits
from astropy.table import Table

from astropy import units as u
from utilities.catalogHandler import CatalogHandler
from utilities.bandEnum import BandEnum
from utilities.catalogAvailableEnum import CatalogAvailableEnum
from utilities.jadesApertureEnum import JadesApertureEnum

class JadesHandler(CatalogHandler):
    def __init__(self, path):
        super().__init__(path)
        self.cat_flag : dict[str, Table] = {'original': None}
        self.cat_size : dict[str, Table] = {'original': None}
        self.cat_circ_bsub_conv : dict[str, Table] = {'original': None}
        self.cat_circ : dict[str, Table] = {'original': None}
        self.cat_circ_bsub : dict[str, Table] = {'original': None}
        self.cat_circ_conv : dict[str, Table] = {'original': None}
        self.cat_miri : dict[str, Table] = {'original': None}
        self.cat_photoz : dict[str, Table] = {'original': None}
        self.type = CatalogAvailableEnum.JADES
        
    # HDU layout of the v1.1 master catalog on disk.
    _MASTER_HDU_INDEX = {
        'FILTERS': 1, 'FLAG': 2, 'SIZE': 3, 'CIRC': 4,
        'CIRC_BSUB': 5, 'CIRC_CONV': 6, 'CIRC_BSUB_CONV': 7,
        'KRON' : 8, 'KRON_CONV': 9, 'MIRI': 10, 'PHOTOZ': 11,
        'PHOTOZ_KRON': 12,
    }
    
    @staticmethod
    def _find_hdu(hdul, *substrings):
        """Return the first table HDU whose EXTNAME matches any of the given
        substrings (case-insensitive, '+' ignored), or None if absent.

        Derived catalogs are looked up by name rather than position because the
        master layout (photom@1, lephare@2, cigale@4, bd@6) does not hold for
        the trimmed files (different order, fewer/no extensions)."""
        wanted = [s.replace('+', '').lower() for s in substrings]
        for h in hdul[1:]:
            name = (h.name or '').replace('+', '').lower()
            if any(w in name for w in wanted):
                return h
        return None

    def load_catalog(self, apply_cuts=True):
        # try:
        import os
        if not os.path.exists(self.catalog_path):
            raise FileNotFoundError(f"Catalog file not found at {self.catalog_path}")
        
        def _load(hdu, *names):
            h = self._find_hdu(hdu, *names)
            return Table(h.data) if h is not None else None
        
        with fits.open(self.catalog_path) as hdu:
            self.hdu = hdu
            print(f" HDU Info: {hdu.info()}")
            self.hdr = hdu[1].header
            self.cat_flag['original'] = _load(hdu, 'FLAG')
            self.cat_size['original'] = _load(hdu, 'SIZE')
            self.cat_circ['original'] = _load(hdu, 'CIRC')
            self.cat_circ_bsub['original'] = _load(hdu, 'CIRC_BSUB')
            self.cat_circ_conv['original'] = _load(hdu, 'CIRC_CONV')
            self.cat_circ_bsub_conv['original'] = _load(hdu, 'CIRC_BSUB_CONV')
            self.cat_miri['original'] = _load(hdu, 'MIRI')
            self.cat_photoz['original'] = _load(hdu, 'PHOTOZ')
        print("Catalog loaded successfully.")
        
        if apply_cuts:
            self.purity_cut()
            self.clean_miri_cut()
            # self.make_selection_cut()
        
    
    def purity_cut(self):
        #selects where FLAG_BN < 2
        # self.get_filter_cut(filter_func=lambda flag: flag['FLAG_BN'] < 2, filtername='condition_purity', filter_to_take_from='original')
        #for now just copy the original catalog to condition_purity since the unsure if we want to apply the FLAG_BN < 2 cut or not.
        # self.cat_flag['condition_purity'] = self.cat_flag['original']
        # self.cat_size['condition_purity'] = self.cat_size['original']
        # self.cat_circ_bsub_conv['condition_purity'] = self.cat_circ_bsub_conv['original']
        # self.cat_miri['condition_purity'] = self.cat_miri['original']
        # self.cat_photoz['condition_purity'] = self.cat_photoz['original']
        
        self.get_filter_cut(filter_func=lambda flag, size, circ_bsub_conv, miri, photoz: flag['ID'] != 0, filtername='condition_purity', filter_to_take_from='original')
        
        
    def clean_miri_cut(self, aperature = JadesApertureEnum.APER_0p5.value):
        # aperature = JadesApertureEnum.APER_0p1.value
        #selects where MIRI F770W_CIRC5 > 0
        self.get_filter_cut(filter_func=lambda flag, size, circ_bsub_conv, miri, photoz: np.logical_and(miri[f'F770W_CIRC{aperature}_BSUB'] > 0, miri[f'F770W_CIRC{aperature}_bkg_BSUB'] != 0), filtername='condition_clean_miri', filter_to_take_from='condition_purity')
        
    
    def make_selection_cut(self, aperature = JadesApertureEnum.APER_0p5.value):
        # aperature = JadesApertureEnum.APER_0p1.value # use the smallest aperture for the selection cut since it is the most sensitive to point sources and we want to be inclusive in our selection cut to not miss any potential high-redshift candidates, and then we can apply more stringent cuts later on the clean sample with MIRI and z > 4.
        condition_detection_aper = self.get_filter_cut(
            filtername='condition_detection_aper',
            filter_func=lambda flag, size, circ_bsub_conv, miri, photoz: (
                (np.asarray(circ_bsub_conv[f'F115W_CIRC{aperature}']) / np.asarray(circ_bsub_conv[f'F115W_CIRC{aperature}_ei']) < 3)
                & (np.asarray(circ_bsub_conv[f'F150W_CIRC{aperature}']) / np.asarray(circ_bsub_conv[f'F150W_CIRC{aperature}_ei']) < 3)
                & (np.asarray(circ_bsub_conv[f'F277W_CIRC{aperature}']) / np.asarray(circ_bsub_conv[f'F277W_CIRC{aperature}_ei']) < 3)
                & (np.asarray(circ_bsub_conv[f'F444W_CIRC{aperature}']) / np.asarray(circ_bsub_conv[f'F444W_CIRC{aperature}_ei']) >= 5)
                & (np.asarray(miri[f'F770W_CIRC{aperature}_BSUB']) / np.asarray(miri[f'F770W_CIRC{aperature}_ei_BSUB']) >= 5)
            ),
            filter_to_take_from='condition_clean_miri'
        )
        
        
    def get_filter_cut(self, filter_func, filtername = "default_filter", filter_to_take_from = "original"):
        """
        Apply an additional filter to the already-loaded catalogs.

        `filter_func` may be either:
          - a callable with signature (cat_flag, cat_size, cat_circ_bsub_conv, cat_miri, cat_photoz) -> np.ndarray[bool]
            that returns a boolean mask
          - or a boolean array-like mask directly (matching the current catalog lengths)

        The method updates the stored tables in-place (like `purity_cut`) and
        returns the boolean mask that was applied.
        """
        if self.cat_flag is None or self.cat_size is None or self.cat_circ_bsub_conv is None or self.cat_miri is None or self.cat_photoz is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if filter_to_take_from not in self.cat_flag:
            raise ValueError(f"Filter to take from '{filter_to_take_from}' not found in catalog. Available options: {list(self.cat_flag.keys())}")
        if filtername in self.cat_flag:
            print(f"Filter '{filtername}' already exists in catalog. Overwriting it.")
            
        if isinstance(filter_func, np.ndarray) or isinstance(filter_func, list):
            mask = np.asarray(filter_func, dtype=bool)
            if mask.shape[0] != len(self.cat_flag[filter_to_take_from]):
                raise ValueError(f"Filter function array length {mask.shape[0]} does not match number of entries in catalog {len(self.cat_flag[filter_to_take_from])}.")
        elif callable(filter_func):
            mask = filter_func(self.cat_flag[filter_to_take_from], self.cat_size[filter_to_take_from], self.cat_circ_bsub_conv[filter_to_take_from], self.cat_miri[filter_to_take_from], self.cat_photoz[filter_to_take_from])
            if mask.shape[0] != len(self.cat_flag[filter_to_take_from]):
                raise ValueError(f"Filter function output length {mask.shape[0]} does not match number of entries in catalog {len(self.cat_flag[filter_to_take_from])}.")
        else:
            raise ValueError(f"Filter function must be either a numpy array or a callable function. Got {type(filter_func)}.")
        
        self.cat_flag[filtername] = self.cat_flag[filter_to_take_from][mask]
        self.cat_size[filtername] = self.cat_size[filter_to_take_from][mask]
        self.cat_circ_bsub_conv[filtername] = self.cat_circ_bsub_conv[filter_to_take_from][mask]
        self.cat_miri[filtername] = self.cat_miri[filter_to_take_from][mask]
        self.cat_photoz[filtername] = self.cat_photoz[filter_to_take_from][mask]
        self.cat_circ[filtername] = self.cat_circ[filter_to_take_from][mask] if self.cat_circ[filter_to_take_from] is not None else None
        self.cat_circ_bsub[filtername] = self.cat_circ_bsub[filter_to_take_from][mask] if self.cat_circ_bsub[filter_to_take_from] is not None else None
        self.cat_circ_conv[filtername] = self.cat_circ_conv[filter_to_take_from][mask] if self.cat_circ_conv[filter_to_take_from] is not None else None
        
        print(f"Applied filter '{filtername}' to catalog. {mask.sum()} entries remain out of {len(self.cat_flag[filter_to_take_from])}, fraction: {mask.sum() / len(self.cat_flag[filter_to_take_from]):.2f}.")
        
        return mask
    
    
    
    def get_cat_flag(self, filtername = "original"):
        if filtername not in self.cat_flag:
            raise ValueError(f"Filter name '{filtername}' not found in catalog. Available options: {list(self.cat_flag.keys())}")
        return self.cat_flag[filtername]
    
    def get_cat_size(self, filtername = "original"):
        if filtername not in self.cat_size:
            raise ValueError(f"Filter name '{filtername}' not found in catalog. Available options: {list(self.cat_size.keys())}")
        return self.cat_size[filtername]
    
    def get_cat_circ_bsub_conv(self, filtername = "original"):
        if filtername not in self.cat_circ_bsub_conv:
            raise ValueError(f"Filter name '{filtername}' not found in catalog. Available options: {list(self.cat_circ_bsub_conv.keys())}")
        return self.cat_circ_bsub_conv[filtername]
    
    def get_cat_miri(self, filtername = "original"):
        if filtername not in self.cat_miri:
            raise ValueError(f"Filter name '{filtername}' not found in catalog. Available options: {list(self.cat_miri.keys())}")
        return self.cat_miri[filtername]
    
    def get_cat_photoz(self, filtername = "original"):
        if filtername not in self.cat_photoz:
            raise ValueError(f"Filter name '{filtername}' not found in catalog. Available options: {list(self.cat_photoz.keys())}")
        return self.cat_photoz[filtername]
    
    def get_photometry_catalog(self, filtername = "original"):
        return self.get_cat_circ_bsub_conv(filtername)
    
    def get_size_catalog(self, filtername = "original"):
        return self.get_cat_size(filtername)
    
    def get_radius_col_name(self):
        """
        Returns the column names for the radius and radius error in the photometry catalog. For JADES, the column names are given as A, B, FWHM in the SIZE table, and we can use the A column as the radius and the A_err column as the radius error. So we can just return 'A' and 'A_err' as the column names for the radius and radius error.
        """
        return 'A', None # JADES does not provide radius error, so we return None for the radius error column name.

    def get_filter_catalog_name_convention(self, band : BandEnum = BandEnum.F444W, aperture : int = 2):
        band = self._normalize_band(band)
        if band == BandEnum.F770W:
            return f"{str(band).upper()}_CIRC{aperture}_BSUB", f"{str(band).upper()}_CIRC{aperture}_ei_BSUB"
        else:
            return f"{str(band).upper()}_CIRC{aperture}", f"{str(band).upper()}_CIRC{aperture}_ei"
        
    def get_filter_magnitude(self, filtername: str = "original", band: BandEnum = BandEnum.F444W, aperture: int = 2):
        """Return magnitudes in AB mag. In JADES, the aperature index is in [0, 6] corresponding to circular apertures of radius [0.1, 0.15, 0.25, 0.3, 0.35, 0.5] arcsec. The magnitude is calculated from the flux using the formula mag = -2.5 * log10(flux) + 31.4 since JADES flux is in nJy, and the error is propagated using mag_err = 2.5 / ln(10) * (flux_err / flux).
        """
        band = self._normalize_band(band)
        flux, flux_err = self.get_filter_flux(filtername, band, aperture)
        print(f"Calculating magnitude for filter '{filtername}', band '{band}', aperture {aperture}: flux = {flux[:5]}, flux_err = {flux_err[:5]}")
        return self._get_magnitude_from_flux_njy(flux, flux_err)
    
    def get_filter_flux(self, filtername : str = "original", band : BandEnum = BandEnum.F444W, aperture : int = 2):
        filter_col_name, filter_err_col_name = self.get_filter_catalog_name_convention(band, aperture)
        print(f"Getting flux for filter '{filtername}', band '{band}', aperture {aperture}: column names = '{filter_col_name}', '{filter_err_col_name}'")
        band = self._normalize_band(band)
        if band == BandEnum.F770W:
            print(f"Using MIRI catalog for F770W fluxes since they are background-subtracted, while the main photometry catalog is not background-subtracted for F770W.")
            photom_catalog = self.get_cat_miri(filtername)
        else:
            photom_catalog = self.get_photometry_catalog(filtername)
        return self._getcol(photom_catalog, filter_col_name), self._getcol(photom_catalog, filter_err_col_name)
        
    def get_magnitude_catalog_name_convention(self, band : BandEnum = BandEnum.F444W, aperture : int = 2):
        # will have to manually convert flux to magnitude since the catalog only provides flux and flux error for the circular apertures, not magnitude and magnitude error. So we can just return the same column names as the flux but with "mag" instead of "flux" in the name, and then in get_filter_magnitude we can convert the flux to magnitude using the formula mag = -2.5 * log10(flux) + 31.4 since JADES in nJy. Error can be propagated using the formula mag_err = 2.5 / ln(10) * (flux_err / flux)
        band = self._normalize_band(band)
        if band == BandEnum.F770W:
            return f"{str(band).upper()}_CIRC{aperture}_BSUB", f"{str(band).upper()}_CIRC{aperture}_ei_BSUB"
        else:
            return f"{str(band).upper()}_CIRC{aperture}", f"{str(band).upper()}_CIRC{aperture}_ei"
        
    @staticmethod
    def _get_magnitude_from_flux_njy(flux, flux_err):
        # Convert flux in nJy to magnitude using the formula mag = -2.5 * log10(flux) + 8.9 + 22.5 -- > + 31.4, and propagate error using mag_err = 2.5 / ln(10) * (flux_err / flux)
        mag = -2.5 * np.log10(flux) + 31.4
        mag_err = 2.5 / np.log(10) * (flux_err / flux)
        return mag, mag_err
    
    def get_photoz_catalog(self, filtername : str = "original"):
        return self.get_cat_photoz(filtername)
    
    def get_z_redshift(self, filtername = "original"):
        return self.get_photoz_catalog(filtername)['z_ml'] #try z_ml for now (maximum-likelihood redshift), but we can also try z_a (median redshift) and z_spec (spectroscopic redshift)



    
    
    
    
    
    
    
    
    
    
    
    
    
    def save_reduced_catalog(self, path, reduced='light'):
        """
        Save a reduced version of the catalog containing only the most essential columns for quick loading and analysis. The 'light' version includes only columns FLAG, SIZE, CIRC_BULB_CONV, MIRI, and PHOTOZ tables, while the 'lighter' version includes only a subset of columns from these tables.

        Args:
            path (str): The file path to save the reduced catalog to.
            reduced (str, optional): The level of reduction to apply. Options are 'light' or 'lighter' or 'lightest'. Defaults to 'light'.
            """
            
            
        if reduced == 'light':
            #resave the cosmos web master catalog but leave cigale, bd, ml-morpho, se++aper, and galfitm-morpo empty to save space and load time, since we won't be using those columns in our analysis

            self.save_catalog_streamed(path,
                                    tables_to_keep=('FLAG', 'SIZE', 'CIRC_BSUB_CONV', 'MIRI', 'PHOTOZ'),)
            
        elif reduced == 'lighter':
            # now resave but  within photometry hotcold and se++, remove all columns except id, ra, dec, radius_sersic, radius_sersic_err, sersic, sersic_err, type, warn_flag, mag_model_f*, mag_err_model_f*, mag_aper_f*, mag_err_aper_f*, flux_model_f*, flux_err_model_f*, flux_aper_f*, and flux_err_aper_f* 
            columns_to_keep = ['ID', 'RA', 'DEC', 'FLAG_BN', 'A', 'B', 'FWHM', 
                               'F115W_CIRC0', 'F115W_CIRC0_bkg', 'F115W_CIRC0_ei', 'F150W_CIRC0', 'F150W_CIRC0_bkg', 'F150W_CIRC0_ei', 'F277W_CIRC0', 'F277W_CIRC0_bkg', 'F277W_CIRC0_ei', 'F444W_CIRC0', 'F444W_CIRC0_bkg', 'F444W_CIRC0_ei', 'F770W_CIRC0', 'F770W_CIRC0_BSUB', 'F770W_CIRC0_bkg_BSUB', 'F770W_CIRC0_ei', 'F770W_CIRC0_ei_BSUB',
                               
                               'F115W_CIRC1', 'F115W_CIRC1_bkg', 'F115W_CIRC1_ei', 'F150W_CIRC1', 'F150W_CIRC1_bkg', 'F150W_CIRC1_ei', 'F277W_CIRC1', 'F277W_CIRC1_bkg', 'F277W_CIRC1_ei', 'F444W_CIRC1', 'F444W_CIRC1_bkg', 'F444W_CIRC1_ei', 'F770W_CIRC1', 'F770W_CIRC1_BSUB', 'F770W_CIRC1_bkg_BSUB', 'F770W_CIRC1_ei', 'F770W_CIRC1_ei_BSUB',
                               
                               'F115W_CIRC2', 'F115W_CIRC2_bkg', 'F115W_CIRC2_ei', 'F150W_CIRC2', 'F150W_CIRC2_bkg', 'F150W_CIRC2_ei', 'F277W_CIRC2', 'F277W_CIRC2_bkg', 'F277W_CIRC2_ei', 'F444W_CIRC2', 'F444W_CIRC2_bkg', 'F444W_CIRC2_ei', 'F770W_CIRC2', 'F770W_CIRC2_BSUB', 'F770W_CIRC2_bkg_BSUB', 'F770W_CIRC2_ei', 'F770W_CIRC2_ei_BSUB',
                               
                               'F115W_CIRC3', 'F115W_CIRC3_bkg', 'F115W_CIRC3_ei', 'F150W_CIRC3', 'F150W_CIRC3_bkg', 'F150W_CIRC3_ei', 'F277W_CIRC3', 'F277W_CIRC3_bkg', 'F277W_CIRC3_ei', 'F444W_CIRC3', 'F444W_CIRC3_bkg', 'F444W_CIRC3_ei', 'F770W_CIRC3', 'F770W_CIRC3_BSUB', 'F770W_CIRC3_bkg_BSUB', 'F770W_CIRC3_ei', 'F770W_CIRC3_ei_BSUB',
                               
                               'F115W_CIRC4', 'F115W_CIRC4_bkg', 'F115W_CIRC4_ei', 'F150W_CIRC4', 'F150W_CIRC4_bkg', 'F150W_CIRC4_ei', 'F277W_CIRC4', 'F277W_CIRC4_bkg', 'F277W_CIRC4_ei', 'F444W_CIRC4', 'F444W_CIRC4_bkg', 'F444W_CIRC4_ei', 'F770W_CIRC4', 'F770W_CIRC4_BSUB', 'F770W_CIRC4_bkg_BSUB', 'F770W_CIRC4_ei', 'F770W_CIRC4_ei_BSUB',
                               
                               'F115W_CIRC5', 'F115W_CIRC5_bkg', 'F115W_CIRC5_ei', 'F150W_CIRC5', 'F150W_CIRC5_bkg', 'F150W_CIRC5_ei', 'F277W_CIRC5', 'F277W_CIRC5_bkg', 'F277W_CIRC5_ei', 'F444W_CIRC5', 'F444W_CIRC5_bkg', 'F444W_CIRC5_ei', 'F770W_CIRC5', 'F770W_CIRC5_BSUB', 'F770W_CIRC5_bkg_BSUB', 'F770W_CIRC5_ei', 'F770W_CIRC5_ei_BSUB',
                               
                               'F115W_CIRC6', 'F115W_CIRC6_bkg', 'F115W_CIRC6_ei', 'F150W_CIRC6', 'F150W_CIRC6_bkg', 'F150W_CIRC6_ei', 'F277W_CIRC6', 'F277W_CIRC6_bkg', 'F277W_CIRC6_ei', 'F444W_CIRC6', 'F444W_CIRC6_bkg', 'F444W_CIRC6_ei', 'F770W_CIRC6', 'F770W_CIRC6_BSUB', 'F770W_CIRC6_bkg_BSUB', 'F770W_CIRC6_ei', 'F770W_CIRC6_ei_BSUB', 'z_a', 'z_ml', 'z_spec',]
            self.save_catalog_streamed(path,
                                    tables_to_keep=('FLAG', 'SIZE', 'CIRC_BSUB_CONV', 'MIRI', 'PHOTOZ'),
                                    columns_to_keep=columns_to_keep)
        else:
            raise ValueError(f"Invalid reduction level '{reduced}'. Valid options are 'light', 'lighter'.")






    def save_catalog_streamed(self, path, tables_to_keep=('FLAG', 'SIZE', 'CIRC_BSUB_CONV', 'MIRI', 'PHOTOZ'),
                              columns_to_keep=None, mask=None, overwrite=True):
        """
        Memory-lean resave that streams selected HDUs straight from the source
        FITS file on disk (``memmap=True``) instead of the in-memory ``cat_*``
        tables. Tables you don't keep are never read into RAM, and you do NOT
        need to call ``load_catalog`` first.

        Peak memory is at most one extension at a time (roughly zero when no
        ``mask``/``columns_to_keep`` is given, since astropy streams the
        memmapped data to the output file).

        - ``tables_to_keep``: any of 'FLAG', 'SIZE', 'CIRC_BSUB_CONV', 'MIRI', 'PHOTOZ'. Output order is preserved.
        - ``mask``: optional boolean array over the source rows (e.g. a
          ``condition_clean_miri`` mask) applied to every kept table.
        - ``columns_to_keep``: optional list of column names; only columns that
          exist in a given table are read from disk, so unused columns (and the
          461-column B+D table, if dropped) never touch RAM.
        """
        unknown = set(tables_to_keep) - set(self._MASTER_HDU_INDEX)
        if unknown:
            raise ValueError(f"Unknown table(s) {unknown}. Valid: {sorted(self._MASTER_HDU_INDEX)}.")
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)

        with fits.open(self.catalog_path, memmap=True) as src:
            out = fits.HDUList([fits.PrimaryHDU()])
            for name in tables_to_keep:
                src_hdu = src[self._MASTER_HDU_INDEX[name]]

                if columns_to_keep is None and mask is None:
                    # Pure pass-through: astropy streams memmapped data to disk.
                    out.append(fits.BinTableHDU(data=src_hdu.data, name=name))
                else:
                    cols = (columns_to_keep if columns_to_keep is not None
                            else src_hdu.columns.names)
                    new_cols = []
                    for c in cols:
                        if c not in src_hdu.columns.names:
                            continue  # column not present in this table; skip
                        arr = src_hdu.data[c]          # reads only this column
                        if mask is not None:
                            arr = arr[mask]
                        new_cols.append(fits.Column(name=c, array=arr,
                                                    format=src_hdu.columns[c].format))
                    out.append(fits.BinTableHDU.from_columns(new_cols, name=name))

            out.writeto(path, overwrite=overwrite)
        print(f"Catalog streamed to {path} (kept: {list(tables_to_keep)}).")
# AP 2026

from typing import Callable
import requests
import fitz  # PyMuPDF

import numpy as np
from matplotlib import pylab as plt
from astropy.io import ascii, fits
from astropy.table import Table

from astropy import units as u
from utilities.catalogHandler import CatalogHandler
from utilities.bandEnum import BandEnum
from utilities.catalogAvailableEnum import CatalogAvailableEnum
from utilities.cosmosWebApertureEnum import CosmosWebApertureEnum

class CosmosWebHandler(CatalogHandler):
    
    def __init__(self, path, catalog_type='master'):
        
        self.catalog_path = path
        self.hdu : fits.HDUList = None
        self.hdr : fits.Header = None
        self.cat_photom : dict[str,Table] = {}
        self.cat_lephare : dict[str,Table] = {}
        self.cat_cigale : dict[str,Table] = {}
        self.cat_bd : dict[str,Table] = {}
        
        self._png_cache = {}   # sid -> list of PNG byte-strings (one per page)
        self.pdf_url_sourceid = lambda sid: f"https://cosmos2025.iap.fr/fitsmap/data/inspec_plots/cosmos_web_sed_{sid}.pdf"
        
        self.condition_clean : dict[np.ndarray] = {}
        
        if catalog_type != 'photometry' and catalog_type != 'lephare' and catalog_type != 'cigale' and catalog_type != 'bd' and catalog_type != 'master':
            raise NotImplementedError(f"Catalog type '{catalog_type}' not supported. Supported types: 'photometry', 'lephare', 'cigale', 'bd', 'master'.")
        self.catalog_type = catalog_type
        
        super().__init__(path)
        
        self.type = CatalogAvailableEnum.COSMOS_WEB
        
    # HDU layout of the v1.1 master catalog on disk.
    _MASTER_HDU_INDEX = {
        'photometry': 1, 'lephare': 2, 'se_aper': 3, 'cigale': 4,
        'ml_morpho': 5, 'bd': 6, 'galfitm_morpho': 7,
    }
    
    def _render_pdf(self, sid, dpi=130):
        if sid not in self._png_cache:
            r = requests.get(self.pdf_url_sourceid(sid), timeout=30)
            r.raise_for_status()
            doc = fitz.open(stream=r.content, filetype='pdf')
            self._png_cache[sid] = [doc.load_page(p).get_pixmap(dpi=dpi).tobytes('png')
                            for p in range(doc.page_count)]
            doc.close()
        return self._png_cache[sid]

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
            self.cat_photom['original'] = _load(hdu, 'photom') if self.catalog_type in ['photometry', 'master'] else None
            self.cat_lephare['original'] = _load(hdu, 'lephare') if self.catalog_type in ['lephare', 'master'] else None
            self.cat_cigale['original'] = _load(hdu, 'cigale') if self.catalog_type in ['cigale', 'master'] else None
            self.cat_bd['original'] = _load(hdu, 'bd', 'b+d') if self.catalog_type in ['bd', 'master'] else None
        print("Catalog loaded successfully. Tables present: "
              + ", ".join(n for n, t in [('photom', self.cat_photom['original']),
                                         ('lephare', self.cat_lephare['original']),
                                         ('cigale', self.cat_cigale['original']),
                                         ('bd', self.cat_bd['original'])] if t is not None))

        if apply_cuts:
            self.purity_cut()
            self.clean_miri_cut()
        else:
            print("Skipping purity/MIRI cuts (apply_cuts=False).")
            # print("Data loaded successfully.")
        # except Exception as e:
        #     print(f"Error loading data: {e}")
        #     raise

    def _missing_columns(self):
        """Columns the cuts require but that are absent from the loaded tables."""
        need = {'lephare': ['type'],
                'photom': ['warn_flag', 'mag_model_f444w', 'flux_model_f770w']}
        missing = []
        for cat, table in [('lephare', self.cat_lephare.get('original')),
                           ('photom', self.cat_photom.get('original'))]:
            if table is None:
                missing += [f"{cat}:{c}" for c in need[cat]]
            else:
                missing += [f"{cat}:{c}" for c in need[cat] if c not in table.colnames]
        return missing

    def purity_cut(self):
        if self.cat_photom is None or self.cat_lephare is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        missing = self._missing_columns()
        if missing:
            print(f"Skipping purity/MIRI cuts: required columns not in this catalog: {missing}")
            return
        # if self.catalog_type == 'lephare':
        self.condition_clean['condition_clean'] = np.logical_and.reduce((
            self.cat_lephare['original']['type']==0, # Select only galaxies
            self.cat_photom['original']['warn_flag']==0, # No warning flag
            # np.abs(self.cat_photom['original']['mag_model_f444w'])<30, # Remove very faint objects
            np.abs(self.cat_photom['original']['mag_aper_f444w'][:, 2])<30, # Remove very faint objects
            # self.cat_photom['original']['flag_star_hsc']==0, # Remove objects in HSC star mask area # For this project, we will keep since we don't care about the photometry
        ))
        mask = self.condition_clean['condition_clean']
        print(f"Purity cut: {np.sum(mask)} out of {len(self.cat_lephare['original'])} objects remain. Fraction: {np.sum(mask)/len(self.cat_lephare['original']):.2%}")

    
        self.cat_lephare['condition_clean'] = self.cat_lephare['original'][mask]
        self.cat_photom['condition_clean'] = self.cat_photom['original'][mask]
        self.cat_cigale['condition_clean'] = self.cat_cigale['original'][mask] if self.cat_cigale['original'] is not None else None
        self.cat_bd['condition_clean'] = self.cat_bd['original'][mask] if self.cat_bd['original'] is not None else None

    def clean_miri_cut(self, aperature = CosmosWebApertureEnum.APER_0p5.value):
        if 'condition_clean' not in self.condition_clean:
            return  # purity_cut was skipped (required columns absent)
        condition_clean_miri = np.logical_and(self.condition_clean['condition_clean'], self.cat_photom['original']['flux_aper_f770w'][:, aperature]>0)
        self.condition_clean['condition_clean_miri'] = condition_clean_miri

        print(f"MIRI cut: {np.sum(condition_clean_miri)} out of {len(self.cat_lephare['condition_clean'])} objects remain. Fraction: {np.sum(condition_clean_miri)/len(self.cat_lephare['condition_clean']):.2%}")

        self.cat_lephare['condition_clean_miri'] = self.cat_lephare['original'][condition_clean_miri]
        self.cat_photom['condition_clean_miri'] = self.cat_photom['original'][condition_clean_miri]
        self.cat_cigale['condition_clean_miri'] = self.cat_cigale['original'][condition_clean_miri] if self.cat_cigale['original'] is not None else None
        self.cat_bd['condition_clean_miri'] = self.cat_bd['original'][condition_clean_miri] if self.cat_bd['original'] is not None else None

    def make_selection_cut(self, aperature = CosmosWebApertureEnum.APER_0p5.value):
        condition_detection_aper = self.get_filter_cut(
            filtername='condition_detection_aper',
            filter_func=lambda photom, lephare, cigale, bd: (
                (np.asarray(photom['flux_aper_f277w'])[:, aperature] / np.asarray(photom['flux_err_aper_f277w'])[:, aperature] < 3)
                & (np.asarray(photom['flux_aper_f115w'])[:, aperature] / np.asarray(photom['flux_err_aper_f115w'])[:, aperature] < 3)
                & (np.asarray(photom['flux_aper_f150w'])[:, aperature] / np.asarray(photom['flux_err_aper_f150w'])[:, aperature] < 3)
                & (np.asarray(photom['flux_aper_f444w'])[:, aperature] / np.asarray(photom['flux_err_aper_f444w'])[:, aperature] >= 5)
                & (np.asarray(photom['flux_aper_f770w'])[:, aperature] / np.asarray(photom['flux_err_aper_f770w'])[:, aperature] >= 5)
            )
        )

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
            ) if self.cat_cigale[filter_to_take_from] is not None and self.cat_bd[filter_to_take_from] is not None else filter_func(
                self.cat_photom[filter_to_take_from],
                self.cat_lephare[filter_to_take_from],
                None,
                None,
            )
        else:
            mask = np.asarray(filter_func)

        mask = np.asarray(mask, dtype=bool)

        print(f"Filter cut: {np.sum(mask)} out of {len(self.cat_lephare[filter_to_take_from])} objects remain. Fraction: {np.sum(mask)/len(self.cat_lephare[filter_to_take_from]):.2%}")

        self.cat_lephare[filtername] = self.cat_lephare[filter_to_take_from][mask]
        self.cat_photom[filtername] = self.cat_photom[filter_to_take_from][mask]
        self.cat_cigale[filtername] = self.cat_cigale[filter_to_take_from][mask] if self.cat_cigale[filter_to_take_from] is not None else None
        self.cat_bd[filtername] = self.cat_bd[filter_to_take_from][mask] if self.cat_bd[filter_to_take_from] is not None else None

        self.condition_clean[filtername] = mask

        return mask
    
    def get_obj_position(self, filtername : str = "original", objectID : int = None):
        if objectID is None:
            raise ValueError("objectID must be provided.")
        photom_catalog = self.get_photometry_catalog(filtername)
        if photom_catalog is None:
            raise ValueError(f"Photometry catalog for filter '{filtername}' is not loaded.")
        id_col_name = self.get_id_col_name()
        ra_col_name = self.get_ra_col_name()
        dec_col_name = self.get_dec_col_name()
        source_row = photom_catalog[photom_catalog[id_col_name] == objectID]
        if len(source_row) == 0:
            raise ValueError(f"Source ID {objectID} not found in photometry catalog for filter '{filtername}'.")
        return source_row[ra_col_name][0], source_row[dec_col_name][0]
        
    def get_cat_lephare(self, filtername : str = "original"):
        return self.get_dict_filtername(self.cat_lephare, filtername)

    def get_cat_photom(self, filtername : str = "original"):
        # return self.cat_photom[filtername]
        return self.get_dict_filtername(self.cat_photom, filtername)
    
    def get_cat_cigale(self, filtername : str = "original"):
        return self.get_dict_filtername(self.cat_cigale, filtername)
    
    def get_cat_bd(self, filtername : str = "original"):
        return self.get_dict_filtername(self.cat_bd, filtername)
    
    def get_cat_size(self, filtername : str = "original"):
        # returns in degrees so we can convert to arcseconds by multiplying by 3600
        if self.get_photometry_catalog(filtername) is not None and 'radius_sersic' in self.get_photometry_catalog(filtername).colnames:
            return self.get_photometry_catalog(filtername)['radius_sersic'] * 3600, self.get_photometry_catalog(filtername)['radius_sersic_err'] * 3600
        else:
            raise ValueError(f"Size column 'radius_sersic' not found in photometry catalog for filter '{filtername}'.")
        
    def get_radius_col_name(self, filtername = "original"):
        # return 'radius_sersic', 'radius_sersic_err'
        return 0,1
        
    def get_photometry_catalog(self, filtername : str = "original"):
        return self.get_dict_filtername(self.cat_photom, filtername)
    
    def get_photoz_catalog(self, filtername : str = "original"):
        return self.get_dict_filtername(self.cat_lephare, filtername)
    
    def get_z_redshift(self, filtername : str = "original"):
        if self.get_photoz_catalog(filtername) is not None and 'zfinal' in self.get_photoz_catalog(filtername).colnames:
            return self.get_photoz_catalog(filtername)['zfinal']
        else:
            raise ValueError(f"Redshift column 'zfinal' not found in lephare catalog for filter '{filtername}'.")
    
    def get_filter_catalog_name_convention(self, band : BandEnum = BandEnum.F444W, aperture : int = 2):
        return f"flux_aper_{str(band).lower()}", f"flux_err_aper_{str(band).lower()}"
    
    def get_filter_catalog_name_convention_model(self, band : BandEnum = BandEnum.F444W, aperture : int = 2):
        return f"flux_model_{str(band).lower()}", f"flux_err-cal_model_{str(band).lower()}"
    
    def get_magnitude_catalog_name_convention(self, band : BandEnum = BandEnum.F444W, aperture : int = 2):
        return f"mag_aper_{str(band).lower()}", f"mag_err_aper_{str(band).lower()}"
    
    def get_size_catalog(self, filtername : str = "original"):
        return self.get_cat_size(filtername)
        
    def get_filter_magnitude(self, filtername: str = "original", band: BandEnum = BandEnum.F444W, aperture: int = 2):
        # return self._getcol(self.get_photometry_catalog(filtername), self.get_magnitude_catalog_name_convention(band, aperture))[:, aperture], None  # Catalog does not provide magnitude error for circular apertures, so we return None for the error.
        mag_col_name, mag_err_col_name = self.get_magnitude_catalog_name_convention(band, aperture)
        photom_catalog = self.get_photometry_catalog(filtername)
        return self._getcol(photom_catalog, mag_col_name)[:, aperture], None  # Catalog does not provide magnitude error for circular apertures, so we return None for the error.
    
    def get_filter_flux(self, filtername : str = "original", band : BandEnum = BandEnum.F444W, aperture : int = 2):
        filter_col_name, filter_err_col_name = self.get_filter_catalog_name_convention(band, aperture)
        photom_catalog = self.get_photometry_catalog(filtername)
        return self._getcol(photom_catalog, filter_col_name)[:, aperture], self._getcol(photom_catalog, filter_err_col_name)[:, aperture]
    
    def get_filter_flux_of_id(self, source_id, filtername : str = "original", band : BandEnum = BandEnum.F444W, aperture : int = 2, useModel : bool = False):
        if useModel:
            filter_col_name, filter_err_col_name = self.get_filter_catalog_name_convention_model(band, aperture)
        else:
            filter_col_name, filter_err_col_name = self.get_filter_catalog_name_convention(band, aperture)
        photom_catalog = self.get_photometry_catalog(filtername)
        id_col_name = self.get_id_col_name()
        source_row = photom_catalog[photom_catalog[id_col_name] == source_id]
        if len(source_row) == 0:
            raise ValueError(f"Source ID {source_id} not found in photometry catalog for filter '{filtername}'.")
        if useModel:
            return source_row[filter_col_name][0], source_row[filter_err_col_name][0]
        else:
            return source_row[filter_col_name][0][aperture], source_row[filter_err_col_name][0][aperture]
        
    def get_id_col_name(self):
        return 'id'
    
    def get_ra_col_name(self):
        return 'ra'
    
    def get_dec_col_name(self):
        return 'dec'
    
    
    
    
    
    
    
    
    def save_reduced_catalog(self, path, reduced='light'):
        """
        Save a reduced version of the catalog containing only the most essential columns for quick loading and analysis. The 'light' version includes only columns lephlare and photometry tables, while the 'lighter' version includes only a subset of columns

        Args:
            path (str): The file path to save the reduced catalog to.
            reduced (str, optional): The level of reduction to apply. Options are 'light' or 'lighter' or 'lightest'. Defaults to 'light'.
            """
            
            
        if reduced == 'light':
            #resave the cosmos web master catalog but leave cigale, bd, ml-morpho, se++aper, and galfitm-morpo empty to save space and load time, since we won't be using those columns in our analysis

            self.save_catalog_streamed(path,
                                    tables_to_keep=('lephare', 'photometry'))
        elif reduced == 'lighter':
            # now resave but  within photometry hotcold and se++, remove all columns except id, ra, dec, radius_sersic, radius_sersic_err, sersic, sersic_err, type, warn_flag, mag_model_f*, mag_err_model_f*, mag_aper_f*, mag_err_aper_f*, flux_model_f*, flux_err_model_f*, flux_aper_f*, and flux_err_aper_f* 
            columns_to_keep = ['id', 'ra', 'dec', 'radius_sersic', 'radius_sersic_err', 'sersic', 'sersic_err', 'type', 'warn_flag', 'zfinal', 'tile',
                            'mag_model_f115w', 'mag_err_model_f115w', 'mag_aper_f115w', 'mag_err_aper_f115w', 'flux_model_f115w', 'flux_err_model_f115w', 'flux_err-cal_model_f115w', 'flux_aper_f115w', 'flux_err_aper_f115w',
                            'mag_model_f150w', 'mag_err_model_f150w', 'mag_aper_f150w', 'mag_err_aper_f150w', 'flux_model_f150w', 'flux_err_model_f150w', 'flux_err-cal_model_f150w', 'flux_aper_f150w', 'flux_err_aper_f150w',
                            'mag_model_f277w', 'mag_err_model_f277w', 'mag_aper_f277w', 'mag_err_aper_f277w', 'flux_model_f277w', 'flux_err_model_f277w', 'flux_err-cal_model_f277w', 'flux_aper_f277w', 'flux_err_aper_f277w',
                            'mag_model_f444w', 'mag_err_model_f444w', 'mag_aper_f444w', 'mag_err_aper_f444w', 'flux_model_f444w', 'flux_err_model_f444w', 'flux_err-cal_model_f444w', 'flux_aper_f444w', 'flux_err_aper_f444w', 
                            'mag_model_f770w', 'mag_err_model_f770w', 'mag_aper_f770w', 'mag_err_aper_f770w', 'flux_model_f770w', 'flux_err_model_f770w', 'flux_err-cal_model_f770w', 'flux_aper_f770w', 'flux_err_aper_f770w']
            self.save_catalog_streamed(path,
                                    tables_to_keep=('photometry','lephare'),
                                    columns_to_keep=columns_to_keep)

        elif reduced == 'lightest':
            # now resave with just the condition_clean_miri catalog
            with fits.open(self.catalog_path, memmap=True) as s:
                miri_mask = ((s[2].data['type'] == 0) &           # lephare HDU
                            (s[1].data['warn_flag'] == 0) &       # photometry HDU
                            (np.abs(s[1].data['mag_model_f444w']) < 30) &
                            (s[1].data['flux_model_f770w'] > 0))
                
            self.save_catalog_streamed(path,
                                    tables_to_keep=('photometry','lephare'),
                                    columns_to_keep=columns_to_keep,
                                    mask=miri_mask)
        else:
            raise ValueError(f"Invalid reduction level '{reduced}'. Valid options are 'light', 'lighter', or 'lightest'.")

    def save_catalog_streamed(self, path, tables_to_keep=('lephare', 'photometry'),
                              columns_to_keep=None, mask=None, overwrite=True):
        """
        Memory-lean resave that streams selected HDUs straight from the source
        FITS file on disk (``memmap=True``) instead of the in-memory ``cat_*``
        tables. Tables you don't keep are never read into RAM, and you do NOT
        need to call ``load_catalog`` first.

        Peak memory is at most one extension at a time (roughly zero when no
        ``mask``/``columns_to_keep`` is given, since astropy streams the
        memmapped data to the output file).

        - ``tables_to_keep``: any of 'lephare', 'photometry', 'cigale', 'bd',
          'se_aper', 'ml_morpho', 'galfitm_morpho'. Output order is preserved.
        - ``mask``: optional boolean array over the source rows (e.g. a
          ``condition_clean_miri`` mask) applied to every kept table.
        - ``columns_to_keep``: optional list of column names; only columns that
          exist in a given table are read from disk, so unused columns (and the
          461-column B+D table, if dropped) never touch RAM.
        """
        columns_kept = []
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
                    columns_kept.extend(src_hdu.columns.names)
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
                    columns_kept.extend(c.name for c in new_cols)

            out.writeto(path, overwrite=overwrite)
        print(f"Catalog streamed to {path} (kept: {list(tables_to_keep)}), columns kept: {columns_kept}.")
        
        
        
        
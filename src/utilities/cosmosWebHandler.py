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
            self.miri_cut()
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
            np.abs(self.cat_photom['original']['mag_model_f444w'])<30, # Remove very faint objects
            # self.cat_photom['original']['flag_star_hsc']==0, # Remove objects in HSC star mask area # For this project, we will keep since we don't care about the photometry
        ))
        mask = self.condition_clean['condition_clean']
        print(f"Purity cut: {np.sum(mask)} out of {len(self.cat_lephare['original'])} objects remain. Fraction: {np.sum(mask)/len(self.cat_lephare['original']):.2%}")

    
        self.cat_lephare['condition_clean'] = self.cat_lephare['original'][mask]
        self.cat_photom['condition_clean'] = self.cat_photom['original'][mask]
        self.cat_cigale['condition_clean'] = self.cat_cigale['original'][mask] if self.cat_cigale['original'] is not None else None
        self.cat_bd['condition_clean'] = self.cat_bd['original'][mask] if self.cat_bd['original'] is not None else None

    def miri_cut(self):
        if 'condition_clean' not in self.condition_clean:
            return  # purity_cut was skipped (required columns absent)
        condition_clean_miri = np.logical_and(self.condition_clean['condition_clean'], self.cat_photom['original']['flux_model_f770w']>0)
        self.condition_clean['condition_clean_miri'] = condition_clean_miri

        print(f"MIRI cut: {np.sum(condition_clean_miri)} out of {len(self.cat_lephare['condition_clean'])} objects remain. Fraction: {np.sum(condition_clean_miri)/len(self.cat_lephare['condition_clean']):.2%}")

        self.cat_lephare['condition_clean_miri'] = self.cat_lephare['original'][condition_clean_miri]
        self.cat_photom['condition_clean_miri'] = self.cat_photom['original'][condition_clean_miri]
        self.cat_cigale['condition_clean_miri'] = self.cat_cigale['original'][condition_clean_miri] if self.cat_cigale['original'] is not None else None
        self.cat_bd['condition_clean_miri'] = self.cat_bd['original'][condition_clean_miri] if self.cat_bd['original'] is not None else None

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
    
    def get_cat_lephare(self, filtername : str = "original"):
        return self.cat_lephare[filtername]
    
    def get_cat_photom(self, filtername : str = "original"):
        return self.cat_photom[filtername]
    
    def get_cat_cigale(self, filtername : str = "original"):
        return self.cat_cigale[filtername]
    
    def get_cat_bd(self, filtername : str = "original"):
        return self.cat_bd[filtername]
    
    def save_catalog(self, path, columns_to_keep = None, columns_to_remove = None, tables_to_remove = None, filtername = 'original'):
        if self.cat_photom is None or self.cat_lephare is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if filtername not in self.cat_lephare:
            raise ValueError(f"Filter name '{filtername}' not found in catalog. Available filters: {list(self.cat_lephare.keys())}")
        if columns_to_keep is not None and columns_to_remove is not None:
            raise ValueError("Cannot specify both columns_to_keep and columns_to_remove. Please specify only one of them.")

        # tables_to_remove drops whole HDU extensions (e.g. 'cigale', 'bd'). These are
        # separate tables in the master catalog, not columns inside one table.
        tables_to_remove = set(tables_to_remove) if tables_to_remove is not None else set()
        unknown = tables_to_remove - {'lephare', 'photometry', 'cigale', 'bd'}
        if unknown:
            raise ValueError(f"Unknown table(s) {unknown}. Valid tables: 'lephare', 'photometry', 'cigale', 'bd'.")
        
        print(f"Saving catalog with filter '{filtername}' to {path}...")

        lephare_table_to_save = self.cat_lephare[filtername]
        photom_table_to_save = self.cat_photom[filtername]
        cigale_table_to_save = self.cat_cigale[filtername] if self.cat_cigale[filtername] is not None and 'cigale' not in tables_to_remove else None
        bd_table_to_save = self.cat_bd[filtername] if self.cat_bd[filtername] is not None and 'bd' not in tables_to_remove else None

        print(f"Original number of columns: lephare={len(lephare_table_to_save.colnames)}, photometry={len(photom_table_to_save.colnames)}, cigale={len(cigale_table_to_save.colnames) if cigale_table_to_save is not None else 'N/A'}, bd={len(bd_table_to_save.colnames) if bd_table_to_save is not None else 'N/A'}")
        if columns_to_keep is not None:
            lephare_table_to_save = lephare_table_to_save[columns_to_keep]
            photom_table_to_save = photom_table_to_save[columns_to_keep]
            if cigale_table_to_save is not None:
                cigale_table_to_save = cigale_table_to_save[columns_to_keep]
            if bd_table_to_save is not None:
                bd_table_to_save = bd_table_to_save[columns_to_keep]
        elif columns_to_remove is not None:
            lephare_table_to_save = lephare_table_to_save[[col for col in lephare_table_to_save.colnames if col not in columns_to_remove]]
            photom_table_to_save = photom_table_to_save[[col for col in photom_table_to_save.colnames if col not in columns_to_remove]]
            if cigale_table_to_save is not None:
                cigale_table_to_save = cigale_table_to_save[[col for col in cigale_table_to_save.colnames if col not in columns_to_remove]]
            if bd_table_to_save is not None:
                bd_table_to_save = bd_table_to_save[[col for col in bd_table_to_save.colnames if col not in columns_to_remove]]
        print(f"Number of columns after applying columns_to_keep/columns_to_remove: lephare={len(lephare_table_to_save.colnames)}, photometry={len(photom_table_to_save.colnames)}, cigale={len(cigale_table_to_save.colnames) if cigale_table_to_save is not None else 'N/A'}, bd={len(bd_table_to_save.colnames) if bd_table_to_save is not None else 'N/A'}")
                
        hdul = fits.HDUList([fits.PrimaryHDU()])
        if 'lephare' not in tables_to_remove:
            hdul.append(fits.BinTableHDU(lephare_table_to_save, name='lephare'))
        if 'photometry' not in tables_to_remove:
            hdul.append(fits.BinTableHDU(photom_table_to_save, name='photometry'))
        if cigale_table_to_save is not None:
            hdul.append(fits.BinTableHDU(cigale_table_to_save, name='cigale'))
        if bd_table_to_save is not None:
            hdul.append(fits.BinTableHDU(bd_table_to_save, name='bd'))
        hdul.writeto(path, overwrite=True)
        print(f"Catalog saved successfully to {path}.")

    # HDU layout of the v1.1 master catalog on disk.
    _MASTER_HDU_INDEX = {
        'photometry': 1, 'lephare': 2, 'se_aper': 3, 'cigale': 4,
        'ml_morpho': 5, 'bd': 6, 'galfitm_morpho': 7,
    }

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
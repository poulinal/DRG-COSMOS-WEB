# AP 2026

from abc import abstractmethod
# from tkinter.font import names

import numpy as np
from astropy.io import ascii, fits
from astropy.table import Table

class FitFileHandler():
    def __init__(self, path):
        self.fit_path : str = path
        self.hdu : fits.HDUList = None
        self.hdr : fits.Header = None
        self.fit_table : dict[str, Table] = {}
        
        
    def _load(self, hdu, *names):
        h = self._find_hdu(hdu, *names)
        return Table(h.data) if h is not None else None
    
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
        
    @abstractmethod
    def load_fit_file(self):
        pass
    
    @staticmethod
    def _getcol(p, *names):
        # Resolve the set of available column names explicitly. We must NOT fall
        # back to `n in p` on a (possibly masked) astropy Table: that iterates the
        # table and compares Rows, which raises a cryptic
        # "Cannot compare structured or void to non-void arrays" TypeError instead
        # of a clean column lookup.
        colnames = None
        if hasattr(p, 'colnames'):
            colnames = list(p.colnames)
        elif hasattr(p, 'dtype') and getattr(p.dtype, 'names', None):
            colnames = list(p.dtype.names)

        if colnames is not None:
            for n in names:
                if n in colnames:
                    return np.asarray(p[n])
            raise KeyError(f"none of {names} found; available columns: {colnames}")

        # dict-like fallback (plain dict, mapping, etc.)
        for n in names:
            try:
                if n in p:
                    return np.asarray(p[n])
            except TypeError:
                continue
        raise KeyError(f"none of {names} found")
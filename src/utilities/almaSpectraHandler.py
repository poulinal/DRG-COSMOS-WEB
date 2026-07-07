# AP 2026

from abc import abstractmethod
# from tkinter.font import names
from typing import Callable

import numpy as np
from astropy.io import ascii, fits
from astropy.table import Table
from astropy.wcs import FITSFixedWarning

from astropy import units as u
import os
import warnings

from utilities.fitFileHandler import FitFileHandler

class AlmaSpectraHandler(FitFileHandler):
    def __init__(self, path):
        super().__init__(path)
        self.hdu : dict[str, fits.HDUList] = {}
        self.hdr : dict[str, fits.Header] = {}
        
        self.load_all_fit_files_in_directory(self.fit_path)
        

    def load_fit_file(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"FIT file not found at {path}")

        # NB: don't use `with fits.open(...)` here -- we cache the HDUList in
        # self.hdu[path] for later access to the spectral data, and a context
        # manager would close it on block exit, making that data unreadable.
        hdu = fits.open(path)
        self.hdu[path] = hdu
        print(f" HDU Info: {hdu.info()}")
        self.hdr[path] = hdu[0].header if hdu is not None else None
        print(f" Header Info: {self.hdr[path]}")
        # self.fit_table = self._load(hdu, 'photom')
        self.fit_table[path] = None

        print(self.hdr[path]['CTRFRQ']/1e9, "GHz center") if 'CTRFRQ' in self.hdr[path] else print("CTRFRQ not found in header")
        print(self.hdr[path]['EFFBW']/1e9, "GHz effective bandwidth") if 'EFFBW' in self.hdr[path] else print("EFFBW not found in header")
        # exact windows: reconstruct the multiline spw record and read the GHz ranges
        spw = ''.join(c[1] for c in self.hdr[path].cards if c.keyword == 'HISTORY'
                    and str(c.value).lstrip().startswith(('spw','>' ))) if any(c.keyword == 'HISTORY' and str(c.value).lstrip().startswith(('spw','>' )) for c in self.hdr[path].cards) else print("spw not found in header")
        print(spw)

        return self.hdu[path], self.hdr[path], self.fit_table
            
    def load_all_fit_files_in_directory(self, directory_path):
        fit_files = [f for f in os.listdir(directory_path) if f.endswith('.fits')]
        
        for fit_file in fit_files:
            fit_file_path = os.path.join(directory_path, fit_file)
            self.load_fit_file(fit_file_path)
            
        return self.fit_table
    
    def check_fit_file_consistency(self):
        #check loaded files are present in hdu, hdr, and fit_table
        for path in self.hdu.keys():
            if path not in self.hdr or path not in self.fit_table:
                raise ValueError(f"Inconsistent data for {path}: missing hdr or fit_table")
    
    def get_all_fit_files_loaded(self, full : bool = True):
        self.check_fit_file_consistency()
        all_files_full = list(self.hdu.keys())
        if full:
            return all_files_full
        else:
            return [os.path.basename(f) for f in all_files_full]
              
    @staticmethod
    def _as_freq_cube(raw):
        """Return the data as a (nchan, ny, nx) cube regardless of input rank.

        ALMA/CASA images are stored (Stokes, FREQ, Dec, RA), which numpy reads
        as (nstokes, nchan, ny, nx). We assume a single Stokes plane (I) and
        drop it; 3D input is taken to be (nchan, ny, nx) already, and a bare 2D
        image is promoted to a single channel.
        """
        raw = np.asarray(raw)
        if raw.ndim == 4:
            return raw[0]                       # drop the (degenerate) Stokes axis
        if raw.ndim == 3:
            return raw
        if raw.ndim == 2:
            return raw[np.newaxis, ...]
        raise ValueError(f"Unsupported data rank {raw.ndim}; expected 2-4 axes.")

    @staticmethod
    def _channel_frequencies_ghz(wcs, hdr, nchan):
        """Per-channel frequency in GHz from the spectral WCS (fallback CTRFRQ)."""
        if wcs.has_spectral:
            spec = wcs.spectral.pixel_to_world(np.arange(nchan))
            return np.asarray(spec.to(u.GHz).value, dtype=float)
        # No spectral axis: single continuum plane -> use the rest frequency.
        ctr = hdr.get('RESTFRQ', hdr.get('CRVAL3'))
        return np.full(nchan, float(ctr) / 1e9 if ctr is not None else np.nan)

    @staticmethod
    def _robust_rms(plane):
        """MAD-based 1-sigma noise of a plane, NaN-aware (source is negligible)."""
        v = plane[np.isfinite(plane)]
        if v.size == 0:
            return np.nan
        med = np.median(v)
        return float(1.4826 * np.median(np.abs(v - med)))

    def get_alma_spectra(self, files: list[str] = None, box_halfwidth: int = None, box_halfwidth_arcsec: float = None,
                         object_position: tuple[float, float] = None, point_source : bool = True, verbose: bool = False) -> Table:
        """Extract a frequency/flux/error spectrum at one sky position.

        Works for both mfs continuum images (one channel) and spectral cubes
        (many channels): every channel of every file yields one row, and the
        combined table -- sorted by frequency -- is the spectrum at the target.
        Flux is in Jy/beam (== Jy for an unresolved source at the peak).

        Parameters
        ----------
        files : list[str], optional
            Paths to use; defaults to every loaded file.
        box_halfwidth : int
            How the per-channel flux is measured at the source pixel (xc, yc):
              * 0  -> forced photometry: the value at exactly (xc, yc). This is
                      the correct, *unbiased* choice for a spectrum -- use it
                      when hunting for lines.
              * >0 -> peak Jy/beam within a (2*hw+1) pixel box. Absorbs
                      astrometric slop, but biases every channel positive
                      (a noise peak is always >= the central pixel), so it is
                      only appropriate for a known continuum detection.
              * -1 -> box sized to the beam (BMAJ, rounded up), same caveat.
        object_position : tuple[float, float], optional
            Source (RA, Dec) in degrees. Defaults to each image's phase centre
            (CRVAL1/CRVAL2).

        Returns
        -------
        astropy.table.Table
            Sorted by frequency, columns: file, channel, frequency_ghz,
            flux_mjy, flux_err_mjy.
        """
        from astropy.wcs import WCS
        from time import perf_counter

        if files is None:
            files = self.get_all_fit_files_loaded()
            
        if box_halfwidth_arcsec is not None and box_halfwidth is not None:
            raise ValueError("Specify either box_halfwidth or box_halfwidth_arcsec, not both.")
        if box_halfwidth_arcsec is not None:
            # Convert arcseconds to pixels using the pixel scale from the first file's header
            if len(files) == 0:
                raise ValueError("No files provided to determine pixel scale.")
            first_hdr = self.hdr[files[0]]
            cdelt2 = abs(first_hdr.get('CDELT2', 0))  # degrees per pixel
            if cdelt2 <= 0:
                raise ValueError("Invalid CDELT2 value in header; cannot convert arcsec to pixels.")
            box_halfwidth = int(np.ceil((box_halfwidth_arcsec / 3600) / cdelt2))
            print(f"Box half-width converted from {box_halfwidth_arcsec} arcsec to {box_halfwidth} pixels.")
        else:
            if box_halfwidth is None:
                box_halfwidth = 0  # default to forced photometry if neither is specified

        rows = []
        for path in files:
            print(f"\nExtracting spectrum from {path}...")
            if path not in self.hdu:
                raise KeyError(f"{path} has not been loaded")
            hdr = self.hdr[path]
            if verbose:
                wcs = WCS(hdr)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FITSFixedWarning)
                    wcs = WCS(hdr)
            t0 = perf_counter()
            cube = np.asarray(self.hdu[path][0].data)           # forces the lazy disk read
            # print(f"  [timing] read data: {perf_counter() - t0:.2f}s")
            cube = self._as_freq_cube(cube)                     # (nchan, ny, nx)
            nchan, ny, nx = cube.shape

            # Locate the source pixel from its sky coordinates.
            ra, dec = object_position if object_position is not None else \
                (hdr['CRVAL1'], hdr['CRVAL2'])
            xc, yc = wcs.celestial.wcs_world2pix(ra, dec, 0)
            xc, yc = int(round(float(xc))), int(round(float(yc)))
            if not (0 <= xc < nx and 0 <= yc < ny):
                raise ValueError(
                    f"Position (RA={ra}, Dec={dec}) maps to pixel ({xc}, {yc}), "
                    f"outside the {nx}x{ny} image for {path}.")

            # Resolve the search-box half-width (in pixels).
            hw = box_halfwidth
            if hw == -1:
                # BMAJ and CDELT2 are both in degrees (CUNIT2='deg'), so
                # bmaj/cdelt is already the beam FWHM in pixels -- units cancel,
                # no arcsec conversion needed.
                bmaj, cdelt = hdr.get('BMAJ'), abs(hdr.get('CDELT2', 0) or 0)
                hw = int(np.ceil((bmaj / cdelt) / 2)) if bmaj is not None and cdelt > 0 else 0
                print(f"Box half-width autoset to {hw} pixels (beam size of {bmaj} deg (in arcsec: {bmaj * 3600:.1f})) for {path}")

            freqs = self._channel_frequencies_ghz(wcs, hdr, nchan)
            print(f"Channel frequencies range: {freqs[0]:.3f} - {freqs[-1]:.3f} GHz ({nchan} channels)")


            if point_source:
                # Flux for every channel at once (no per-channel Python loop).
                if hw <= 0:
                    flux = cube[:, yc, xc].astype(float)            # forced photometry
                else:
                    y0, y1 = max(yc - hw, 0), min(yc + hw + 1, ny)
                    x0, x1 = max(xc - hw, 0), min(xc + hw + 1, nx)
                    flux = np.nanmax(cube[:, y0:y1, x0:x1], axis=(1, 2)).astype(float)
            else:
                # Flux for every channel at once (no per-channel Python loop).
                # Pixels per beam (Gaussian): Omega_beam / pixel_area.
                bmaj, bmin = hdr['BMAJ'], hdr['BMIN']          # deg
                cdelt1, cdelt2 = abs(hdr['CDELT1']), abs(hdr['CDELT2'])  # deg
                npix_per_beam = (np.pi / (4 * np.log(2))) * bmaj * bmin / (cdelt1 * cdelt2)
                
                y0, y1 = max(yc - hw, 0), min(yc + hw + 1, ny)
                x0, x1 = max(xc - hw, 0), min(xc + hw + 1, nx)

                # Integrated flux density [Jy/beam summed -> Jy] over the aperture, per channel.
                region = cube[:, y0:y1, x0:x1]
                flux = np.nansum(region, axis=(1, 2)) / npix_per_beam


            # # MAD-based 1-sigma noise per channel. nanmedian sorts every
            # # pixel *and* can't use a fast partition (NaN handling), so it is
            # # the dominant cost on these pbcor cubes, which are NaN outside
            # # the primary beam. Two wins: (1) spatially subsample -- the noise
            # # is a robust statistic over >1e4 pixels, so every 2nd pixel is
            # # statistically identical; (2) the NaN footprint (primary beam) is
            # # the same for every channel, so reduce once to the pixels finite
            # # in all channels and use plain np.median (far faster than
            # # nanmedian) on the resulting dense, NaN-free array.
            # t0 = perf_counter()
            # sub = cube[:, ::2, ::2]
            # valid = np.isfinite(sub).all(axis=0)            # (ny', nx') beam footprint
            # vals = sub[:, valid]                            # (nchan, nvalid), NaN-free
            # if vals.shape[1] == 0:
            #     flux_err = np.full(nchan, np.nan)
            # else:
            #     med = np.median(vals, axis=1, keepdims=True)
            #     flux_err = 1.4826 * np.median(np.abs(vals - med), axis=1)
            #instead for speed, calculate flux_err as the stdev of 5 * the hw box around the source pixel, which is a good approximation of the noise in the image
            t0 = perf_counter()
            hw_stdev = int(hw * 5 / 2)
            y0, y1 = max(yc - hw_stdev, 0), min(yc + hw_stdev + 1, ny)
            x0, x1 = max(xc - hw_stdev, 0), min(xc + hw_stdev + 1, nx)
            flux_err = np.std(cube[:, y0:y1, x0:x1], axis=(1, 2)).astype(float)
            # print(f"  [timing] rms ({nchan} chan): {perf_counter() - t0:.2f}s")

            # BUNIT is constant per file -- resolve the scale factor once.
            bunit = hdr.get('BUNIT', '').lower()
            if bunit in ('jy/beam', 'jybm'):
                flux *= 1e3
                flux_err *= 1e3
            elif bunit in ('mjybm', 'mjyb'):
                pass
            else:
                print(f"Warning: unrecognized BUNIT '{bunit}' in {path}; assuming mJy/beam")

            for ch in range(nchan):
                rows.append((path, ch, float(freqs[ch]), float(flux[ch]),
                             float(flux_err[ch])))

        tbl = Table(rows=rows,
                    names=('file', 'channel', 'frequency_ghz', 'flux_mjy', 'flux_err_mjy'))
        tbl.sort('frequency_ghz')
        return tbl
                
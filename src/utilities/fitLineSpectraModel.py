# AP 2026

from typing import Callable
from .lineSpectraModelsEnum import LineSpectraModelsEnum
import numpy as np
from astropy.table import Table
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from astropy import units as u
from astropy import constants as const

# Splatalogue exports (and astroquery.Splatalogue tables) name the rest-frequency
# column inconsistently across versions. Try these in order.
_REST_FREQ_COLUMNS = (
    "Freq-GHz(rest frame,redshifted)",
    "Meas Freq-GHz(rest frame,redshifted)",
    "Freq-GHz",
    "orderedfreq", # in MHz
    "rest_freq_GHz",
    "rest_freq",
    "Ordered Frequency (GHz) (rest frame", 
)
_SPECIES_COLUMNS = ("Species", "name", "molecule", "Chemical Name")

_C_KMS = const.c.to(u.km/u.s).value  # speed of light, km/s

class FitLineSpectraModel:
    """
    A class to fit a line spectra with a given model. This class takes a line spectra model as input and provides a method to fit the model to the provided data.
    """
    def __init__(self, line_spectra_model: LineSpectraModelsEnum = LineSpectraModelsEnum.DoubleGaussian, list_molecule_lines_file: str = None):
        self.line_spectra_model = line_spectra_model
        self.list_molecule_lines_file = list_molecule_lines_file

    def set_line_spectra_model(self, line_spectra_model: LineSpectraModelsEnum):
        """
        Set the line spectra model to be used for fitting.

        Parameters:
        line_spectra_model (LineSpectraModelsEnum): The line spectra model to be used for fitting.
        """
        self.line_spectra_model = line_spectra_model
        
    def set_line_spectra_data(self, line_spectra_data: np.ndarray):
        """
        Set the line spectra data to be fitted.

        Parameters:
        line_spectra_data (np.ndarray): The line spectra data to be fitted.
        """
        self.line_spectra_data = line_spectra_data
        
    def set_list_molecule_lines_file(self, list_molecule_lines_file: str):
        """
        Set the file containing the list of molecule lines.

        Parameters:
        list_molecule_lines_file (str): The file containing the list of molecule lines.
        """
        self.list_molecule_lines_file = list_molecule_lines_file
        
    def get_line_spectra_model(self) -> LineSpectraModelsEnum:
        """
        Get the current line spectra model.

        Returns:
        LineSpectraModelsEnum: The current line spectra model.
        """
        return self.line_spectra_model
    
    def get_line_spectra_data(self) -> np.ndarray:
        """
        Get the current line spectra data.

        Returns:
        np.ndarray: The current line spectra data.
        """
        return self.line_spectra_data
    
    def get_list_molecule_lines_file(self) -> str:
        """
        Get the current file containing the list of molecule lines.

        Returns:
        str: The current file containing the list of molecule lines.
        """
        return self.list_molecule_lines_file

    def fit(self, line_spectra_data: np.ndarray, verbose: bool = False, **kwargs) -> dict | None:
        # pass to the appropriate fitting function based on the selected model
        self.set_line_spectra_data(line_spectra_data)
        if self.line_spectra_model == LineSpectraModelsEnum.DoubleGaussian:
            return self._fit_double_gaussian(line_spectra_data, molecule_lines_file=self.list_molecule_lines_file, verbose=verbose, **kwargs)
        else:
            raise ValueError(f"Unsupported line spectra model: {self.line_spectra_model}")
        
    # ------------------------------------------------------------------ #
    # Model                                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _gaussian(x: np.ndarray, amp: float, mu: float, sigma: float) -> np.ndarray:
        """A single Gaussian: amp * exp(-(x - mu)^2 / (2 sigma^2))."""
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    @staticmethod
    def _make_shared_z_model(rest_freq_1: float, rest_freq_2: float) -> Callable:
        """
        Build a double-Gaussian model whose two centers share a single redshift.

        Both lines come from the same source, so their observed frequencies are
        tied to their (known) rest frequencies through one redshift:
            mu_i = rest_freq_i / (1 + z)
        The free parameters are therefore (z, amp1, sigma1, amp2, sigma2) rather
        than two independent centers -- this encodes the physical constraint that
        makes the fit well posed instead of an ambiguous 6-parameter blob.
        """
        def model(x, z, amp1, sigma1, amp2, sigma2):
            mu1 = rest_freq_1 / (1.0 + z)
            mu2 = rest_freq_2 / (1.0 + z)
            return (
                FitLineSpectraModel._gaussian(x, amp1, mu1, sigma1)
                + FitLineSpectraModel._gaussian(x, amp2, mu2, sigma2)
            )
        return model

    @staticmethod
    def _double_gaussian(x, amp1, mu1, sigma1, amp2, mu2, sigma2):
        """Two Gaussians with fully independent centers (no shared redshift)."""
        return (
            FitLineSpectraModel._gaussian(x, amp1, mu1, sigma1)
            + FitLineSpectraModel._gaussian(x, amp2, mu2, sigma2)
        )

    # ------------------------------------------------------------------ #
    # Line list loading                                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_molecule_lines(molecule_lines_file: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Load a splatalogue / astropy-readable line table.

        Returns:
            (species, rest_freqs) where rest_freqs is in GHz (assumed to match the
            frequency units of the input spectrum) and species are string labels
            used to prefer multi-molecule solutions.
        """
        table = Table.read(molecule_lines_file)

        freq_col = next((c for c in _REST_FREQ_COLUMNS if c in table.colnames), None)
        if freq_col is None:
            raise ValueError(
                f"Could not find a rest-frequency column in {molecule_lines_file}; "
                f"looked for {_REST_FREQ_COLUMNS}, found {table.colnames}."
            )
        species_col = next((c for c in _SPECIES_COLUMNS if c in table.colnames), None)

        raw_freqs = table[freq_col] * u.MHz
        # Convert to GHz if the column is in MHz (Splatalogue default), already in GHz,
        # or has no attached unit but the column name indicates GHz.
        unit_str = str(getattr(raw_freqs, "unit", ""))
        if "MHz" in unit_str:
            rest_freqs = raw_freqs.to(u.GHz)  # ensure it's a Quantity in GHz
            rest_freqs = rest_freqs.value  # convert to plain ndarray of GHz values
        elif "GHz" in unit_str or "GHz" in freq_col or "ghz" in freq_col.lower():
            # pass  # already in GHz
            #convert to GHz if the column is in GHz
            rest_freqs = raw_freqs.to(u.GHz)
            #convert to plain ndarray of GHz values
            rest_freqs = rest_freqs.value
        elif getattr(raw_freqs, "unit", None) is None:
            pass  # plain numeric values; assume GHz to match the input spectrum
        else:
            raise ValueError(
                f"Rest frequency column {freq_col} has unexpected unit {getattr(raw_freqs, 'unit', None)}; "
                f"expected MHz or GHz."
            )
        # Keep rest_freqs as a plain ndarray of GHz values: the spectrum's x axis
        # (and peak_freqs, curve_fit, etc.) are bare floats, and mixing a Quantity
        # in raises UnitConversionError on e.g. `rest_freqs / nu_obs - 1.0`.
        species = (
            np.asarray(table[species_col], dtype=str)
            if species_col is not None
            else np.array([f"line_{i}" for i in range(len(rest_freqs))])
        )

        good = np.isfinite(rest_freqs) & (rest_freqs > 0)
        return species[good], rest_freqs[good]

    # ------------------------------------------------------------------ #
    # Continuum                                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _estimate_continuum(x: np.ndarray, y: np.ndarray, degree: int = 1,
                            n_sigma: float = 3.0, n_iter: int = 5
                            ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate the continuum with an iterative sigma-clipped polynomial fit.

        Emission/absorption lines are outliers to the baseline, so we fit a
        low-order polynomial, clip channels more than `n_sigma` from it, and
        refit -- repeating until the clip mask stabilises. This keeps bright
        lines from dragging the continuum up (or absorption from pulling it down).

        Returns:
            (continuum evaluated at x, polynomial coefficients).
        """
        mask = np.isfinite(y)
        coeffs = np.zeros(degree + 1)
        for _ in range(n_iter):
            if mask.sum() <= degree + 1:
                break
            coeffs = np.polyfit(x[mask], y[mask], degree)
            resid = y - np.polyval(coeffs, x)
            std = np.std(resid[mask])
            if std == 0:
                break
            new_mask = np.isfinite(y) & (np.abs(resid) < n_sigma * std)
            if np.array_equal(new_mask, mask):
                break
            mask = new_mask
        return np.polyval(coeffs, x), coeffs

    # ------------------------------------------------------------------ #
    # Fitting                                                            #
    # ------------------------------------------------------------------ #
    def _fit_double_gaussian(self, data: np.ndarray, molecule_lines_file: str,
                             z_bounds: tuple[float, float] = (0.0, 15.0),
                             max_hypotheses: int = 20, max_peaks: int = 8,
                             vel_tol_kms: float = 300.0, shared_z_chi2_max: float = 2.0,
                             continuum_degree: int | None = 1,
                             verbose: bool = False) -> dict | None:
        """
        Fit the provided data to possible molecule lines via a Double Gaussian.

        The two Gaussians are constrained to share a single redshift, with their
        centers pinned to candidate rest frequencies drawn from the molecule line
        list. Candidate line pairs are generated by finding peaks in the spectrum,
        computing the redshift each (peak, line) pairing implies, and keeping pairs
        whose common redshift places a *second* line on a *second* peak. Pairs
        spanning two different molecules are preferred. Each surviving hypothesis
        is fit with weighted non-linear least squares and ranked by reduced chi^2.

        Args:
            data (np.ndarray): a 3-column array [frequency, flux density, error].
            molecule_lines_file (str): astropy-readable table of possible lines
                (splatalogue export); rest frequencies assumed to be in the same
                units as the spectrum's frequency column (GHz).
            z_bounds: allowed redshift range for candidate hypotheses.
            max_hypotheses: cap on the number of line pairs actually fit.

        Returns:
            dict with the best fit: shared redshift, per-Gaussian parameters,
            matched species/rest frequencies, redshifted (observed) centers,
            reduced chi^2, and the parameter covariance. None if nothing fit.
        """
        x = np.asarray(data[:, 0], dtype=float)
        y = np.asarray(data[:, 1], dtype=float)
        err = np.asarray(data[:, 2], dtype=float)

        # guard against zero/invalid errors so the weighting stays finite
        err = np.where(np.isfinite(err) & (err > 0), err, np.nanmedian(err[err > 0]) or 1.0)

        # Subtract the continuum before doing anything else. Without this a single
        # Gaussian will happily splay out to model a sloped/offset baseline (we saw
        # a sigma=4.5 GHz "line" at the band edge), swamping the real narrow lines.
        # The baseline is fit with iterative sigma-clipping so emission/absorption
        # channels don't drag it. All peak-finding and line fitting then run on the
        # continuum-subtracted flux; the fitted line amplitudes are relative to it.
        continuum = np.zeros_like(y)
        cont_coeffs = None
        if continuum_degree is not None:
            continuum, cont_coeffs = self._estimate_continuum(x, y, continuum_degree)
            y = y - continuum
            if verbose:
                print(f"Subtracted degree-{continuum_degree} continuum "
                      f"(coeffs {np.array2string(cont_coeffs, precision=3)}).")
        cont_info = {"degree": continuum_degree,
                     "coeffs": None if cont_coeffs is None else cont_coeffs.tolist()}

        species, rest_freqs = self._load_molecule_lines(molecule_lines_file)
        print(f"Loaded {rest_freqs} molecule lines from {molecule_lines_file}.") if verbose else None
        if rest_freqs.size == 0:
            raise ValueError("No usable molecule lines were loaded.")

        # noise-scaled peak detection on the observed spectrum. Keep only the most
        # prominent peaks: a low prominence floor lets many noise bumps through, and
        # those generate chance-aligned (and therefore misleading) line hypotheses.
        noise = np.nanmedian(err)
        peaks, props = find_peaks(y, prominence=3.0 * noise)
        if peaks.size == 0:
            return None
        order = np.argsort(props["prominences"])[::-1][:max_peaks]
        peak_freqs = x[peaks[order]]
        
        # print(f"Detected {len(peak_freqs)} peaks in the spectrum, using the {len(order)} most prominent for line fitting. Peaks found at frequencies: {peak_freqs}") if verbose else None

        # Tolerance for calling a predicted line "on" a peak. Two co-redshifted
        # lines never land *exactly* where the line list predicts (peculiar
        # velocities, line widths, catalog precision), so match within a velocity
        # window rather than a fixed number of channels. Keep a one-channel floor
        # so the window can't collapse below the spectral resolution.
        channel = np.median(np.diff(np.sort(x))) if x.size > 1 else 0.0
        vel_frac = vel_tol_kms / _C_KMS
        if verbose:
            print(f"Median channel width is {channel:.3e} GHz; matching lines within "
                  f"+/-{vel_tol_kms:.0f} km/s (~{vel_frac:.2e} fractional).")

        hypotheses = self._generate_hypotheses(
            peak_freqs, species, rest_freqs, z_bounds, channel, vel_frac
        )
        if verbose:
            print(f"Generated {len(hypotheses)} shared-redshift candidate line pairs.")
            for h in hypotheses[:5]:
                print(f"  {h['species'][0]} + {h['species'][1]} at z={h['z']:.4f} "
                      f"(score {h['score']:.3f})")

        # --- 1) shared-redshift attempt (preferred, physically constrained) ---
        best = None
        for hyp in hypotheses[:max_hypotheses]:
            fit = self._fit_hypothesis(x, y, err, hyp, z_bounds)
            if fit is not None and (best is None or fit["reduced_chi2"] < best["reduced_chi2"]):
                best = fit

        if best is not None and best["reduced_chi2"] <= shared_z_chi2_max:
            if verbose:
                print(f"Shared-z fit accepted: reduced chi^2 = {best['reduced_chi2']:.3f} "
                      f"at z = {best['redshift']:.4f}.")
            best["continuum"] = cont_info
            return best

        # --- 2) fall back to independent centers ---
        # No consistent single-redshift pair fit acceptably (or none was proposed),
        # so relax the constraint and fit two free centers, then report the redshift
        # each center implies from its nearest catalog line.
        if verbose:
            shared_chi2 = f"{best['reduced_chi2']:.3f}" if best else "n/a"
            print(f"Shared-z fit poor (reduced chi^2 = {shared_chi2} > {shared_z_chi2_max}); "
                  f"falling back to an independent-center double Gaussian.")

        # print(f"fit independent with rest_freqs: {rest_freqs}, species: {species}, z_bounds: {z_bounds}") if verbose else None
        free = self._fit_independent(x, y, err, peak_freqs, species, rest_freqs,
                                     z_bounds, verbose=verbose)
        # Keep whichever actually describes the data better.
        result = free if (free is not None and
                          (best is None or free["reduced_chi2"] < best["reduced_chi2"])) else best
        if result is not None:
            result["continuum"] = cont_info
        return result

    @staticmethod
    def _generate_hypotheses(peak_freqs, species, rest_freqs, z_bounds, channel, vel_frac):
        """
        Build ranked candidate line pairs from detected peaks.

        For every (peak, line) pairing we compute the implied redshift z; if a
        second line lands within a velocity window of a *different* peak at that
        same z, the pair is a hypothesis. The window at frequency f is
        max(f * vel_frac, channel) -- a fixed velocity offset plus a one-channel
        floor. Hypotheses are scored to prefer (a) two distinct molecules and
        (b) tight alignment with the observed peaks.
        """
        z_lo, z_hi = z_bounds
        seen = set()
        hypotheses = []

        for i, nu_obs in enumerate(peak_freqs):
            # redshift implied by placing each rest line on this peak
            implied_z = rest_freqs / nu_obs - 1.0
            valid = np.where((implied_z >= z_lo) & (implied_z <= z_hi))[0]
            for li in valid:
                z = implied_z[li]
                predicted = rest_freqs / (1.0 + z)  # observed positions at this z
                for j, other_peak in enumerate(peak_freqs):
                    if j == i:
                        continue
                    tol = max(other_peak * vel_frac, channel)
                    matches = np.where(np.abs(predicted - other_peak) <= tol)[0]
                    for mj in matches:
                        # Key on the line pair *and* the redshift: the same pair of
                        # lines is a genuinely different hypothesis at a different z,
                        # and keying on the pair alone would discard all but the first.
                        key = (*sorted((li, int(mj))), round(z, 4))
                        if key in seen:
                            continue
                        seen.add(key)
                        different_molecule = species[li] != species[mj]
                        alignment = abs(predicted[mj] - other_peak)
                        # lower score == better: reward distinct molecules, penalise misalignment
                        score = alignment / tol - (1.0 if different_molecule else 0.0)
                        hypotheses.append({
                            "z": z,
                            "rest_freq_1": rest_freqs[li],
                            "rest_freq_2": rest_freqs[mj],
                            "species": (species[li], species[mj]),
                            "score": score,
                        })

        hypotheses.sort(key=lambda h: h["score"])
        return hypotheses

    def _fit_hypothesis(self, x, y, err, hyp, z_bounds):
        """Weighted non-linear least-squares fit for one candidate line pair."""
        f1, f2 = hyp["rest_freq_1"], hyp["rest_freq_2"]
        model = self._make_shared_z_model(f1, f2)

        z0 = hyp["z"]
        amp_scale = float(np.nanmax(y))
        sigma0 = max(np.median(np.diff(np.sort(x))) if x.size > 1 else 1.0, 1e-6)
        span = float(np.ptp(x)) or 1.0

        # p0 = (z, amp1, sigma1, amp2, sigma2)
        p0 = [z0, amp_scale, sigma0, amp_scale, sigma0]
        lower = [z_bounds[0], 0.0, sigma0 * 0.1, 0.0, sigma0 * 0.1]
        upper = [z_bounds[1], 2.0 * amp_scale, span, 2.0 * amp_scale, span]

        try:
            popt, pcov = curve_fit(
                model, x, y, p0=p0, sigma=err, absolute_sigma=True,
                bounds=(lower, upper), maxfev=10000,
            )
        except (RuntimeError, ValueError):
            return None

        residual = (y - model(x, *popt)) / err
        dof = max(x.size - len(popt), 1)
        reduced_chi2 = float(np.sum(residual ** 2) / dof)

        z = float(popt[0])
        return {
            "method": "shared_z",
            "redshift": z,
            "gaussian_1": {"amplitude": float(popt[1]), "sigma": float(popt[2]),
                           "center": f1 / (1.0 + z), "rest_freq": f1,
                           "species": hyp["species"][0]},
            "gaussian_2": {"amplitude": float(popt[3]), "sigma": float(popt[4]),
                           "center": f2 / (1.0 + z), "rest_freq": f2,
                           "species": hyp["species"][1]},
            "reduced_chi2": reduced_chi2,
            "covariance": pcov,
            "params": popt,
        }

    # ------------------------------------------------------------------ #
    # Independent-center fallback                                        #
    # ------------------------------------------------------------------ #
    def _fit_independent(self, x, y, err, peak_freqs, species, rest_freqs,
                         z_bounds, verbose: bool = False):
        """
        Fit a double Gaussian with two *independent* centers.

        Used when no single-redshift pair describes the data well (e.g. two lines
        whose observed positions imply different redshifts). Candidate center
        pairs are drawn from the detected peaks; each pair is fit with six free
        parameters and ranked by reduced chi^2. Each fitted center is then
        annotated with the catalog line and redshift it implies -- so the caller
        still learns "center 1 ~ CO(5-4) at z=5.02, center 2 ~ H2O at z=4.94".
        """
        from itertools import combinations

        if len(peak_freqs) < 2:
            return None

        amp_scale = float(np.nanmax(y))
        sigma0 = max(np.median(np.diff(np.sort(x))) if x.size > 1 else 1.0, 1e-6)
        span = float(np.ptp(x)) or 1.0
        xmin, xmax = float(np.min(x)), float(np.max(x))

        # p = (amp1, mu1, sigma1, amp2, mu2, sigma2)
        lower = [0.0, xmin, sigma0 * 0.1, 0.0, xmin, sigma0 * 0.1]
        upper = [2.0 * amp_scale, xmax, span, 2.0 * amp_scale, xmax, span]

        best = None
        for i, j in combinations(range(len(peak_freqs)), 2):
            print(f"Fitting independent centers at peaks {peak_freqs[i]:.4f} and {peak_freqs[j]:.4f}...") if verbose else None
            mu1_0, mu2_0 = peak_freqs[i], peak_freqs[j]
            p0 = [amp_scale, mu1_0, sigma0, amp_scale, mu2_0, sigma0]
            try:
                popt, pcov = curve_fit(
                    self._double_gaussian, x, y, p0=p0, sigma=err,
                    absolute_sigma=True, bounds=(lower, upper), maxfev=10000,
                )
            except (RuntimeError, ValueError):
                continue

            residual = (y - self._double_gaussian(x, *popt)) / err
            dof = max(x.size - len(popt), 1)
            reduced_chi2 = float(np.sum(residual ** 2) / dof)

            if best is None or reduced_chi2 < best["reduced_chi2"]:
                g1, g2 = self._identify_pair(float(popt[1]), float(popt[4]),
                                             species, rest_freqs, z_bounds)
                
                print(f"Independent fit: reduced chi^2 = {reduced_chi2:.3f}; "
                      f"center1 {(popt[1])} : redshift {g1['redshift']:.4f}, "
                      f"center2 {(popt[4])} : redshift {g2['redshift']:.4f}.") if verbose else None
                best = {
                    "method": "independent",
                    "gaussian_1": {"amplitude": float(popt[0]), "center": float(popt[1]),
                                   "sigma": float(popt[2]), **g1},
                    "gaussian_2": {"amplitude": float(popt[3]), "center": float(popt[4]),
                                   "sigma": float(popt[5]), **g2},
                    "reduced_chi2": reduced_chi2,
                    "covariance": pcov,
                    "params": popt,
                }

        if best is not None and verbose:
            print(f"Independent fit: reduced chi^2 = {best['reduced_chi2']:.3f}; "
                  f"center1 {self._describe_center(best['gaussian_1'])}, "
                  f"center2 {self._describe_center(best['gaussian_2'])}.")
        return best

    @staticmethod
    def _describe_center(g: dict) -> str:
        """Human-readable one-liner for a fitted center, tolerating unmatched
        lines (redshift/species may be None when no catalog line is in range)."""
        if g["redshift"] is None:
            return f"{g['center']:.4f} GHz (no catalog line in z-range)"
        return f"{g['center']:.4f} GHz ~ {g['species']} (z={g['redshift']:.4f})"

    @staticmethod
    def _identify_pair(mu1, mu2, species, rest_freqs, z_bounds):
        """
        Assign two observed centers to catalog lines *jointly*.

        Each center alone is ambiguous (any line implies some redshift), so pick
        the pair of lines whose implied redshifts are most consistent -- i.e.
        minimise |z1 - z2|. This recovers the physically natural identification
        (both lines from ~one source) even though the centers were fit freely,
        and reports the small residual redshift offset between them.
        """
        z_lo, z_hi = z_bounds
        print(f"rest_freqs: {rest_freqs}, mu1={mu1:.4f}, mu2={mu2:.4f}, z_bounds=({z_lo}, {z_hi})")
        # redshift_z = rest_freq / observed_freq - 1
        z1 = rest_freqs / mu1 - 1.0
        z2 = rest_freqs / mu2 - 1.0
        v1 = np.where((z1 >= z_lo) & (z1 <= z_hi))[0]
        v2 = np.where((z2 >= z_lo) & (z2 <= z_hi))[0]
        
        print(f"v1 indices: {v1}, v2 indices: {v2}")
        print(f"z_lo={z_lo}, z_hi={z_hi}, mu1={mu1:.4f}, mu2={mu2:.4f}. z1={np.sort(z1)}, z2={np.sort(z2)}")
        
        print(f"z1 candidates: {z1[v1]} for mu1={mu1:.4f}, z2 candidates: {z2[v2]} for mu2={mu2:.4f}") if v1.size and v2.size else None

        def entry(zi, idx):
            return {"species": species[idx], "rest_freq": float(rest_freqs[idx]),
                    "redshift": float(zi[idx])}
        none = {"species": None, "rest_freq": None, "redshift": None}

        if v1.size and v2.size:
            # both centers have candidates: pick the most redshift-consistent pair
            diff = np.abs(z1[v1][:, None] - z2[v2][None, :])
            a, b = np.unravel_index(np.argmin(diff), diff.shape)
            return entry(z1, v1[a]), entry(z2, v2[b])
        # otherwise identify each center on its own (nearest line in frequency,
        # i.e. smallest redshift correction); a spurious center just stays None.
        g1 = entry(z1, v1[np.argmin(z1[v1])]) if v1.size else none
        print(f"Identified center1: {g1['species']} at z={g1['redshift']}" if g1['species'] else "Center1 has no catalog line in z-range.") 
        g2 = entry(z2, v2[np.argmin(z2[v2])]) if v2.size else none
        return g1, g2
        
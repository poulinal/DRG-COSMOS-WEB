"""
Autotuning driver for a CIGALE run
==================================

The systematic grid built from ``pcigale.ini`` can easily explode to hundreds
of millions of models, which is both slow and memory hungry, while most of that
grid is spent far away from the χ² minimum of any given object.

:class:`Autotuner` addresses this with two strategies:

* **Budget management** — before any fit, the grid is coarsened so that the
  total number of models stays below a budget (``1e6`` by default). Multi-valued
  axes are sub-sampled evenly, always keeping their endpoints, starting with the
  axis contributing the most models.

* **Per-object zoom refinement** — a first coarse fit of the whole catalogue
  gives, for each object, the grid parameters of its best model. Each object is
  then re-fitted on a grid *zoomed* around those best values: continuous axes
  are resampled with a finer spacing inside a window bracketing the best value,
  while discrete/tabulated axes (metallicity, ionisation parameter, …) are
  pinned to their best value since intermediate values have no template. This is
  repeated until the reduced χ² stops improving or a maximum number of rounds is
  reached.

The class is driven through the internals of the analysis module passed to it
(``_compute``, :class:`ParametersManager`, :class:`ObservationsManager`) rather
than by shelling out, so it reuses the exact same fitting code path as a normal
run.

This module lives in ``cigale-run`` (rather than in the pcigale submodule) so
the submodule stays pristine.
"""

import copy
from pathlib import Path

import numpy as np
from astropy.table import Column, Table

from pcigale.managers.observations import ObservationsManager
from pcigale.managers.parameters import ParametersManager
from pcigale.utils.console import INFO, WARNING, console
from pcigale.utils.io import read_table

# Default ceiling on the number of models in a single grid.
DEFAULT_MAX_MODELS = 1_000_000

# Parameters that vary continuously and can therefore be safely resampled at a
# finer resolution during refinement. Anything not listed here (metallicity,
# ionisation parameter, gas metallicity, electron density, discrete template
# indices, booleans, strings, …) is treated as discrete and pinned to its
# best-fit value during refinement. Module names that are not present simply
# have all of their parameters treated as discrete.
CONTINUOUS_PARAMS = {
    "sfh2exp": {"tau_main", "tau_burst", "f_burst", "age", "burst_age"},
    "sfhdelayed": {"tau_main", "age_main", "tau_burst", "age_burst", "f_burst"},
    "sfhdelayedbq": {"tau_main", "age_main", "age_bq", "r_sfr", "burst_age"},
    "sfhperiodic": {"period", "age", "tau"},
    "dustatt_modified_CF00": {"Av_ISM", "mu", "slope_ISM", "slope_BC"},
    "dustatt_modified_starburst": {
        "E_BV_lines",
        "E_BV_factor",
        "powerlaw_slope",
        "uv_bump_amplitude",
    },
    "dale2014": {"alpha", "fracAGN"},
    "skirtor2016": {"EBV", "delta", "fracAGN"},
    "fritz2006": {"fracAGN"},
}

# The redshift axis is handled specially: pinned to the best-fit redshift of
# each object rather than zoomed.
REDSHIFT_MODULE = "redshifting"
REDSHIFT_PARAM = "redshift"


def _is_seq(value):
    """Whether a configuration value is a genuine sequence (not a bare string)."""
    return isinstance(value, (list, tuple)) and not isinstance(value, str)


def _axis_len(value):
    """Number of grid points contributed by a single parameter value."""
    return len(value) if _is_seq(value) else 1


def count_models(conf):
    """Total number of models in the systematic grid described by ``conf``.

    This is the product, over all SED modules, of the number of parameter
    combinations of each module, and mirrors ``ParametersManagerGrid.size``
    without instantiating the manager (which mutates ``conf``).
    """
    total = 1
    for module in conf["sed_modules"]:
        module_combos = 1
        for value in conf["sed_modules_params"][module].values():
            module_combos *= max(_axis_len(value), 1)
        total *= module_combos
    return total


def _subsample_keep_endpoints(values, target_len):
    """Evenly sub-sample ``values`` down to ``target_len`` keeping the endpoints."""
    n = len(values)
    if target_len >= n:
        return list(values)
    if target_len <= 1:
        return [values[0]]
    idx = np.linspace(0, n - 1, target_len)
    idx = sorted(set(int(round(i)) for i in idx))
    return [values[i] for i in idx]


class Autotuner:
    """Drive an analysis module with grid-budget management and per-object
    zoom refinement.

    Parameters
    ----------
    module: AnalysisModule
        The analysis module instance whose internals are reused to fit the
        models (typically a ``PdfAnalysis``).
    conf: dictionary
        Contents of ``pcigale.ini`` as returned by ``Configuration.configuration``.
    """

    def __init__(self, module, conf):
        self.module = module
        self.conf = conf

        params = conf["analysis_params"]
        self.max_models = int(params.get("max_models", DEFAULT_MAX_MODELS))
        self.max_rounds = int(params.get("autotune_rounds", 3))
        self.tol = float(params.get("autotune_tol", 0.01))

    # ------------------------------------------------------------------ #
    # Budget management
    # ------------------------------------------------------------------ #
    def coarsen(self, conf):
        """Return a copy of ``conf`` whose grid fits within the model budget.

        The axis contributing the most models is halved repeatedly (keeping its
        endpoints) until the total is under ``max_models`` or no axis can be
        reduced further.
        """
        conf = copy.deepcopy(conf)
        total = count_models(conf)
        if total <= self.max_models:
            return conf

        console.print(
            f"{INFO} Grid has {total:,} models, above the budget of "
            f"{self.max_models:,}. Coarsening the grid."
        )

        while count_models(conf) > self.max_models:
            # Find the multi-valued axis with the most points.
            best = None  # (module, param, length)
            for module in conf["sed_modules"]:
                for param, value in conf["sed_modules_params"][module].items():
                    n = _axis_len(value)
                    if n >= 3 and (best is None or n > best[2]):
                        best = (module, param, n)

            if best is None:
                console.print(
                    f"{WARNING} Could not coarsen the grid below the budget: "
                    "no axis left with more than two values."
                )
                break

            module, param, n = best
            values = list(conf["sed_modules_params"][module][param])
            target = max(2, (n + 1) // 2)
            conf["sed_modules_params"][module][param] = _subsample_keep_endpoints(
                values, target
            )

        console.print(f"{INFO} Coarsened grid has {count_models(conf):,} models.")
        return conf

    # ------------------------------------------------------------------ #
    # Fitting helpers
    # ------------------------------------------------------------------ #
    def _fit(self, conf):
        """Run one fit and return ``(obs, params, results)``.

        This reuses the analysis module's own model computation and Bayesian
        analysis so the results are identical to a normal run.
        """
        # ``_compute`` caches the models on the module when there is a single
        # block; clear it so each grid is recomputed rather than reused.
        if hasattr(self.module, "_models"):
            del self.module._models

        params = ParametersManager(conf)
        obs = ObservationsManager(conf, params)
        results = self.module._compute(conf, obs, params)
        results.best.analyse_chi2()
        return obs, params, results

    def _reduced_chi2(self, obs, results):
        """Reduced χ² per object for the current results."""
        data = [obs.table[band].data for band in obs.tofit]
        nobs = np.count_nonzero(np.isfinite(data), axis=0)
        return results.best.chi2 / np.maximum(nobs - 1, 1)

    def _best_params(self, params, index):
        """Grid parameters (per module) of the best model of one object."""
        combined = params.from_index(int(index))
        return {
            module: dict(combined[i])
            for i, module in enumerate(params.modules)
        }

    # ------------------------------------------------------------------ #
    # Zoom refinement
    # ------------------------------------------------------------------ #
    @staticmethod
    def _refine_axis(values, best):
        """Resample a continuous axis with a finer spacing around ``best``.

        A window is opened between the neighbours bracketing the best value in
        the original grid and resampled with the same number of points, giving
        a finer resolution. Integer-valued axes are kept integer and positive
        axes are kept positive.
        """
        v = sorted(set(float(x) for x in values))
        n = len(v)
        if n < 2:
            return [best]

        best = float(best)
        idx = min(range(n), key=lambda k: abs(v[k] - best))
        low = v[idx - 1] if idx > 0 else v[idx] - (v[idx + 1] - v[idx])
        high = v[idx + 1] if idx < n - 1 else v[idx] + (v[idx] - v[idx - 1])

        if min(v) > 0:  # keep strictly positive axes positive
            low = max(low, min(v) / 10.0)
        if high <= low:
            return [best]

        new = np.linspace(low, high, n)

        if all(float(x).is_integer() for x in values):
            new = sorted(set(int(round(x)) for x in new))
        else:
            new = sorted(set(float(x) for x in new))

        return new if len(new) >= 1 else [best]

    def _zoom_conf(self, base_conf, best_params, redshift):
        """Build a per-object grid zoomed around ``best_params``.

        Continuous axes are resampled finer around their best value; every other
        axis is pinned to its best value. The redshift is pinned to the object's
        best-fit value.
        """
        conf = copy.deepcopy(base_conf)
        sm_params = conf["sed_modules_params"]

        for module in conf["sed_modules"]:
            continuous = CONTINUOUS_PARAMS.get(module, set())
            for param, orig_value in list(base_conf["sed_modules_params"][module].items()):
                if module == REDSHIFT_MODULE and param == REDSHIFT_PARAM:
                    sm_params[module][param] = [redshift]
                    continue

                best_value = best_params[module][param]
                if param in continuous and _axis_len(orig_value) >= 2:
                    sm_params[module][param] = self._refine_axis(
                        orig_value, best_value
                    )
                else:
                    # Discrete, tabulated, boolean or single-valued: pin it.
                    sm_params[module][param] = best_value

        return conf

    def _refine_object(self, obj_id, redshift, best_params, base_conf):
        """Iteratively zoom the grid for a single object.

        Returns the final ``(best_params, reduced_chi2, rounds)``.
        """
        # Build a single-row data file so the standard pipeline fits just this
        # object, and point a dedicated configuration at it.
        obj_conf = copy.deepcopy(base_conf)
        table = read_table(base_conf["data_file"])
        row = table[table["id"].astype(str) == str(obj_id)]
        obj_file = Path("out") / f"autotune_{obj_id}.fits"
        row.write(obj_file, format="fits", overwrite=True)
        obj_conf["data_file"] = str(obj_file)

        best = best_params
        best_chi2 = np.inf
        rounds = 0

        for r in range(1, self.max_rounds + 1):
            zoom = self._zoom_conf(obj_conf, best, redshift)
            zoom = self.coarsen(zoom)

            obs, params, results = self._fit(zoom)
            chi2 = float(self._reduced_chi2(obs, results)[0])
            new_best = self._best_params(params, results.best.index[0])

            improvement = (best_chi2 - chi2) / best_chi2 if np.isfinite(best_chi2) else 1.0
            console.print(
                f"{INFO} Object {obj_id}: round {r}, reduced χ² = {chi2:.4g} "
                f"(Δ = {improvement:+.2%})."
            )

            rounds = r
            if chi2 < best_chi2:
                best, best_chi2 = new_best, chi2

            if np.isfinite(improvement) and improvement < self.tol:
                console.print(
                    f"{INFO} Object {obj_id}: converged (improvement below "
                    f"{self.tol:.1%})."
                )
                break

        obj_file.unlink(missing_ok=True)
        return best, best_chi2, rounds

    # ------------------------------------------------------------------ #
    # Entry point
    # ------------------------------------------------------------------ #
    def run(self):
        """Run the autotuning process over the whole catalogue."""
        self.module.prepare_dirs()

        # --- Round 0: coarse fit of the whole catalogue ----------------- #
        console.rule("Autotune: coarse catalogue fit")
        coarse_conf = self.coarsen(self.conf)
        obs, params, results = self._fit(coarse_conf)
        chi2_0 = self._reduced_chi2(obs, results)

        ids = [str(i) for i in obs.table["id"]]
        redshifts = np.asarray(obs.table["redshift"])
        seeds = {
            obj_id: self._best_params(params, results.best.index[i])
            for i, obj_id in enumerate(ids)
        }

        # --- Per-object zoom refinement --------------------------------- #
        summary = {"id": [], "reduced_chi_square": [], "rounds": []}
        continuous_cols = {}

        for i, obj_id in enumerate(ids):
            console.rule(f"Autotune: refining {obj_id} ({i + 1}/{len(ids)})")
            best, chi2, rounds = self._refine_object(
                obj_id, float(redshifts[i]), seeds[obj_id], self.conf
            )

            summary["id"].append(obj_id)
            summary["reduced_chi_square"].append(chi2)
            summary["rounds"].append(rounds)

            for module in self.conf["sed_modules"]:
                for param in CONTINUOUS_PARAMS.get(module, set()):
                    if param in best.get(module, {}):
                        col = f"{module}.{param}"
                        continuous_cols.setdefault(col, []).append(
                            float(best[module][param])
                        )

            improvement = (chi2_0[i] - chi2) / chi2_0[i] if chi2_0[i] else 0.0
            console.print(
                f"{INFO} Object {obj_id}: reduced χ² {chi2_0[i]:.4g} → "
                f"{chi2:.4g} ({improvement:+.2%} over the coarse fit)."
            )

        self._save_summary(summary, continuous_cols)

    def _save_summary(self, summary, continuous_cols):
        """Write the per-object autotuning summary to ``out/autotune_results.fits``."""
        table = Table()
        table.add_column(Column(summary["id"], name="id"))
        table.add_column(
            Column(summary["reduced_chi_square"], name="best.reduced_chi_square")
        )
        table.add_column(Column(summary["rounds"], name="autotune.rounds"))
        for col, values in sorted(continuous_cols.items()):
            if len(values) == len(summary["id"]):
                table.add_column(Column(values, name=f"best.{col}"))

        out = Path("out")
        table.write(out / "autotune_results.txt", format="ascii.fixed_width",
                    delimiter=None, overwrite=True)
        table.write(out / "autotune_results.fits", format="fits", overwrite=True)
        console.print(
            f"{INFO} Autotuning results saved to {out / 'autotune_results.fits'}."
        )


def autotune_process(module, cfg):
    """Run the autotuning process for ``cfg`` using the analysis ``module``.

    Parameters
    ----------
    module: AnalysisModule
        The analysis module instance (e.g. from ``get_module(...)``) whose
        internals are reused to fit the models.
    cfg: dictionary
        Contents of pcigale.ini as returned by ``Configuration.configuration``.
    """
    Autotuner(module, cfg).run()

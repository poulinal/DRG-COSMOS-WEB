"""Automatic best-fit SED plotting for a CIGALE run.

This is the programmatic equivalent of running ``pcigale-plots sed`` with its
default options, so that plots are produced automatically at the end of a run.
One plot is written per object in ``outdir``, using the
``results.fits``/``observations.fits`` files produced by the analysis module.

It lives in ``cigale-run`` (rather than in the pcigale submodule) so the
submodule stays pristine.
"""

from pathlib import Path
from types import SimpleNamespace

from pcigale_plots.plot_types.sed import AVAILABLE_SERIES, SED
from pcigale.utils.console import INFO, console


def make_plots(cfg, outdir="out", sed_type="mJy", fmt="pdf"):
    """Generate the best-fit SED plots for the results of a run.

    Parameters
    ----------
    cfg: dictionary
        Contents of pcigale.ini in the form of a dictionary, as returned by
        ``Configuration.configuration``.
    outdir: str or pathlib.Path
        Directory containing the run results and where the plots are written.
        Defaults to ``"out"``, matching the analysis module's ``prepare_dirs``.
    sed_type: str
        ``"mJy"`` (observed frame, flux) or ``"lum"`` (rest frame, luminosity).
    fmt: str
        Output format, e.g. ``"pdf"`` or ``"png"``.
    """
    outdir = Path(outdir)

    # The SED plotter expects an object exposing the raw configuration through a
    # ``.config`` attribute (as the Configuration object does), whereas here we
    # only have the validated dictionary. Wrap it so the plotter is reused
    # unchanged.
    config = SimpleNamespace(config=cfg)

    console.print(f"{INFO} Generating the best-fit SED plots.")
    SED(
        config,
        sed_type=sed_type,
        nologo=False,
        xrange=(False, False),
        yrange=(False, False),
        series=list(AVAILABLE_SERIES),
        format=fmt,
        outdir=outdir,
    )

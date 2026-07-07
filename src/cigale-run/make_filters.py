"""Generate CIGALE transmission files for radio/mm bands, then register them.

CIGALE has no built-in VLA 3 GHz or ALMA filters, so we create narrow tophat
passbands (effectively monochromatic, which is the right approximation for an
unresolved radio/mm continuum point) and add them to the filter database.

Usage:
    python make_filters.py          # writes the .dat files
    pcigale-filters add vla.S.dat alma.band3.dat alma.champs.dat   # register them
"""

import numpy as np
import scipy.constants as cst

# Edit these to match your actual measurements. (name, description, central frequency [GHz])
BANDS = [
    ("vla.S",       "VLA S-band 3 GHz continuum",      3.0),
    ("alma.A3Band3",  "ALMA A3 continuum in Band 3",             100),
    ("alma.A3Band4",  "ALMA A3 continuum in Band 4",             145),
    ("alma.A3Band6",  "ALMA A3 continuum in Band 6",         250),
    ("alma.A3Band7",  "ALMA A3 continuum in Band 7",         345), 
    ("alma.A3Band8",  "ALMA A3 continuum in Band 8",         460),  
    ("alma.champs", "ALMA CHAMPS continuum",         230),   # <-- set CHAMPS freq
]

FRAC_WIDTH = 0.02   # fractional bandwidth of the tophat (2% -> effectively monochromatic)
N_POINTS = 9        # points across the band (need >2 for the trapz integration)


def write_filter(name, desc, freq_ghz, frac_width=FRAC_WIDTH, npoints=N_POINTS):
    # Central wavelength in metres, then Angstrom (CIGALE expects Angstrom).
    lam_centre_m = cst.c / (freq_ghz * 1e9)
    lam_centre_A = lam_centre_m * 1e10

    half = 0.5 * frac_width * lam_centre_A
    wl = np.linspace(lam_centre_A - half, lam_centre_A + half, npoints)
    tr = np.ones_like(wl)          # flat tophat; "energy" type -> used as-is

    fname = f"{name}.dat"
    header = f"# {name}\n# energy\n# {desc}\n"
    with open(fname, "w") as f:
        f.write(header)
        for w, t in zip(wl, tr):
            f.write(f"{w:.6e} {t:.6f}\n")
    print(f"wrote {fname}: {freq_ghz} GHz -> {lam_centre_A:.3e} A ({npoints} pts)")


if __name__ == "__main__":
    for name, desc, freq in BANDS:
        write_filter(name, desc, freq)
    print("\nNow register them:")
    print("  pcigale-filters add " + " ".join(f"{b[0]}.dat" for b in BANDS))

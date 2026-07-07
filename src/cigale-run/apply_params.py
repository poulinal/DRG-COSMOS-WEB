"""Apply the tuned CIGALE parameter grid to pcigale.ini.

Run this AFTER `config.generate_conf()`, because genconf regenerates every
module section back to its defaults. Workflow:

    1. write the data FITS, set data_file + sed_modules + analysis_method in pcigale.ini
    2. config.generate_conf()      (run_cigale.py with genconf uncommented)
    3. python apply_params.py      <-- this script, re-applies the grid
    4. config ... .process(cfg)    (run_cigale.py with run uncommented)

It asserts each default value is present exactly once, so it is safe to run
only on a freshly generated pcigale.ini (a second run will fail loudly rather
than corrupt anything). Grid tuned for the z~5.025 dusty red galaxy with AGN.
"""

from pathlib import Path
from argparse import ArgumentParser

INI = Path(__file__).parent / "pcigale.ini"


def repl(s, old, new):
    assert s.count(old) == 1, f"expected 1 match for {old!r}, got {s.count(old)} (run genconf first?)"
    return s.replace(old, new)


def main(dust_attenuation_module="dustatt_modified_CF00"):
    s = INI.read_text()

    # SFH -- age_main capped < ~1163 Myr (age of universe at z=5.025)
    s = repl(s, "    tau_main = 2000.0", "    tau_main = 0.1, 500, 1000, 3000 # Default 2000.0")
    s = repl(s, "    age_main = 5000",   "    age_main = 100, 1000, 1500 # Default 5000") 
    
    # SSP -- Chabrier IMF
    s = repl(s, "    imf = 0", "    imf = 1 # Default 0 (Salpeter)")
    
    # Nebular
    s = repl(s, "    logU = -2.0", "    logU = -3.6, -2.0, -1.5 # Default -2.0")
    s = repl(s, "    zgas = 0.02", "    zgas = 0.001, 0.016, 0.033 # Default 0.02")
    s = repl(s, "    ne = 100", "    ne = 10, 100, 1000 # Default 100")
    
    # Dust attenuation -- wide E(B-V) grid for a dusty galaxy
    if dust_attenuation_module == "dustatt_modified_starburst":
        s = repl(s, "    E_BV_lines = 0.3", "    E_BV_lines = 0.1, 0.5, 1.3, 2.1, 3.0 # Default 0.3")
        s = repl(s, "    powerlaw_slope = 0.0", "    powerlaw_slope = -1.0, 0.0, 1.0 # Default 0.0")
    elif dust_attenuation_module == "dustatt_modified_CF00":
        s = repl(s, "    Av_ISM = 1.0", "    Av_ISM = 0.5, 1.5, 2.0, 3.0, 4.0 # Default 1.0")
        s = repl(s, "    mu = 0.4", "    mu = 0.01, 0.3, 0.7 # Default 0.4")
        s = repl(s, "    slope_ISM = -0.7", "    slope_ISM = -3, 0.3, 3 # Default -0.7")
        s = repl(s, "    slope_BC = -1.3", "    slope_BC = -3, 0.3, 3 # Default -1.3")
    
    # dale2014 -- SF-heated dust shape; its fracAGN stays 0 (skirtor handles AGN)
    s = repl(s, "    alpha = 2.0", "    alpha = 1.0, 3.0 # Default 2.0")
    
    # skirtor2016 -- vary AGN fraction + type-1/type-2 viewing angle, fix torus geometry
    s = repl(s, "    fracAGN = 0.1", "    fracAGN = 0.0, 0.3, 0.5 # Default 0.1")
    s = repl(s, "    delta = 0", "    delta = 0, 1 # Default 0")
    s = repl(s, "    i = 30", "    i = 0, 30, 90 # Default 30")
    
    # radio -- span radio-quiet -> radio-loud AGN
    # s = repl(s, "    R_agn = 10", "    R_agn = 1, 10, 100 # Default 10")
    
    # analysis -- curated output variables, save best SED, mock validation
    import re
    if dust_attenuation_module == "dustatt_modified_starburst":
        s = re.sub(r"  variables = .*",
                   "  variables = stellar.m_star, sfh.sfr, sfh.sfr100Myrs, sfh.age_main, "
                   "dust.luminosity, agn.fracAGN, agn.luminosity, agn.accretion_power, "
                   "attenuation.E_BV_lines, attenuation.E_BVs, radio.R_agn, "
                   "radio.P_agn_1p4GHz, universe.redshift",
                   s, count=1)
    elif dust_attenuation_module == "dustatt_modified_CF00":
        s = re.sub(r"  variables = .*",
                   "  variables = stellar.m_star, sfh.sfr, sfh.sfr100Myrs, sfh.age_main, "
                   "dust.luminosity, agn.fracAGN, agn.luminosity, agn.accretion_power, "
                   "attenuation.Av_ISM, attenuation.Av_BC, radio.R_agn, "
                   "radio.P_agn_1p4GHz, universe.redshift",
                   s, count=1)
        
    s = repl(s, "  save_best_sed = False", "  save_best_sed = True")
    s = repl(s, "  mock_flag = False", "  mock_flag = True")

    INI.write_text(s)
    print("applied tuned grid to pcigale.ini")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dust-attenuation-module", default="dustatt_modified_CF00", choices=["dustatt_modified_CF00", "dustatt_modified_starburst"], help="Dust attenuation module to use")
    args = parser.parse_args()

    args = parser.parse_args()
    if args.dust_attenuation_module is None:
        print("No dust attenuation module specified, using default: dustatt_modified_CF00")
        args.dust_attenuation_module = "dustatt_modified_CF00"
    elif args.dust_attenuation_module not in ["dustatt_modified_CF00", "dustatt_modified_starburst"]:
        raise ValueError(f"Invalid dust attenuation module: {args.dust_attenuation_module}")
    elif args.dust_attenuation_module == "dustatt_modified_CF00":
        print("Applying tuned grid for dustatt_modified_CF00...")
    elif args.dust_attenuation_module == "dustatt_modified_starburst":
        print("Applying tuned grid for dustatt_modified_starburst...")
    main(dust_attenuation_module=args.dust_attenuation_module)  # or "dustatt_modified_starburst"

from pathlib import Path

from pcigale.session.configuration import Configuration
from pcigale.analysis_modules import get_module
from argparse import ArgumentParser

import multiprocessing as mp

parser = ArgumentParser()
parser.add_argument("--cigale-step", default="run", choices=["init", "genconf", "run", "autotune-run"], help="CIGALE step to run")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    args = parser.parse_args()
    config = Configuration(Path(__file__).parent / "pcigale.ini")
    # path to the ini
    
    if args.cigale_step == "init":
        # --- pcigale init ---  (only if pcigale.ini doesn't exist yet)
        config.create_blank_conf()
        # now edit pcigale.ini: data_file, sed_modules, analysis_method, etc.
    elif args.cigale_step == "genconf":
        # --- pcigale genconf ---
        config.generate_conf()
        # edit the module parameter values in pcigale.ini if needed
        from apply_params import main as apply_params_main

        apply_params_main(dust_attenuation_module="dustatt_modified_CF00")  # or "dustatt_modified_starburst"
        # apply_params_main(dust_attenuation_module="dustatt_modified_starburst")
    elif args.cigale_step == "run":
        # --- pcigale run ---
        cfg = config.configuration                 # validated dict
        get_module(cfg["analysis_method"]).process(cfg)

        #auto make plots for the results (equivalent of `pcigale-plots sed`)
        from make_plots import make_plots
        make_plots(cfg)
    elif args.cigale_step == "autotune-run":
        #strategically manage grid to keep the number of models below 1e6, while still exploring the parameter space and reducing chi^2
        cfg = config.configuration                 # validated dict
        from autotune_cigale import autotune_process
        autotune_process(get_module(cfg["analysis_method"]), cfg)

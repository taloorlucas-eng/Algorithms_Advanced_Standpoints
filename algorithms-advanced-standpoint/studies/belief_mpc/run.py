"""Explicit lightweight and benchmark entry points; see README for test status."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from model import (make_settings, replace, run_planning_experiment, run_closed_loop,
                   report_closed_loop, run_sweep)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['planning', 'smoke', 'sweep'], default='planning')
    args = parser.parse_args()
    out = Path(__file__).resolve().parent / 'outputs' / args.mode
    cfg = make_settings('double_integrator')
    if args.mode == 'sweep':
        run_sweep(cfg, str(out), horizons=(5, 15, 30))
    else:
        cfg = replace(cfg, H=5, T=8, trials=1, lbfgs_outer_steps=3, lbfgs_max_iter=10)
        if args.mode == 'planning':
            run_planning_experiment(cfg, output_dir=str(out))
        else:
            result = run_closed_loop(cfg)
            report_closed_loop(result, str(out), print_steps=False)

if __name__ == '__main__':
    main()

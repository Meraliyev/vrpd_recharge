import os, json, glob, shutil
from main import run_experiment, load_config

def run_batch(configs, seeds):
    run_dirs=[]
    for cfg_path in configs:
        base = load_config(cfg_path)
        for s in seeds:
            cfg = dict(base); cfg['seed']=int(s)
            rd, _ = run_experiment(cfg)
            run_dirs.append(rd)
    return run_dirs

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='+', default=['config.json'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[1,2,3,4,5])
    args = ap.parse_args()
    run_dirs = run_batch(args.configs, args.seeds)
    print("Runs:", "\n".join(run_dirs))

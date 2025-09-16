import argparse, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main(pattern, out_png=None):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f'No files match: {pattern}')
    dfs = [pd.read_csv(p) for p in files]
    df = pd.concat(dfs, ignore_index=True)
    if 'method' not in df or 'makespan' not in df:
        raise ValueError('Results CSV must have columns: method, makespan')
    g = df.groupby('method')['makespan']
    mu = g.mean().sort_values()
    se = g.sem().reindex(mu.index)
    plt.figure()
    plt.bar(range(len(mu)), mu.values, yerr=se.values, capsize=4)
    plt.xticks(range(len(mu)), mu.index, rotation=20, ha='right')
    plt.ylabel('Makespan (mean ± s.e.)')
    plt.title('Comparison across seeds')
    out = out_png or 'aggregated_results.png'
    plt.savefig(out, bbox_inches='tight', dpi=150)
    print(f'Saved plot: {out}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('pattern', help='Glob for results CSVs, e.g. "results_seed*_N50.csv"')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    main(args.pattern, args.out)

import argparse, glob, pandas as pd
from math import sqrt

def wilcoxon_signed_rank(x, y):
    # Simple Wilcoxon signed-rank for paired samples (no ties handling sophistication)
    diffs = [a-b for a,b in zip(x,y) if a!=b]
    if not diffs:
        return 0.0, 1.0
    absdiff = list(map(abs, diffs))
    ranks = pd.Series(absdiff).rank().tolist()
    Wpos = sum(r for d,r in zip(diffs, ranks) if d>0)
    Wneg = sum(r for d,r in zip(diffs, ranks) if d<0)
    W = min(Wpos, Wneg)
    # Normal approximation
    n = len(diffs)
    mu = n*(n+1)/4.0
    sigma = sqrt(n*(n+1)*(2*n+1)/24.0)
    z = (W - mu)/sigma if sigma>0 else 0.0
    # two-sided p-value via normal approx
    try:
        import mpmath as mp
        p = 2*(1 - 0.5*(1+mp.erf(abs(z)/sqrt(2))))
    except Exception:
        # fallback rough
        p = 0.0
    return z, float(p)

def main(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f'No files match: {pattern}')
    # Expect all CSVs to have 'method' and 'makespan'
    rows = []
    for f in files:
        df = pd.read_csv(f)
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    methods = df['method'].unique().tolist()
    print("Methods:", methods)
    # Compare the last method vs first two
    baseline = [m for m in methods if 'CW' in m or 'ALNS' in m][0]
    target = [m for m in methods if 'RL' in m or 'Transformer' in m][-1]
    # Group by run (file) and extract paired values
    by_run = {}
    for f in files:
        df1 = pd.read_csv(f)
        vals = dict(zip(df1['method'], df1['makespan']))
        by_run[f] = vals
    xs = [by_run[f][baseline] for f in files if baseline in by_run[f] and target in by_run[f]]
    ys = [by_run[f][target] for f in files if baseline in by_run[f] and target in by_run[f]]
    z, p = wilcoxon_signed_rank(xs, ys)
    print(f"Wilcoxon baseline={baseline} vs target={target}: z={z:.3f}, p~{p:.3g} (lower makespan is better)")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('pattern', help='Glob like runs/*/results.csv')
    args = ap.parse_args()
    main(args.pattern)

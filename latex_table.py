import argparse, glob, pandas as pd, numpy as np

def main(pattern, out_tex='table_results.tex'):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f'No files match: {pattern}')
    frames=[]
    for run_csv in files:
        df = pd.read_csv(run_csv)
        df['run'] = run_csv
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    g = df.groupby('method')['makespan']
    mu = g.mean()
    se = g.sem()
    # build LaTeX
    rows=[]
    for m in mu.index:
        rows.append(f"{m} & {mu[m]:.2f} $\pm$ {se[m]:.2f} \\")
    body = "\n".join(rows)
    tex = f"""\begin{{table}}[ht]
\centering
\begin{{tabular}}{{l c}}
\toprule
Method & Makespan (mean $\pm$ s.e.) \\ \midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{Comparison across runs.}}
\label{{tab:results}}
\end{{table}}
"""
    with open(out_tex, 'w', encoding='utf-8') as f:
        f.write(tex)
    print(f"Saved LaTeX table: {out_tex}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('pattern', help='Glob like runs/*/results.csv')
    ap.add_argument('--out', default='table_results.tex')
    args = ap.parse_args()
    main(args.pattern, args.out)

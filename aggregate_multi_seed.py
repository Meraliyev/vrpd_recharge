
import os, re, glob, json, math, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_run_meta(path):
    # Expect path like runs/<tag>_seed<SEED>_N<N>/results.csv; return (seed, N, tag, run_dir)
    run_dir = os.path.dirname(path)
    base = os.path.basename(run_dir)
    mN = re.search(r"_N(\d+)", base)
    mS = re.search(r"_seed(\d+)", base)
    tag = base.split("_seed")[0] if "_seed" in base else base
    N = int(mN.group(1)) if mN else None
    seed = int(mS.group(1)) if mS else None
    return seed, N, tag, run_dir

def load_all(csv_glob):
    rows = []
    files = sorted(glob.glob(csv_glob))
    if not files:
        raise SystemExit(f"No files match: {csv_glob}")
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[WARN] Could not read {f}: {e}")
            continue
        seed, N, tag, run_dir = parse_run_meta(f)
        df = df.copy()
        df["seed"] = seed
        df["N"] = N
        df["tag"] = tag
        df["run_dir"] = run_dir
        rows.append(df)
    if not rows:
        raise SystemExit("No valid CSV files loaded.")
    all_df = pd.concat(rows, ignore_index=True)
    return all_df

def se(x):
    x = np.asarray(x, dtype=float)
    return np.std(x, ddof=1) / max(1, math.sqrt(len(x)))

def wilcoxon_signed_rank(x, y):
    # Paired Wilcoxon (two-sided) using normal approximation with continuity correction
    diffs = [a - b for a, b in zip(x, y) if (a - b) != 0]
    n = len(diffs)
    if n == 0:
        return 0.0, 1.0, 0.0, 0.0, 0.0  # z, p, Wpos, Wneg, r_rb
    absd = np.abs(diffs)
    ranks = pd.Series(absd).rank(method="average").values
    Wpos = float(np.sum(r for d, r in zip(diffs, ranks) if d > 0))
    Wneg = float(np.sum(r for d, r in zip(diffs, ranks) if d < 0))
    W = min(Wpos, Wneg)
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (W - mu - 0.5 * np.sign(W - mu)) / sigma if sigma > 0 else 0.0
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    R = n * (n + 1) / 2.0
    r_rb = (Wpos - Wneg) / R
    return float(z), float(p), float(Wpos), float(Wneg), float(r_rb)

def summarize(df):
    # Compute mean and s.e. per method
    g = df.groupby("method", as_index=False).agg(
        mean_mk=("makespan", "mean"),
        se_mk=("makespan", se),
        n=("makespan", "count")
    ).sort_values("mean_mk")
    return g

def aggregate(all_df, group_by_N=True):
    # If group_by_N and column N exists, return dict {N: df}; else {'ALL': df}
    out = {}
    if group_by_N and "N" in all_df.columns and not all_df["N"].isna().all():
        groups = sorted([int(n) for n in all_df["N"].dropna().unique()])
        for N in groups:
            sub = all_df[all_df["N"] == N]
            out[N] = summarize(sub)
    else:
        out["ALL"] = summarize(all_df)
    return out

def paired_tests(all_df):
    # Paired Wilcoxon: Proposed vs NN, Proposed vs ALNS
    piv = all_df.pivot_table(index=["N","seed"], columns="method", values="makespan", aggfunc="first")
    prop_cols = [c for c in piv.columns if ("PointerRL" in c) or ("Transformer" in c) or ("Proposed" in c)]
    if not prop_cols:
        return {}
    prop = prop_cols[0]
    out = {}
    for target in ["NN+LS+Drone", "ALNS+LS+Drone"]:
        if target not in piv.columns: continue
        sub = piv[[prop, target]].dropna()
        if sub.empty: continue
        x = sub[prop].tolist()
        y = sub[target].tolist()
        z, p, Wpos, Wneg, r_rb = wilcoxon_signed_rank(x, y)
        delta = float(np.mean(np.array(x) - np.array(y)))
        rel = float(np.mean((np.array(y) - np.array(x)) / np.array(y))) * 100.0
        out[target] = {"z": z, "p": p, "delta": delta, "rel_impr_pct": rel, "r_rb": r_rb, "n": len(sub)}
    return out

def write_latex_table(summary_df, out_tex, caption, label):
    def fmt(x): return f"{x:.3f}"
    lines = []
    lines.append(r"\begin{table}[!t]")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Mean makespan $\downarrow$ & s.e. \\")
    lines.append(r"\midrule")
    for _, r in summary_df.iterrows():
        lines.append(f"{r['method']} & {fmt(r['mean_mk'])} & {fmt(r['se_mk'])} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\end{table}")
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] wrote {out_tex}")

def write_latex_pvalues(pvals, out_tex, label="tab:pvalues"):
    if not pvals:
        print("[WARN] No p-values computed (no proposed method found or no overlap).")
        return
    lines = []
    lines.append(r"\begin{table}[!t]")
    lines.append(r"\caption{Paired Wilcoxon signed-rank tests (two-sided) comparing the proposed method against NN and ALNS.}")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Comparison & $n$ & $z$ & $p$ & Rank-biserial $r$ \\")
    lines.append(r"\midrule")
    for target, res in pvals.items():
        z = f"{res['z']:.3f}"; p = f"{res['p']:.3g}"; rrb = f"{res['r_rb']:.3f}"; n = f"{res['n']}"
        comp = f"Proposed vs. {target}"
        lines.append(f"{comp} & {n} & {z} & {p} & {rrb} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\end{table}")
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] wrote {out_tex}")

def make_barplot(summary_df, out_png, title='Makespan (mean ± s.e.)'):
    plt.figure(figsize=(7,4))
    x = np.arange(len(summary_df))
    means = summary_df['mean_mk'].values
    errs = summary_df['se_mk'].values
    plt.bar(x, means, yerr=errs, capsize=4)
    plt.xticks(x, summary_df['method'].tolist(), rotation=30, ha='right')
    plt.ylabel('Makespan (lower is better)')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[OK] wrote {out_png}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="runs/*/results.csv", help="Glob for results.csv files")
    ap.add_argument("--outdir", default="paper", help="Output dir for LaTeX/plots")
    ap.add_argument("--group_by_N", action="store_true", help="Aggregate separately per N")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    all_df = load_all(args.glob)
    groups = aggregate(all_df, group_by_N=args.group_by_N)

    for key, summ in groups.items():
        label_key = f"{key}" if key != "ALL" else "all"
        out_tex = os.path.join(args.outdir, f"table_results_{label_key}.tex")
        cap = "Aggregate makespan (mean ± s.e.) over seeds" + (f" for $N={key}$" if key!='ALL' else "") + "."
        write_latex_table(summ, out_tex, caption=cap, label=f"tab:agg_{label_key}")
        out_png = os.path.join(args.outdir, f"agg_bars_{label_key}.png")
        ttl = "Makespan (mean ± s.e.)" + (f" — N={key}" if key!='ALL' else "")
        make_barplot(summ, out_png, title=ttl)

    pvals = paired_tests(all_df)
    out_p = os.path.join(args.outdir, "pvalues.tex")
    write_latex_pvalues(pvals, out_p)

if __name__ == "__main__":
    main()

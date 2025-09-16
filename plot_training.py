import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main(log_csv, out_png=None):
    df = pd.read_csv(log_csv)
    if 'epoch' not in df or 'avg_makespan' not in df:
        raise ValueError('CSV must have columns: epoch, avg_makespan (see train_* CSV)')
    plt.figure()
    plt.plot(df['epoch'], df['avg_makespan'], label='Avg makespan')
    plt.xlabel('Epoch')
    plt.ylabel('Average makespan')
    plt.title('Training Curve')
    plt.legend()
    out = out_png or log_csv.replace('.csv', '.png')
    plt.savefig(out, bbox_inches='tight', dpi=150)
    print(f'Saved plot: {out}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('log_csv')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    main(args.log_csv, args.out)

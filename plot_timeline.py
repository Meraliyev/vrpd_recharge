import argparse, json
import matplotlib.pyplot as plt

def main(solutions_json, method='CW'):
    with open(solutions_json, 'r', encoding='utf-8') as f:
        sol = json.load(f)
    if method=='CW':
        e = sol['heuristics']['CW']['res'].get('events', [])
    elif method=='NN':
        e = sol['heuristics']['NN']['res'].get('events', [])
    elif method=='RL':
        if sol.get('rl') is None:
            raise SystemExit('RL solution not present in JSON')
        e = sol['rl'].get('events', [])
    else:
        raise SystemExit('method must be one of: NN, CW, RL')
    if not e:
        raise SystemExit('No events found; run with updated scheduler or ensure assignments exist.')
    plt.figure()
    for idx, (_,j,_, t0, t1, _) in enumerate(e):
        plt.plot([t0, t1], [idx, idx], linewidth=2)
        plt.scatter([t0, t1], [idx, idx])
        plt.text(t0, idx+0.1, f'cust {j}', fontsize=8)
    plt.xlabel('Time')
    plt.ylabel('Sortie index')
    plt.title(f'Drone sorties timeline — {method}')
    out = solutions_json.replace('.json', f'_{method}_timeline.png')
    plt.savefig(out, bbox_inches='tight', dpi=150)
    print(f'Saved: {out}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('solutions_json')
    ap.add_argument('--method', default='CW', choices=['NN','CW','RL'])
    args = ap.parse_args()
    main(args.solutions_json, args.method)

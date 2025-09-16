import argparse, json
import matplotlib.pyplot as plt

def plot_instance(ax, coords):
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    ax.scatter(xs[1:], ys[1:], marker='o', label='Customers')
    ax.scatter([coords[0][0]], [coords[0][1]], marker='s', s=80, label='Depot')
    for i,(x,y) in enumerate(coords):
        ax.annotate(str(i), (x,y), fontsize=8, xytext=(3,3), textcoords='offset points')

def plot_truck(ax, coords, tour):
    xs = [coords[i][0] for i in tour]
    ys = [coords[i][1] for i in tour]
    ax.plot(xs, ys, linestyle='-', linewidth=1.5, label='Truck route')

def plot_drone(ax, coords, assignments):
    # Draw polyline u->j->v for each sortie
    for (u,j,v) in assignments:
        x = [coords[u][0], coords[j][0], coords[v][0]]
        y = [coords[u][1], coords[j][1], coords[v][1]]
        ax.plot(x, y, linestyle='--', linewidth=1.0, label='Drone sortie')

def main(solutions_json, method='CW'):
    with open(solutions_json, 'r', encoding='utf-8') as f:
        sol = json.load(f)
    coords = sol['coords']
    if method=='CW':
        t = sol['heuristics']['CW']['res']['tour_after']
        a = sol['heuristics']['CW']['res']['assignments']
    elif method=='NN':
        t = sol['heuristics']['NN']['res']['tour_after']
        a = sol['heuristics']['NN']['res']['assignments']
    elif method=='RL':
        if sol.get('rl') is None:
            raise SystemExit('RL solution not present in JSON (run with --epochs>0 and --save_json)')
        t = sol['rl']['tour']
        a = sol['rl']['assignments']
    else:
        raise SystemExit('method must be one of: NN, CW, RL')
    plt.figure()
    ax = plt.gca()
    plot_instance(ax, coords)
    plot_truck(ax, coords, t)
    plot_drone(ax, coords, a)
    ax.set_title(f'Route visualization — {method}')
    ax.legend()
    out = solutions_json.replace('.json', f'_{method}_routes.png')
    plt.savefig(out, bbox_inches='tight', dpi=150)
    print(f'Saved: {out}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('solutions_json', help='solutions_seed*_N*.json')
    ap.add_argument('--method', default='CW', choices=['NN','CW','RL'])
    args = ap.parse_args()
    main(args.solutions_json, args.method)

# main.py (drop-in)
import os, json, argparse
import numpy as np
import pandas as pd

from data import generate_instance
from truck_heuristics import clarke_wright, two_opt, or_opt, total_distance
from drone_scheduler import greedy_drone_assignment

# Optional RL pointer (kept for compatibility; you can remove if not needed)
try:
    from rl_pointer import train_pointer, best_of_k, postprocess_truck as postprocess_truck_ptr
except Exception:
    train_pointer = None
    best_of_k = None
    def postprocess_truck_ptr(coords, tour): return tour

def nn_heuristic(coords, demands, capacity):
    N = len(coords)-1
    unv = set(range(1, N+1))
    routes = []
    cur, load = 0, 0
    route = [0]
    def d(a,b): return float(np.linalg.norm(coords[a]-coords[b]))
    while unv:
        nxt, best = None, 1e9
        for j in unv:
            if load + demands[j] <= capacity:
                dd = d(cur,j)
                if dd < best:
                    best, nxt = dd, j
        if nxt is None:
            route.append(0)
            routes.append(route)
            route, cur, load = [0], 0, 0
            continue
        route.append(nxt)
        cur = nxt
        load += demands[nxt]
        unv.remove(nxt)
    route.append(0)
    routes.append(route)
    tour = []
    for r in routes:
        if tour and tour[-1]!=0: tour.append(0)
        tour.extend(r)
    return tour

def build_truck_tour(coords, demands, capacity, truck_method='cw'):
    if truck_method == 'nn':
        t = nn_heuristic(coords, demands, capacity)
    else:
        t = clarke_wright(coords, demands, capacity)
    # ALNS option
    if truck_method == 'alns':
        try:
            from alns import alns_search
            t = alns_search(coords, demands, capacity, t, iters=2000)
        except Exception:
            # fallback if ALNS module not present
            pass
    # Local search clean-up
    t = two_opt(coords, t, 2000)
    t = or_opt(coords, t, 3, 2000)
    return t

def schedule_with_drone(coords, tour, v_truck, endurance, recharge, scheduler='beam', window=10):
    """
    Schedules drone sorties on a fixed truck tour using either:
      - 'beam' scheduler (preferred), or
      - 'greedy' scheduler (fallback/explicit).

    Returns a dict with keys: tour_after, assignments, makespan, truck_time, total_wait, events
    """
    if scheduler == 'beam':
        try:
            from drone_scheduler_beam import beam_search_assign
            res = beam_search_assign(
                coords, tour,
                v_truck=v_truck, v_drone=2.0 * v_truck,
                endurance=endurance, recharge=recharge,
                beam_size=8, top_k=5
            )
            # beam_search_assign already includes truck_time, total_wait, events
            return {
                'tour_after': res['tour_after'],
                'assignments': res['assignments'],
                'makespan': res['makesspan'] if 'makesspan' in res else res['makespan'],
                'truck_time': res.get('truck_time', float('nan')),
                'total_wait': res.get('total_wait', float('nan')),
                'events': res.get('events', [])
            }
        except Exception as e:
            print(f"[Warn] beam scheduler failed: {type(e).__name__}: {repr(e)}. Falling back to greedy.")

    # ---- Greedy fallback or explicit 'greedy' ----
    from drone_scheduler import greedy_drone_assignment
    res = greedy_drone_assignment(
        coords, tour,
        v_truck=v_truck,
        v_drone=2.0 * v_truck,
        endurance=endurance,
        recharge=recharge,
        window=window
    )
    # greedy_drone_assignment already returns these keys in your repo
    return {
        'tour_after': res['tour_after'],
        'assignments': res['assignments'],
        'makespan': res['makespan'],
        'truck_time': res.get('truck_time', float('nan')),
        'total_wait': res.get('total_wait', float('nan')),
        'events': res.get('events', [])
    }


def run(
    seed=1, N=60, capacity=50, v_truck=1.0, endurance=0.7, recharge=0.1,
    epochs=2000, K=32, eval_only=False, save_json=False,
    scheduler='beam', truck_method='cw', window=10,
    out_dir='runs', tag='exp'
):
    # Instance
    inst = generate_instance(N=N, seed=seed, capacity=capacity)
    coords, demands = inst['coords'], inst['demands']

    # Baseline 1: NN + LS + scheduler
    tour_nn = build_truck_tour(coords, demands, capacity, truck_method='nn')
    res_nn = schedule_with_drone(coords, tour_nn, v_truck, endurance, recharge, scheduler=scheduler, window=window)

    # Baseline 2: CW/ALNS + LS + scheduler
    tour_cw = build_truck_tour(coords, demands, capacity, truck_method=truck_method)
    res_cw = schedule_with_drone(coords, tour_cw, v_truck, endurance, recharge, scheduler=scheduler, window=window)

    results = []
    results.append({'method':'NN+LS+Drone','makespan':res_nn['makespan'],
                    'truck_time':res_nn.get('truck_time', np.nan),
                    'wait':res_nn.get('total_wait', np.nan)})
    results.append({'method':('ALNS+LS+Drone' if truck_method=='alns' else 'CW+LS+Drone'),
                    'makespan':res_cw['makespan'],
                    'truck_time':res_cw.get('truck_time', np.nan),
                    'wait':res_cw.get('total_wait', np.nan)})

    # Optional: Pointer RL (kept for compatibility)
    if (not eval_only) and (epochs>0) and train_pointer is not None and best_of_k is not None:
        policy, _ = train_pointer([inst], epochs=epochs, capacity=capacity,
                                  v_truck=v_truck, endurance=endurance, recharge=recharge,
                                  device='cpu', seed=seed)
        rl_best = best_of_k(policy, inst, capacity=capacity, v_truck=v_truck,
                            endurance=endurance, recharge=recharge, K=K, device='cpu', seed=seed+1)
        tour_rl = postprocess_truck_ptr(coords, rl_best[1])
        res_rl = schedule_with_drone(coords, tour_rl, v_truck, endurance, recharge, scheduler=scheduler, window=window)
        results.append({'method':f'PointerRL(K={K})+LS+Drone',
                        'makespan':res_rl['makespan'], 'truck_time':np.nan, 'wait':np.nan})
    # Print and save
    df = pd.DataFrame(results)
    print("\n=== Makespan (lower is better) ===")
    print(df.to_string(index=False))
    os.makedirs(out_dir, exist_ok=True)
    run_dir = os.path.join(out_dir, f"{tag}_seed{seed}_N{N}")
    os.makedirs(run_dir, exist_ok=True)
    csv_path = os.path.join(run_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    if save_json:
        sol = {
            'seed': int(seed), 'N': int(N), 'capacity': int(capacity),
            'v_truck': float(v_truck), 'endurance': float(endurance), 'recharge': float(recharge),
            'coords': coords.tolist(), 'demands': demands.tolist(),
            'heuristics': {
                'NN': {'tour': tour_nn, 'res': res_nn},
                ('ALNS' if truck_method=='alns' else 'CW'): {'tour': tour_cw, 'res': res_cw}
            }
        }
        with open(os.path.join(run_dir, "solutions.json"), "w", encoding="utf-8") as f:
            json.dump(sol, f)
        print(f"Saved: {os.path.join(run_dir, 'solutions.json')}")
    return df

def batch_run_from_config(cfg):
    N          = int(cfg.get('N', 60))
    capacity   = int(cfg.get('capacity', 50))
    v_truck    = float(cfg.get('v_truck', 1.0))
    endurance  = float(cfg.get('endurance', 0.7))
    recharge   = float(cfg.get('recharge', 0.1))
    epochs     = int(cfg.get('epochs', 0))
    K          = int(cfg.get('K', 0))
    eval_only  = bool(cfg.get('eval_only', False))
    save_json  = bool(cfg.get('save_json', True))
    scheduler    = cfg.get('scheduler', 'beam')
    truck_method = cfg.get('truck_method', 'cw')
    window       = int(cfg.get('window', 10))
    out_dir      = cfg.get('out_dir', 'runs')
    tag          = cfg.get('tag', 'exp')
    seeds = cfg.get('seeds', [cfg.get('seed', 1)])
    if isinstance(seeds, int):
        seeds = [seeds]
    for s in seeds:
        run(seed=int(s), N=N, capacity=capacity, v_truck=v_truck, endurance=endurance, recharge=recharge,
            epochs=epochs, K=K, eval_only=eval_only, save_json=save_json,
            scheduler=scheduler, truck_method=truck_method, window=window, out_dir=out_dir, tag=tag)
    return True

def load_config(path='config.json'):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='config.json')
    ap.add_argument('--use_config', action='store_true')
    # CLI fallbacks (optional)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--N', type=int, default=60)
    ap.add_argument('--capacity', type=int, default=50)
    ap.add_argument('--v_truck', type=float, default=1.0)
    ap.add_argument('--endurance', type=float, default=0.7)
    ap.add_argument('--recharge', type=float, default=0.1)
    ap.add_argument('--epochs', type=int, default=0)
    ap.add_argument('--K', type=int, default=0)
    ap.add_argument('--eval_only', action='store_true')
    ap.add_argument('--save_json', action='store_true')
    ap.add_argument('--scheduler', type=str, default='beam')
    ap.add_argument('--truck_method', type=str, default='cw')
    ap.add_argument('--window', type=int, default=10)
    ap.add_argument('--out_dir', type=str, default='runs')
    ap.add_argument('--tag', type=str, default='exp')
    args = ap.parse_args()

    if len(os.sys.argv) <= 2 or args.use_config:
        cfg = load_config(args.config)
        batch_run_from_config(cfg)
    else:
        run(**vars(args))

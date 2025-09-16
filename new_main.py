# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 14:39:10 2025

@author: sduai
"""

# main.py (drop-in, meta-enabled)
import os, json, argparse
import numpy as np
import pandas as pd

from data import generate_instance
from truck_heuristics import clarke_wright, two_opt, or_opt, total_distance
from drone_scheduler import greedy_drone_assignment

# Metaheuristics
from metaheuristics import sa_search, tabu_search, ga_search, vns_search

# Optional RL pointer (kept for compatibility)
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
    if truck_method == 'alns':
        try:
            from alns import alns_search
            t = alns_search(coords, demands, capacity, t, iters=2000)
        except Exception:
            pass
    t = two_opt(coords, t, 2000)
    t = or_opt(coords, t, 3, 2000)
    return t

def build_truck_tour_meta(coords, demands, capacity, method='sa'):
    base = build_truck_tour(coords, demands, capacity, truck_method='cw')
    if method == 'sa':
        t = sa_search(coords, base, iters=4000, T0=1.0, cooling=0.9995, seed=1)
    elif method == 'tabu':
        t = tabu_search(coords, base, iters=2000, tabu_tenure=25, seed=1)
    elif method == 'ga':
        t = ga_search(coords, base, pop_size=30, gens=200, mut_rate=0.2, seed=1)
    elif method == 'vns':
        t = vns_search(coords, base, k_max=3, iters=800, seed=1)
    else:
        t = base
    t = two_opt(coords, t, 1000)
    t = or_opt(coords, t, 3, 1000)
    return t

def schedule_with_drone(coords, tour, v_truck, endurance, recharge, scheduler='beam', window=10):
    # Try beam if available
    try:
        from drone_scheduler_beam import beam_search_assign
    except Exception:
        beam_search_assign = None

    if scheduler == 'beam' and beam_search_assign is not None:
        try:
            res = beam_search_assign(
                coords, tour,
                v_truck=v_truck, v_drone=2.0 * v_truck,
                endurance=endurance, recharge=recharge,
                beam_size=8, top_k=5
            )
            return {
                'tour_after': res['tour_after'],
                'assignments': res['assignments'],
                'makespan': res.get('makespan', res.get('makesspan')),
                'truck_time': res.get('truck_time', float('nan')),
                'total_wait': res.get('total_wait', float('nan')),
                'events': res.get('events', [])
            }
        except Exception as e:
            print(f"[Warn] beam scheduler failed: {type(e).__name__}: {e!r}. Falling back to greedy.")

    # Greedy fallback (or if scheduler == 'greedy')
    res = greedy_drone_assignment(
        coords, tour,
        v_truck=v_truck,
        v_drone=2.0 * v_truck,
        endurance=endurance,
        recharge=recharge,
        window=window
    )
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
    inst = generate_instance(N=N, seed=seed, capacity=capacity)
    coords, demands = inst['coords'], inst['demands']

    # NN baseline
    tour_nn = build_truck_tour(coords, demands, capacity, truck_method='nn')
    res_nn = schedule_with_drone(coords, tour_nn, v_truck, endurance, recharge, scheduler=scheduler, window=window)

    # CW/ALNS baseline
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

    # Metaheuristics: SA / Tabu / GA / VNS
    try:
        tour_sa = build_truck_tour_meta(coords, demands, capacity, 'sa')
        res_sa = schedule_with_drone(coords, tour_sa, v_truck, endurance, recharge, scheduler=scheduler, window=window)
        results.append({'method':'Meta-SA+LS+Drone','makespan':res_sa['makespan'],
                        'truck_time':res_sa.get('truck_time', np.nan),'wait':res_sa.get('total_wait', np.nan)})
    except Exception as e:
        print(f"[Warn] SA failed: {type(e).__name__}: {e!r}")

    try:
        tour_tabu = build_truck_tour_meta(coords, demands, capacity, 'tabu')
        res_tabu = schedule_with_drone(coords, tour_tabu, v_truck, endurance, recharge, scheduler=scheduler, window=window)
        results.append({'method':'Meta-Tabu+LS+Drone','makespan':res_tabu['makespan'],
                        'truck_time':res_tabu.get('truck_time', np.nan),'wait':res_tabu.get('total_wait', np.nan)})
    except Exception as e:
        print(f"[Warn] Tabu failed: {type(e).__name__}: {e!r}")

    try:
        tour_ga = build_truck_tour_meta(coords, demands, capacity, 'ga')
        res_ga = schedule_with_drone(coords, tour_ga, v_truck, endurance, recharge, scheduler=scheduler, window=window)
        results.append({'method':'Meta-GA+LS+Drone','makespan':res_ga['makespan'],
                        'truck_time':res_ga.get('truck_time', np.nan),'wait':res_ga.get('total_wait', np.nan)})
    except Exception as e:
        print(f"[Warn] GA failed: {type(e).__name__}: {e!r}")

    try:
        tour_vns = build_truck_tour_meta(coords, demands, capacity, 'vns')
        res_vns = schedule_with_drone(coords, tour_vns, v_truck, endurance, recharge, scheduler=scheduler, window=window)
        results.append({'method':'Meta-VNS+LS+Drone','makespan':res_vns['makespan'],
                        'truck_time':res_vns.get('truck_time', np.nan),'wait':res_vns.get('total_wait', np.nan)})
    except Exception as e:
        print(f"[Warn] VNS failed: {type(e).__name__}: {e!r}")

    # Optional: Pointer RL
    rl_detail = None
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
        rl_detail = {'tour': tour_rl, 'res': res_rl}

    # Save table
    df = pd.DataFrame(results)
    print("\n=== Makespan (lower is better) ===")
    print(df.to_string(index=False))
    os.makedirs(out_dir, exist_ok=True)
    run_dir = os.path.join(out_dir, f"{tag}_seed{seed}_N{N}")
    os.makedirs(run_dir, exist_ok=True)
    csv_path = os.path.join(run_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Save solutions.json (CW/ALNS + NN + RL)
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
        if rl_detail is not None:
            sol['rl'] = rl_detail
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

    # Accept "seeds" OR "seed_list" OR single "seed"
    seeds = cfg.get('seeds', cfg.get('seed_list', [cfg.get('seed', 1)]))
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

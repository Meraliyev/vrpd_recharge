
import os, json, argparse, numpy as np, pandas as pd
from data import generate_instance
from truck_heuristics import clarke_wright, two_opt, or_opt, total_distance
from drone_scheduler import greedy_drone_assignment
from metaheuristics import sa_search, tabu_search, ga_search, vns_search

try:
    from drone_scheduler_beam import beam_search_assign
except Exception:
    beam_search_assign = None

def nn_heuristic(coords, demands, capacity):
    N = len(coords)-1
    unv=set(range(1,N+1)); routes=[]; cur=0; load=0; route=[0]
    def d(a,b): import numpy as np; return float(np.linalg.norm(coords[a]-coords[b]))
    while unv:
        nxt=None; best=1e9
        for j in unv:
            if load + demands[j] <= capacity:
                dd=d(cur,j)
                if dd < best: best=dd; nxt=j
        if nxt is None:
            route.append(0); routes.append(route); route=[0]; cur=0; load=0; continue
        route.append(nxt); cur=nxt; load+=demands[nxt]; unv.remove(nxt)
    route.append(0); routes.append(route)
    tour=[]
    for r in routes:
        if tour and tour[-1]!=0: tour.append(0)
        tour.extend(r)
    return tour

def build_truck_tour(coords, demands, capacity, truck_method='cw'):
    if truck_method == 'nn':
        t = nn_heuristic(coords, demands, capacity)
    else:
        t = clarke_wright(coords, demands, capacity)
    t = two_opt(coords, t, 2000); t = or_opt(coords, t, 3, 2000)
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
    t = two_opt(coords, t, 1000); t = or_opt(coords, t, 3, 1000)
    return t

def schedule_with_drone(coords, tour, v_truck, endurance, recharge, scheduler='beam', window=10):
    if scheduler=='beam' and beam_search_assign is not None:
        try:
            res = beam_search_assign(coords, tour, v_truck=v_truck, v_drone=2.0*v_truck,
                                     endurance=endurance, recharge=recharge, beam_size=8, top_k=5)
            return {
                'tour_after': res['tour_after'], 'assignments': res['assignments'],
                'makespan': res['makespan'], 'truck_time': res.get('truck_time', np.nan),
                'total_wait': res.get('total_wait', np.nan), 'events': res.get('events', [])
            }
        except Exception as e:
            print(f"[Warn] beam scheduler failed: {type(e).__name__}: {e}. Falling back to greedy.")
    res = greedy_drone_assignment(coords, tour, v_truck=v_truck, v_drone=2.0*v_truck,
                                  endurance=endurance, recharge=recharge, window=window)
    return res

def run(seed=1, N=60, capacity=50, v_truck=1.0, endurance=0.7, recharge=0.1,
        scheduler='beam', window=10, out_dir='runs', tag='exp'):
    inst = generate_instance(N=N, seed=seed, capacity=capacity)
    coords, demands = inst['coords'], inst['demands']

    rows = []

    # NN baseline
    tour_nn = build_truck_tour(coords, demands, capacity, 'nn')
    res_nn = schedule_with_drone(coords, tour_nn, v_truck, endurance, recharge, scheduler, window)
    rows.append({'method':'NN+LS+Drone','makespan':res_nn['makespan'],
                 'truck_time':res_nn.get('truck_time', np.nan),'wait':res_nn.get('total_wait', np.nan)})

    # CW baseline
    tour_cw = build_truck_tour(coords, demands, capacity, 'cw')
    res_cw = schedule_with_drone(coords, tour_cw, v_truck, endurance, recharge, scheduler, window)
    rows.append({'method':'CW+LS+Drone','makespan':res_cw['makespan'],
                 'truck_time':res_cw.get('truck_time', np.nan),'wait':res_cw.get('total_wait', np.nan)})

    # Metaheuristics
    tour_sa = build_truck_tour_meta(coords, demands, capacity, 'sa')
    res_sa = schedule_with_drone(coords, tour_sa, v_truck, endurance, recharge, scheduler, window)
    rows.append({'method':'Meta-SA+LS+Drone','makespan':res_sa['makespan'],
                 'truck_time':res_sa.get('truck_time', np.nan),'wait':res_sa.get('total_wait', np.nan)})

    tour_tabu = build_truck_tour_meta(coords, demands, capacity, 'tabu')
    res_tabu = schedule_with_drone(coords, tour_tabu, v_truck, endurance, recharge, scheduler, window)
    rows.append({'method':'Meta-Tabu+LS+Drone','makespan':res_tabu['makespan'],
                 'truck_time':res_tabu.get('truck_time', np.nan),'wait':res_tabu.get('total_wait', np.nan)})

    tour_ga = build_truck_tour_meta(coords, demands, capacity, 'ga')
    res_ga = schedule_with_drone(coords, tour_ga, v_truck, endurance, recharge, scheduler, window)
    rows.append({'method':'Meta-GA+LS+Drone','makespan':res_ga['makespan'],
                 'truck_time':res_ga.get('truck_time', np.nan),'wait':res_ga.get('total_wait', np.nan)})

    tour_vns = build_truck_tour_meta(coords, demands, capacity, 'vns')
    res_vns = schedule_with_drone(coords, tour_vns, v_truck, endurance, recharge, scheduler, window)
    rows.append({'method':'Meta-VNS+LS+Drone','makespan':res_vns['makespan'],
                 'truck_time':res_vns.get('truck_time', np.nan),'wait':res_vns.get('total_wait', np.nan)})

    df = pd.DataFrame(rows)
    print("\\n=== Makespan (lower is better) ===")
    print(df.to_string(index=False))

    os.makedirs(out_dir, exist_ok=True)
    run_dir = os.path.join(out_dir, f"{tag}_seed{seed}_N{N}")
    os.makedirs(run_dir, exist_ok=True)
    df.to_csv(os.path.join(run_dir, "results_meta.csv"), index=False)
    print(f"Saved: {os.path.join(run_dir, 'results_meta.csv')}")

    # Save one solutions.json for viz (CW)
    sol = {'seed': int(seed), 'N': int(N),
           'coords': coords.tolist(), 'demands': demands.tolist(),
           'heuristics': {'CW': {'tour': tour_cw, 'res': res_cw}}}
    with open(os.path.join(run_dir, "solutions.json"), "w", encoding="utf-8") as f:
        json.dump(sol, f)
    print(f"Saved: {os.path.join(run_dir, 'solutions.json')}")
    return df

def load_config(path='config.json'):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='config.json')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--N', type=int, default=60)
    ap.add_argument('--capacity', type=int, default=50)
    ap.add_argument('--v_truck', type=float, default=1.0)
    ap.add_argument('--endurance', type=float, default=0.7)
    ap.add_argument('--recharge', type=float, default=0.1)
    ap.add_argument('--scheduler', type=str, default='beam')
    ap.add_argument('--window', type=int, default=10)
    ap.add_argument('--out_dir', type=str, default='runs')
    ap.add_argument('--tag', type=str, default='exp')
    args = ap.parse_args()

    if os.path.exists(args.config):
        cfg = load_config(args.config)
        seed = int(cfg.get('seed', args.seed))
        N = int(cfg.get('N', args.N))
        capacity = int(cfg.get('capacity', args.capacity))
        v_truck = float(cfg.get('v_truck', args.v_truck))
        endurance = float(cfg.get('endurance', args.endurance))
        recharge = float(cfg.get('recharge', args.recharge))
        scheduler = cfg.get('scheduler', args.scheduler)
        window = int(cfg.get('window', args.window))
        out_dir = cfg.get('out_dir', args.out_dir)
        tag = cfg.get('tag', args.tag)
    else:
        seed, N, capacity = args.seed, args.N, args.capacity
        v_truck, endurance, recharge = args.v_truck, args.endurance, args.recharge
        scheduler, window, out_dir, tag = args.scheduler, args.window, args.out_dir, args.tag

    run(seed=seed, N=N, capacity=capacity, v_truck=v_truck, endurance=endurance, recharge=recharge,
        scheduler=scheduler, window=window, out_dir=out_dir, tag=tag)

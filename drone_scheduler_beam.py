import numpy as np
from drone_scheduler import simulate_schedule, remove_customers_from_tour

def dist(a,b):
    return float(np.linalg.norm(a-b))

def precompute_candidates(coords, base_tour, demands, v_truck=1.0, v_drone=2.0, endurance=0.7, top_k=5, forward_window=10):
    N = len(coords)-1
    edges = [(base_tour[t], base_tour[t+1]) for t in range(len(base_tour)-1)]
    customers = [n for n in base_tour if n!=0]
    unique=[]
    seen=set()
    for x in customers:
        if x not in seen:
            unique.append(x); seen.add(x)
    candidates = [[] for _ in edges]
    for t,(u,v) in enumerate(edges):
        # pick forward_window rendezvous limited by index (we'll set v as fixed = base_tour[t+1] for simplicity)
        # We'll allow only rendezvous at v to keep branching reasonable, still very effective
        for j in unique:
            if j in (u,v): 
                continue
            # is sortie feasible?
            dtime = (dist(coords[u], coords[j]) + dist(coords[j], coords[v]))/v_drone
            if dtime <= endurance + 1e-12:
                # truck saving from removing j
                try:
                    idx = base_tour.index(j)
                    p, q = base_tour[idx-1], base_tour[idx+1]
                    saving = dist(coords[p], coords[j]) + dist(coords[j], coords[q]) - dist(coords[p], coords[q])
                except ValueError:
                    saving = 0.0
                score = saving  # could subtract small multiple of dtime
                candidates[t].append((score, (u,j,v), dtime, saving))
        # keep top_k by score
        candidates[t].sort(key=lambda x: -x[0])
        candidates[t] = candidates[t][:top_k]
    return candidates

def beam_search_assign(coords, base_tour, v_truck=1.0, v_drone=2.0, endurance=0.7, recharge=0.1, beam_size=8, top_k=5):
    """Beam search over assignments per edge with pruning; rendezvous fixed to immediate successor (u->v)."""
    candidates = precompute_candidates(coords, base_tour, None, v_truck, v_drone, endurance, top_k=top_k)
    # state: (makespan, assigned_set, assignments_list, tour_after)
    init_sim = simulate_schedule(coords, base_tour, [], v_truck, v_drone, endurance, recharge)
    best = (init_sim['makespan'], frozenset(), [], base_tour)
    beam = [best]
    for t, cand_list in enumerate(candidates):
        next_beam = []
        for mk, assigned, asg, tour in beam:
            # option 1: skip this edge
            next_beam.append((mk, assigned, asg, tour))
            # option 2: assign one candidate (u,j,v) if j not already assigned and u,v in tour
            for _, (u,j,v),_,_ in cand_list:
                if j in assigned: 
                    continue
                tour2 = remove_customers_from_tour(tour, {j})
                if (u not in tour2) or (v not in tour2):
                    continue
                sim = simulate_schedule(coords, tour2, asg + [(u,j,v)], v_truck, v_drone, endurance, recharge)
                if not sim.get('feasible', False):
                    continue
                next_beam.append((sim['makespan'], assigned|{j}, asg+[(u,j,v)], tour2))
        # prune
        next_beam.sort(key=lambda x: x[0])
        beam = next_beam[:beam_size]
    best_state = min(beam, key=lambda x: x[0])
    mk, assigned, asg, tour = best_state
    sim = simulate_schedule(coords, tour, asg, v_truck, v_drone, endurance, recharge)
    return {
        'tour_after': tour,
        'assignments': asg,
        'makespan': mk,
        'truck_time': sim.get('truck_time', float('nan')),
        'total_wait': sim.get('total_wait', float('nan')),
        'events': sim.get('events', [])
    }

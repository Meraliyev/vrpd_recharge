import numpy as np

def dist(a,b):
    return float(np.linalg.norm(a-b))

def truck_time_for_tour(coords, tour, v_truck):
    return sum(dist(coords[tour[i]], coords[tour[i+1]])/v_truck for i in range(len(tour)-1))

# --- in drone_scheduler.py ---

def simulate_schedule(coords, tour, assignments,
                      v_truck=1.0, v_drone=2.0,
                      endurance=0.7, recharge=0.1):
    """
    Simulate one truck + one drone timeline.

    tour: list with depot at both ends (0 ... 0)
    assignments: list of (u, j, v) tuples:
        launch at truck node u, drone serves customer j, rendezvous at truck node v.

    Returns dict:
      {feasible, makespan, truck_time, total_wait, events}
      events: list of (u, j, v, launch_time, rendez_time, wait_added_at_v)
    """
    assert tour[0] == 0 and tour[-1] == 0
    node_pos = {node: i for i, node in enumerate(tour)}

    # ---- Robust validation of assignments against the CURRENT truck tour ----
    # 1) Launch (u) and rendezvous (v) must still be on the truck tour.
    # 2) The drone customer j must NOT be on the truck tour anymore
    #    (since j should have been removed from the truck route).
    for (u, j, v) in assignments:
        if (u not in node_pos) or (v not in node_pos):
            return {'feasible': False}
        if j in node_pos:
            return {'feasible': False}
    # ------------------------------------------------------------------------

    # Precompute edge times and cumulative arrival times (no waiting yet).
    def _dist(a, b):
        import numpy as np
        return float(np.linalg.norm(coords[a] - coords[b]))

    edge_times = [_dist(tour[i], tour[i+1]) / v_truck for i in range(len(tour)-1)]
    arr = [0.0]
    for et in edge_times:
        arr.append(arr[-1] + et)

    # Sort assignments by launch position along the truck tour.
    asg = sorted(assignments, key=lambda x: node_pos[x[0]])

    t_shift = 0.0          # accumulated truck waiting due to long sorties
    drone_ready_time = 0.0 # after landing, drone needs recharge before next launch
    events = []

    for (u, j, v) in asg:
        iu, iv = node_pos[u], node_pos[v]
        if iv <= iu:
            return {'feasible': False}

        # Earliest feasible launch: truck arrives at u (with current shift) AND drone ready.
        launch_time = max(arr[iu] + t_shift, drone_ready_time)

        # Drone sortie time (u->j->v); enforce endurance.
        dtime = (_dist(u, j) + _dist(j, v)) / v_drone
        if dtime > endurance + 1e-12:
            return {'feasible': False}

        rendez_time = launch_time + dtime
        truck_at_v = arr[iv] + t_shift

        # Truck waits if the drone arrives later than the truck at v.
        wait = max(0.0, rendez_time - truck_at_v)
        if wait > 0:
            t_shift += wait
            truck_at_v += wait

        # Drone recharge window (can overlap with truck movement).
        drone_ready_time = rendez_time + recharge

        events.append((u, j, v, launch_time, rendez_time, wait))

    makespan = arr[-1] + t_shift
    return {
        'feasible': True,
        'makespan': makespan,
        'truck_time': arr[-1],
        'total_wait': t_shift,
        'events': events
    }


def remove_customers_from_tour(tour, remove_set):
    return [n for n in tour if n==0 or n not in remove_set]

def greedy_drone_assignment(coords, base_tour, v_truck=1.0, v_drone=2.0, endurance=0.7, recharge=0.1, window=8):
    """Greedy hill-climbing over assignments:
    Iteratively try assigning a (u, j, v) that most reduces makespan.
    * window: how far forward v can be from u (limits search, speeds up).
    Returns details: {tour_after, assignments, makespan, truck_time, total_wait, events}
    """
    assert base_tour[0]==0 and base_tour[-1]==0
    cur_asg = []
    cur_remove = set()
    cur_tour = base_tour[:]
    best_sim = simulate_schedule(coords, cur_tour, cur_asg, v_truck, v_drone, endurance, recharge)
    if not best_sim.get('feasible', False):
        raise RuntimeError('Initial schedule infeasible — check inputs')
    improved=True
    while improved:
        improved=False
        candidate_best = None
        # --- NEW: lock truck nodes used as launch/rendezvous so we don't remove them later ---
        locked = set()
        for (uu, jj, vv) in cur_asg:
            locked.add(uu)
            locked.add(vv)
        # --------------------------------------------------------------------------------------
        customers = [n for n in cur_tour if n != 0 and n not in cur_remove and n not in locked]
        for j in customers:
            idx = cur_tour.index(j)
            iu = idx-1  # launch at predecessor on truck tour
            for offset in range(1, window+1):
                ik = min(idx-1+offset, len(cur_tour)-2)
                u = cur_tour[iu]; v = cur_tour[ik+1]
                new_remove = set(cur_remove); new_remove.add(j)
                new_tour = remove_customers_from_tour(base_tour, new_remove)
                if u not in new_tour or v not in new_tour:
                    continue
                sim = simulate_schedule(coords, new_tour, cur_asg + [(u,j,v)], v_truck, v_drone, endurance, recharge)
                if not sim.get('feasible', False):
                    continue
                delta = sim['makespan'] - best_sim['makespan']
                if candidate_best is None or delta < candidate_best[0]:
                    candidate_best = (delta, (u,j,v), new_remove, new_tour, sim)
        if candidate_best is not None and candidate_best[0] < -1e-9:
            improved=True
            delta,(u,j,v),cur_remove,cur_tour,best_sim = candidate_best
            cur_asg.append((u,j,v))
        else:
            break
    return {
        'tour_after': cur_tour,
        'assignments': cur_asg,
        'makespan': best_sim['makespan'],
        'truck_time': best_sim['truck_time'],
        'total_wait': best_sim['total_wait'],
        'events': best_sim.get('events', [])
    }

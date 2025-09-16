import numpy as np, random, math
from truck_heuristics import total_distance

def dist_pt(coords, i, j):
    return float(np.linalg.norm(coords[i]-coords[j]))

def split_routes_by_zeros(tour):
    routes=[]; cur=[]
    for n in tour:
        cur.append(n)
        if n==0 and len(cur)>1:
            if cur[0]!=0: cur=[0]+cur
            if cur[-1]!=0: cur=cur+[0]
            routes.append(cur); cur=[]
    if cur: 
        if cur[0]!=0: cur=[0]+cur
        if cur[-1]!=0: cur=cur+[0]
        routes.append(cur)
    # merge adjacent to avoid duplicates
    merged=[]
    for r in routes:
        if not merged: merged.append(r)
        else:
            if merged[-1][-1]==0 and r[0]==0:
                merged[-1] = merged[-1] + r[1:]
            else:
                merged.append(r)
    return merged

def route_load(route, demands):
    return int(sum(demands[n] for n in route if n!=0))

def to_tour(routes):
    tour=[]
    for r in routes:
        if tour and tour[-1]!=0: tour.append(0)
        if r[0]!=0: r=[0]+r
        if r[-1]!=0: r=r+[0]
        tour.extend(r)
    if not tour: return [0,0]
    if tour[0]!=0: tour=[0]+tour
    if tour[-1]!=0: tour=tour+[0]
    return tour

def greedy_insert(routes, demands, capacity, coords, removed):
    """Insert removed customers one by one greedily by minimum additional distance respecting capacity."""
    for j in removed:
        best=None
        # try all routes and all positions
        for rid, r in enumerate(routes):
            load = route_load(r, demands)
            if load + demands[j] > capacity: 
                continue
            # try inserting at each position between nodes
            for pos in range(1, len(r)):
                a, b = r[pos-1], r[pos]
                delta = (dist_pt(coords, a, j) + dist_pt(coords, j, b) - dist_pt(coords, a, b))
                if best is None or delta < best[0]:
                    best = (delta, rid, pos)
        # if no feasible route, start a new one [0,j,0]
        if best is None:
            routes.append([0, j, 0])
        else:
            _, rid, pos = best
            routes[rid] = routes[rid][:pos] + [j] + routes[rid][pos:]
    return routes

def shaw_relatedness(coords, demands, a, b, w_d=1.0, w_q=0.1):
    return w_d * float(np.linalg.norm(coords[a]-coords[b])) + w_q * abs(int(demands[a])-int(demands[b]))

def random_removal(coords, demands, tour, q):
    customers = [n for n in tour if n!=0]
    removed = set(random.sample(customers, min(q, len(customers))))
    remain = [n for n in tour if (n==0 or n not in removed)]
    return remain, list(removed)

def shaw_removal(coords, demands, tour, q):
    customers = [n for n in tour if n!=0]
    if not customers:
        return tour, []
    seed = random.choice(customers)
    removed = {seed}
    while len(removed) < min(q, len(customers)):
        # choose next most related to any removed
        cand = None; best = 1e18
        for c in customers:
            if c in removed: continue
            rel = min(shaw_relatedness(coords, demands, c, r) for r in removed)
            if rel < best:
                best = rel; cand = c
        removed.add(cand)
    remain = [n for n in tour if (n==0 or n not in removed)]
    return remain, list(removed)

def worst_removal(coords, demands, tour, q):
    # remove customers with largest saving if removed (myopic)
    gains=[]
    for i in range(1, len(tour)-1):
        j=tour[i]
        if j==0: continue
        a,b=tour[i-1], tour[i+1]
        saving = (dist_pt(coords,a,j)+dist_pt(coords,j,b)-dist_pt(coords,a,b))
        gains.append((saving, i, j))
    gains.sort(reverse=True)
    to_remove = [j for _,_,j in gains[:q]]
    remain = [n for n in tour if (n==0 or n not in to_remove)]
    return remain, to_remove

def alns_search(coords, demands, capacity, init_tour, iters=2000, q_frac=0.2, temp=1.0, cooling=0.9995, reaction=0.1):
    """Adaptive LNS over truck tour (CVRP)."""
    rng = random.Random(1234)
    n_customers = sum(1 for x in init_tour if x!=0)
    q = max(1, int(q_frac * n_customers))
    destroy_ops = [random_removal, shaw_removal, worst_removal]
    repair_ops = [greedy_insert]
    w_destroy = [1.0]*len(destroy_ops)
    w_repair = [1.0]*len(repair_ops)
    s_best = init_tour[:]
    f_best = total_distance(coords, s_best)
    s = s_best[:]; f = f_best
    T = temp
    for it in range(iters):
        # pick ops
        d_idx = random.choices(range(len(destroy_ops)), weights=w_destroy, k=1)[0]
        r_idx = random.choices(range(len(repair_ops)), weights=w_repair, k=1)[0]
        destroy = destroy_ops[d_idx]; repair = repair_ops[r_idx]
        # apply destroy
        s_remain, removed = destroy(coords, demands, s, q)
        # repair
        routes = split_routes_by_zeros(s_remain)
        routes = repair(routes, demands, capacity, coords, removed)
        cand = to_tour(routes)
        f_cand = total_distance(coords, cand)
        # accept?
        delta = f_cand - f
        accept = (delta < 0) or (math.exp(-delta/max(1e-9,T)) > rng.random())
        if accept:
            s, f = cand, f_cand
            # update operator weights (simple)
            gain = max(0.0, (f_best - f_cand))
            w_destroy[d_idx] = (1-reaction)*w_destroy[d_idx] + reaction*(1 + 5*(gain>0) + 10*(f_cand < f_best))
            w_repair[r_idx]  = (1-reaction)*w_repair[r_idx]  + reaction*(1 + 5*(gain>0) + 10*(f_cand < f_best))
            if f_cand + 1e-9 < f_best:
                s_best, f_best = cand, f_cand
        T *= cooling
    return s_best

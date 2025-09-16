import numpy as np

def dist(a,b):
    return float(np.linalg.norm(a-b))

def total_distance(coords, tour):
    return sum(dist(coords[tour[i]], coords[tour[(i+1)%len(tour)]]) for i in range(len(tour)))

def clarke_wright(coords, demands, capacity):
    N = len(coords)-1
    d0 = np.array([dist(coords[0], coords[i]) for i in range(N+1)], dtype=float)
    savings = []
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            s = d0[i] + d0[j] - dist(coords[i], coords[j])
            savings.append((s,i,j))
    savings.sort(reverse=True)
    routes = {i: [0,i,0] for i in range(1,N+1)}
    loads = {i: int(demands[i]) for i in range(1,N+1)}
    def find(node):
        for k,r in routes.items():
            if node in r[1:-1]: return k
        return None
    for s,i,j in savings:
        ri, rj = find(i), find(j)
        if ri is None or rj is None or ri==rj: 
            continue
        if loads[ri] + loads[rj] > capacity:
            continue
        a = routes[ri]; b = routes[rj]
        merged = None
        if a[-2]==i and b[1]==j: merged = a[:-1] + b[1:]
        elif b[-2]==j and a[1]==i: merged = b[:-1] + a[1:]
        if merged is not None:
            routes[ri] = merged
            loads[ri] += loads[rj]
            del routes[rj]; del loads[rj]
    tour = []
    for _,r in routes.items():
        if tour and tour[-1]!=0: tour.append(0)
        tour.extend(r)
    return tour

def two_opt(coords, tour, max_iter=2000):
    best = tour[:]
    n = len(best)
    def length(t):
        return total_distance(coords, t)
    best_len = length(best)
    improved = True; it=0
    while improved and it < max_iter:
        improved=False; it+=1
        for i in range(1, n-2):
            for k in range(i+1, n-1):
                if k-i==1: continue
                new = best[:i] + best[i:k+1][::-1] + best[k+1:]
                new_len = length(new)
                if new_len + 1e-9 < best_len:
                    best, best_len = new, new_len
                    improved=True
    return best

def or_opt(coords, tour, max_block=3, max_iter=2000):
    t = tour[:]
    n = len(t)
    def L(tt): return total_distance(coords, tt)
    bestL = L(t)
    it=0; improved=True
    while improved and it < max_iter:
        improved=False; it+=1
        for block in range(1, max_block+1):
            for i in range(1, n-1-block):
                seg = t[i:i+block]
                if 0 in seg: continue
                remain = t[:i] + t[i+block:]
                for j in range(1, len(remain)):
                    if j==i or j==i+1: continue
                    cand = remain[:j] + seg + remain[j:]
                    if cand[0]!=0 or cand[-1]!=0: continue
                    newL = L(cand)
                    if newL + 1e-9 < bestL:
                        t, bestL = cand, newL
                        n = len(t)
                        improved=True
                        break
                if improved: break
            if improved: break
    return t

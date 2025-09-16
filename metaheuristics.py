# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 14:36:00 2025

@author: sduai
"""

# metaheuristics.py
# Truck tour metaheuristics (single-truck CVRP-style tour with depot=0, zeros separating routes)
# Reuses your total_distance and basic neighborhoods for compactness.

import numpy as np, random, math
from truck_heuristics import total_distance

def _dist(coords, a, b):
    return float(np.linalg.norm(coords[a] - coords[b]))

def _two_opt_move(tour, i, k):
    if k - i <= 1:
        return tour
    return tour[:i] + tour[i:k+1][::-1] + tour[k+1:]

def _relocate_move(tour, i, j):
    # move node at position i to before position j
    if i == j or i+1 == j:
        return tour
    node = tour[i]
    if node == 0:
        return tour
    t = tour[:i] + tour[i+1:]
    j2 = j if j < i else j-1
    return t[:j2] + [node] + t[j2:]

def _random_customer_positions(tour):
    idxs = [i for i,x in enumerate(tour) if x != 0]
    if len(idxs) < 2:
        return None
    i, j = random.sample(idxs, 2)
    i, j = min(i,j), max(i,j)
    return i, j

def _length(coords, tour):
    return total_distance(coords, tour)

# ---------------- Simulated Annealing ----------------
def sa_search(coords, tour, iters=4000, T0=1.0, cooling=0.9995, seed=1):
    random.seed(seed)
    best = tour[:]; bestL = _length(coords, best)
    cur = best[:];  curL = bestL
    T = T0
    for _ in range(iters):
        if random.random() < 0.5:
            pair = _random_customer_positions(cur)
            if not pair: continue
            i, k = pair
            cand = _two_opt_move(cur, i, k)
        else:
            pair = _random_customer_positions(cur)
            if not pair: continue
            i, j = pair
            cand = _relocate_move(cur, i, j)
        L = _length(coords, cand)
        d = L - curL
        if d < 0 or math.exp(-d / max(T, 1e-9)) > random.random():
            cur, curL = cand, L
            if L + 1e-9 < bestL:
                best, bestL = cand, L
        T *= cooling
    return best

# ---------------- Tabu Search ----------------
def tabu_search(coords, tour, iters=2000, tabu_tenure=25, seed=1):
    random.seed(seed)
    cur = tour[:]; curL = _length(coords, cur)
    best = cur[:]; bestL = curL
    tabu = {}
    for it in range(iters):
        best_move = None
        best_delta = float('inf')
        for _ in range(120):  # sample neighborhood
            pair = _random_customer_positions(cur)
            if not pair: continue
            if random.random() < 0.5:
                i, k = pair
                cand = _two_opt_move(cur, i, k)
                sig = ('2opt', cur[i], cur[k])
            else:
                i, j = pair
                cand = _relocate_move(cur, i, j)
                sig = ('rel', cur[i], cur[j-1] if j>0 else 0)
            L = _length(coords, cand)
            delta = L - curL
            tabu_ok = (sig not in tabu) or (it - tabu[sig] > tabu_tenure)
            if delta < best_delta and (tabu_ok or L + 1e-9 < bestL):  # aspiration
                best_move = (cand, sig, L)
                best_delta = delta
        if best_move is None:
            # random kick
            pair = _random_customer_positions(cur)
            if not pair: break
            cur = _two_opt_move(cur, *pair)
            curL = _length(coords, cur)
            continue
        cand, sig, L = best_move
        cur, curL = cand, L
        tabu[sig] = it
        if L + 1e-9 < bestL:
            best, bestL = cand, L
    return best

# ---------------- Genetic Algorithm ----------------
def _ordered_crossover(p1, p2):
    a, b = sorted(random.sample(range(len(p1)), 2))
    child = [None]*len(p1)
    child[a:b+1] = p1[a:b+1]
    fill = [x for x in p2 if x not in child]
    ptr = 0
    for i in range(len(p1)):
        if child[i] is None:
            child[i] = fill[ptr]; ptr += 1
    return child

def ga_search(coords, tour, pop_size=30, gens=200, mut_rate=0.2, seed=1):
    random.seed(seed)
    base = [x for x in tour if x!=0]
    n = len(base)
    def rand_tour():
        p = base[:]; random.shuffle(p)
        return [0]+p+[0]
    pop = [tour[:]] + [rand_tour() for _ in range(pop_size-1)]
    def fit(t): return 1.0/(_length(coords, t)+1e-9)
    for _ in range(gens):
        # selection (tournament) + crossover
        new_pop = []
        for _ in range(pop_size//2):
            a,b = random.sample(range(len(pop)), 2)
            c,d = random.sample(range(len(pop)), 2)
            pa = pop[a] if fit(pop[a])>fit(pop[b]) else pop[b]
            pb = pop[c] if fit(pop[c])>fit(pop[d]) else pop[d]
            # flatten no-zeros sequences
            sa = [x for x in pa if x!=0]
            sb = [x for x in pb if x!=0]
            child_seq = _ordered_crossover(sa, sb)
            child = [0] + child_seq + [0]
            # mutation (swap)
            if random.random() < mut_rate and n>=2:
                i, j = sorted(random.sample(range(1, n+1), 2))
                child[i], child[j] = child[j], child[i]
            new_pop.extend([pa, child])
        pop = new_pop[:pop_size]
        # brief local clean-up
        for i in range(len(pop)):
            t = pop[i]
            for _ in range(4):
                pair = _random_customer_positions(t)
                if not pair: break
                if random.random()<0.5:
                    t = _two_opt_move(t, *pair)
                else:
                    t = _relocate_move(t, *pair)
            pop[i] = t
    best = max(pop, key=fit)
    return best

# ---------------- Variable Neighborhood Search ----------------
def vns_search(coords, tour, k_max=3, iters=800, seed=1):
    random.seed(seed)
    best = tour[:]; bestL = _length(coords, best)
    def _shake(t, k):
        u = t[:]
        for _ in range(k):
            pair = _random_customer_positions(u)
            if not pair: break
            if random.random()<0.5:
                u = _two_opt_move(u, *pair)
            else:
                u = _relocate_move(u, *pair)
        return u
    for _ in range(iters):
        k = 1
        cur = best[:]
        while k <= k_max:
            shaken = _shake(cur, k)
            improved = False
            for _ in range(60):
                pair = _random_customer_positions(shaken)
                if not pair: break
                cand = _two_opt_move(shaken, *pair) if random.random()<0.5 else _relocate_move(shaken, *pair)
                if _length(coords, cand) + 1e-9 < _length(coords, shaken):
                    shaken = cand; improved = True
            curL = _length(coords, shaken)
            if curL + 1e-9 < bestL:
                best, bestL = shaken, curL
                k = 1
            else:
                k += 1
    return best

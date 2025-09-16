from drone_scheduler import simulate_schedule, remove_customers_from_tour

def assignment_vns(coords, base_tour, init_assignments, v_truck=1.0, v_drone=2.0, endurance=0.7, recharge=0.1, max_iter=200):
    """Refine a given set of assignments by local moves:
    - relocate: move one customer's rendezvous (same launch)
    - reassign: choose a different launch edge
    - drop/add: drop one and add another
    This is a simple VNS with first-improvement.
    """
    best_asg = init_assignments[:]
    best_tour = remove_customers_from_tour(base_tour, {j for _,j,_ in best_asg})
    best_sim = simulate_schedule(coords, best_tour, best_asg, v_truck, v_drone, endurance, recharge)
    if not best_sim.get('feasible', False):
        raise RuntimeError('Initial assignments infeasible')
    for it in range(max_iter):
        improved=False
        # try relocate rendezvous for each assg
        for idx,(u,j,v) in enumerate(list(best_asg)):
            # try rendezvous at the next 2 successors along tour
            # we assume v must be in tour; pick alternatives by scanning tour
            tour_nodes = [n for n in best_tour]
            if v not in tour_nodes or u not in tour_nodes: 
                continue
            vidx = tour_nodes.index(v); uidx = tour_nodes.index(u)
            for vidx2 in [vidx+1, min(vidx+2, len(tour_nodes)-1)]:
                vv = tour_nodes[vidx2]
                if vv==0 and vidx2==len(tour_nodes)-1: pass
                cand_asg = best_asg[:]; cand_asg[idx]=(u,j,vv)
                cand_tour = remove_customers_from_tour(base_tour, {jj for _,jj,_ in cand_asg})
                sim = simulate_schedule(coords, cand_tour, cand_asg, v_truck, v_drone, endurance, recharge)
                if sim.get('feasible', False) and sim['makespan'] + 1e-9 < best_sim['makespan']:
                    best_asg, best_tour, best_sim = cand_asg, cand_tour, sim
                    improved=True
                    break
            if improved: break
        if improved: continue
        # try drop/add: drop worst single assignment and try to add a new one greedily
        worst_idx = None; worst_mk = best_sim['makespan']
        for idx in range(len(best_asg)):
            tmp = best_asg[:idx]+best_asg[idx+1:]
            tmp_tour = remove_customers_from_tour(base_tour, {jj for _,jj,_ in tmp})
            sim = simulate_schedule(coords, tmp_tour, tmp, v_truck, v_drone, endurance, recharge)
            if sim.get('feasible', False) and sim['makespan'] < worst_mk:
                worst_mk = sim['makespan']; worst_idx = idx
        if worst_idx is not None:
            best_asg = best_asg[:worst_idx] + best_asg[worst_idx+1:]
            best_tour = remove_customers_from_tour(base_tour, {jj for _,jj,_ in best_asg})
            best_sim = simulate_schedule(coords, best_tour, best_asg, v_truck, v_drone, endurance, recharge)
            improved=True
        if not improved:
            break
    return {
        'assignments': best_asg,
        'tour_after': best_tour,
        'makespan': best_sim['makespan']
    }

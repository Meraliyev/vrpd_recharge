import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from truck_heuristics import total_distance, two_opt, or_opt
from drone_scheduler import greedy_drone_assignment

def to_tensor(x): 
    return torch.tensor(x, dtype=torch.float32)

class Encoder(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.inp = nn.Linear(3, d_model)  # x, y, demand/scale
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
    def forward(self, feats):
        h = self.inp(feats)
        return h + self.ff(h)

class AttentionDecoder(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.ctx = nn.Parameter(torch.randn(d_model))
    def forward(self, H, mask, last_emb):
        q = self.q(last_emb + self.ctx)
        att = torch.matmul(self.k(H), q) / (H.size(-1)**0.5)
        att = att.masked_fill(~mask, -1e9)
        return F.log_softmax(att, dim=0)

class Critic(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
    def forward(self, H):
        return self.net(H.mean(dim=0))

class PointerPolicy(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.enc = Encoder(d_model)
        self.dec = AttentionDecoder(d_model)
        self.crit = Critic(d_model)
    def forward(self, feats, mask, last_idx):
        H = self.enc(feats)
        last = H[last_idx]
        return H, self.dec(H, mask, last)
    def value(self, feats):
        H = self.enc(feats)
        return self.crit(H).squeeze(-1)

def build_tour(policy, coords, demands, capacity, device='cpu'):
    n = coords.shape[0]-1
    feats = torch.tensor(np.concatenate([coords, (demands/max(1,demands.max())).reshape(-1,1)], axis=1), dtype=torch.float32, device=device)
    mask = torch.zeros(n+1, dtype=torch.bool, device=device); mask[:] = True; mask[0]=False
    tour=[0]; load=0; last=0; logps=[]
    while mask[1:].any():
        capmask = mask.clone()
        for j in range(1, n+1):
            if capmask[j] and load + demands[j] > capacity:
                capmask[j]=False
        if not capmask[1:].any():
            tour.append(0); load=0; last=0
            continue
        H, logp = policy(feats, capmask, last)
        j = torch.distributions.Categorical(logits=logp).sample().item()
        logps.append(logp[j])
        tour.append(j)
        load += demands[j]; mask[j]=False; last=j
    if tour[-1]!=0: tour.append(0)
    return tour, torch.stack(logps).sum()

def postprocess_truck(coords, tour):
    t = two_opt(coords, tour, max_iter=2000)
    t = or_opt(coords, t, max_block=3, max_iter=2000)
    return t

def evaluate_instance(coords, demands, capacity, tour, v_truck, endurance, recharge, window=8):
    t = postprocess_truck(coords, tour)
    res = greedy_drone_assignment(coords, t, v_truck=v_truck, v_drone=2.0*v_truck, endurance=endurance, recharge=recharge, window=window)
    return res['makespan']

def train_pointer(instances, epochs=2000, lr=1e-4, capacity=50, v_truck=1.0, endurance=0.7, recharge=0.1, device='cpu', seed=1, log_every=10):
    torch.manual_seed(seed); np.random.seed(seed)
    policy = PointerPolicy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    history=[]
    for ep in range(1, epochs+1):
        losses=[]; rewards=[]
        for inst in instances:
            coords = inst['coords']; demands = inst['demands'].astype(np.int64)
            tour, logp = build_tour(policy, coords, demands, capacity, device)
            mk = evaluate_instance(coords, demands, capacity, tour, v_truck, endurance, recharge)
            R = -mk
            b = policy.value(torch.tensor(np.concatenate([coords, (demands/max(1,demands.max())).reshape(-1,1)], axis=1), dtype=torch.float32, device=device))
            loss = -(R - b).detach() * logp + 0.5*(R - b)**2  # actor + critic
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
            losses.append(loss.item()); rewards.append(R)
        if ep % max(1, log_every) == 0:
            avgR = float(np.mean(rewards))
            history.append({'epoch': ep, 'avg_reward': avgR, 'avg_makespan': -avgR})
            print(f"[RL] {ep}/{epochs} avgR={avgR:.3f}")
    return policy, history

def best_of_k(policy, inst, capacity, v_truck, endurance, recharge, K=32, device='cpu', seed=1):
    torch.manual_seed(seed); np.random.seed(seed)
    coords = inst['coords']; demands = inst['demands'].astype(np.int64)
    best=None
    for k in range(max(1,K)):
        tour,_ = build_tour(policy, coords, demands, capacity, device)
        mk = evaluate_instance(coords, demands, capacity, tour, v_truck, endurance, recharge)
        if best is None or mk < best[0]:
            best = (mk, tour)
    return best

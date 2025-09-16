import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from truck_heuristics import two_opt, or_opt
from drone_scheduler_beam import beam_search_assign

class MHAEncoder(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.proj = nn.Linear(3, d_model)
    def forward(self, feats):
        H = self.proj(feats)  # [B, N+1, D]
        return self.enc(H)

class PointerDecoder(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model))
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
    def forward(self, H, mask, last_emb):
        q = self.Wq(last_emb + self.query)  # [B,D]
        logits = torch.einsum('bnd,bd->bn', self.Wk(H), q) / (H.size(-1)**0.5)
        logits = logits.masked_fill(~mask, -1e9)
        return F.log_softmax(logits, dim=-1)

class TransformerPolicy(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.enc = MHAEncoder(d_model, n_heads, n_layers)
        self.dec = PointerDecoder(d_model)
        self.v = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model,1))
    def forward(self, feats, mask, last_idx):
        H = self.enc(feats)  # [B, N+1, D]
        B = H.size(0)
        last = torch.stack([H[b, last_idx[b]] for b in range(B)], dim=0)
        return H, self.dec(H, mask, last)
    def value(self, feats):
        H = self.enc(feats)
        return self.v(H.mean(dim=1)).squeeze(-1)

def sample_tour(policy, coords, demands, capacity, device='cpu', greedy=False):
    B = coords.shape[0]
    Np1 = coords.shape[1]
    feats = torch.tensor(np.concatenate([coords, (demands/np.maximum(1, demands.max(axis=1, keepdims=True))).reshape(B,Np1,1)], axis=2), dtype=torch.float32, device=device)
    mask = torch.ones(B, Np1, dtype=torch.bool, device=device)
    mask[:,0] = False
    tours=[ [0] for _ in range(B) ]
    loads = np.zeros(B, dtype=np.int64)
    last_idx = torch.zeros(B, dtype=torch.long, device=device)
    logps = torch.zeros(B, device=device)
    remaining = mask[:,1:].any(dim=1)
    while remaining.any():
        capmask = mask.clone()
        for b in range(B):
            if not remaining[b]: continue
            for j in range(1, Np1):
                if capmask[b,j] and (loads[b] + demands[b,j] > capacity[b]):
                    capmask[b,j]=False
        none_feasible = (~capmask[:,1:]).all(dim=1)
        # return-to-depot where needed
        for b in range(B):
            if remaining[b] and none_feasible[b]:
                tours[b].append(0); loads[b]=0; last_idx[b]=0
        H, logp = policy(feats, capmask, last_idx)
        if greedy:
            j = torch.argmax(logp, dim=1)
        else:
            j = torch.distributions.Categorical(logits=logp).sample()
        for b in range(B):
            if not remaining[b]: continue
            jb = int(j[b].item())
            logps[b] += logp[b, jb]
            tours[b].append(jb)
            if jb>0:
                loads[b] += int(demands[b, jb])
                mask[b, jb] = False
                last_idx[b] = jb
            else:
                loads[b]=0; last_idx[b]=0
        remaining = mask[:,1:].any(dim=1)
    for b in range(B):
        if tours[b][-1]!=0: tours[b].append(0)
    return tours, logps

def postprocess_truck(coords, tours):
    out=[]
    for b,t in enumerate(tours):
        tt = two_opt(coords[b], t, max_iter=2000)
        tt = or_opt(coords[b], tt, max_block=3, max_iter=2000)
        out.append(tt)
    return out

def evaluate(coords, tours, v_truck, endurance, recharge, beam_size=8):
    B = len(tours)
    mks = []
    for b in range(B):
        res = beam_search_assign(coords[b], tours[b], v_truck=v_truck[b], v_drone=2.0*v_truck[b], endurance=endurance[b], recharge=recharge[b], beam_size=beam_size, top_k=5)
        mks.append(res['makespan'])
    return np.array(mks, dtype=np.float32)

def train_scst(num_epochs=1000, batch_size=16, N=50, capacity=50, v_truck=1.0, endurance=0.7, recharge=0.1, device='cpu', seed=1, beam_size=8):
    torch.manual_seed(seed); np.random.seed(seed)
    policy = TransformerPolicy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-4)
    hist=[]
    for ep in range(1, num_epochs+1):
        # sample batch of instances
        coords = np.random.rand(batch_size, N+1, 2).astype(np.float32)
        demands = np.random.randint(1, 11, size=(batch_size, N+1)).astype(np.int64); demands[:,0]=0
        cap = np.array([capacity]*batch_size, dtype=np.int64)
        v = np.array([v_truck]*batch_size, dtype=np.float32)
        E = np.array([endurance]*batch_size, dtype=np.float32)
        R = np.array([recharge]*batch_size, dtype=np.float32)
        # sample and greedy baseline
        tours_s, logps = sample_tour(policy, coords, demands, cap, device=device, greedy=False)
        tours_g, _ = sample_tour(policy, coords, demands, cap, device=device, greedy=True)
        tours_s = postprocess_truck(coords, tours_s)
        tours_g = postprocess_truck(coords, tours_g)
        mk_s = evaluate(coords, tours_s, v, E, R, beam_size=beam_size)
        mk_g = evaluate(coords, tours_g, v, E, R, beam_size=beam_size)
        R_s = -torch.tensor(mk_s, dtype=torch.float32, device=device)
        R_g = -torch.tensor(mk_g, dtype=torch.float32, device=device)
        adv = (R_s - R_g).detach()
        loss = -(adv * logps).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
        if ep % max(1, num_epochs//20) == 0:
            avg_mk = float(mk_s.mean())
            hist.append({'epoch': ep, 'avg_makespan': avg_mk})
            print(f"[SCST] {ep}/{num_epochs} avg_makespan={avg_mk:.3f}")
    return policy, hist

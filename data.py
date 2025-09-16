import numpy as np

def generate_instance(N=50, seed=1, demand_low=1, demand_high=10, capacity=50):
    rng = np.random.default_rng(seed)
    coords = rng.random((N+1, 2)).astype(np.float32)  # 0 is depot
    demands = rng.integers(demand_low, demand_high+1, size=N+1).astype(np.int32)
    demands[0] = 0
    return {
        'N': int(N),
        'coords': coords,
        'demands': demands,
        'capacity': int(capacity)
    }

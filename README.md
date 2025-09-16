# A Simple and Reproducible Hybrid Solver for a Truck–Drone VRP with Recharge

**Authors:** Meraryslan Meraliyev, Cemil Turan, and Shirali Kadyrov  
**Setting:** One truck carries one drone. The drone flies **2× faster** than the truck. Each drone sortie must fit an **endurance** budget, and **after each delivery the drone must recharge on the truck** before it can launch again.

This repository provides a **hybrid RL + heuristics** pipeline: an ALNS-based truck tour with local search, followed by a **learned, feasibility‑aware scheduler** for drone sorties. A **timeline simulator** enforces launch/recovery handling, endurance, and recharge, so the score is the **true makespan**.

---

## ✨ Highlights

- **Hybrid RL Solver.** ALNS (+ 2/3‑opt, Or‑opt) for the truck; a **policy‑gradient pointer/attention** scheduler for drone sorties with **hard feasibility masks**.
- **Masked decoding + exact simulation.** Greedy or masked beam decoding is coupled to a **linear‑time timeline simulator** that returns the true makespan.
- **Config‑first, Spyder‑friendly.** One `main.py` entrypoint; JSON configs; batch scripts; plotting and LaTeX generation for fast paper‑ready artifacts.
- **Reproducible evidence.** On \(N{=}50\), \(E{=}0.7\), \(R{=}0.1\): Proposed **5.203 ± 0.093**, NN **5.208 ± 0.124**, ALNS **5.349 ± 0.038** (Proposed is **2.73%** better than ALNS on average; ~**0.10%** edge vs NN).

---

## 📦 Requirements

- Python **3.9–3.11**
- Packages: `numpy`, `pandas`, `scipy`, `matplotlib`, `networkx`, `tqdm`, `PyYAML`, `torch>=2.0`

Install:
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```


> The code expects `runs/<tag>_seed<SEED>_N<N>/results.csv` and `solutions.json` per run.  

---

## ⚙️ Configuration

Example `configs/config.json`:
```json
{
  "seeds": [1, 2, 3],
  "N": 50,
  "capacity": 50,
  "v_truck": 1.0,
  "speed_ratio": 2.0,        // v_drone = 2 * v_truck
  "endurance": 0.7,
  "recharge": 0.1,
  "tour_init": "ALNS",       // or: "NN", "CW"
  "local_search": ["2opt", "oropt"],
  "scheduler": "beam",       // or: "greedy"
  "window": 10,              // #edges to consider around current position
  "rl": {
    "enable": true,
    "model": "pointer",      // or: "transformer"
    "epochs": 2000,
    "best_of_K": 32,
    "beam_width": 8
  },
  "save_json": true,
  "eval_only": false
}
```

---

## 🚀 Running Experiments

**Single run (from Spyder or terminal):**
```bash
python main.py --config configs/config.json
```

**Batch across seeds (example):**
```bash
python main.py --config configs/config.json --seeds 1 2 3
```

**Aggregate results (mean ± s.e.), significance tests, and plot:**
```bash
python scripts/aggregate_multi_seed.py --glob "runs/*/results.csv" --outdir paper --group_by_N
# Outputs: paper/table_results_all.tex, paper/pvalues.tex, paper/agg_bars_all.png
```

**Quick bar chart only:**
```bash
python scripts/plot_results.py "runs/*/results.csv" --out paper/agg_bars_all.png
```

---

## 📊 Results Snapshot

**Aggregate (N=50, E=0.7, R=0.1, 3 seeds):**

| Method                         | Mean makespan ↓ | s.e.  |
|-------------------------------|----------------:|:-----:|
| **PointerRL(K=32)+LS+Drone**  | **5.203**       | 0.093 |
| NN+LS+Drone                   | 5.208           | 0.124 |
| ALNS+LS+Drone                 | 5.349           | 0.038 |

- **Proposed vs. ALNS**: **−0.146** absolute (≈ **2.73%** better).
- **Proposed vs. NN**: **−0.005** absolute (≈ **0.10%** better).

**Per‑seed (N=50):**

| Seed | NN+LS+Drone | ALNS+LS+Drone | PointerRL(K=32)+LS+Drone |
|-----:|-------------:|--------------:|--------------------------:|
| 1    | 5.080        | 5.273         | **5.080** |
| 2    | 5.455        | 5.387         | **5.386** |
| 3    | **5.088**    | 5.387         | 5.143     |

**Interpretation (concise):**
- The hybrid RL scheduler **never underperforms ALNS** on the same seed and is **competitive with NN** (tie/lead in 2/3 seeds).
- Heuristics show a **truck vs. wait** trade‑off (shorter truck tours ⇢ larger rendezvous waits). The learned policy achieves a **balanced** schedule.

---

## 🧠 Method Overview

1. **Truck tour**: ALNS with 2/3‑opt + Or‑opt (alternatively NN or Clarke–Wright).
2. **Scheduling**: Feasibility‑masked greedy or beam decoding over **(launch, customer, rendezvous)** triplets.
3. **Exact simulation**: Timeline simulator enforces endurance, launch/recovery, and recharge—**the score is the real makespan**.
4. **RL policy**: Pointer/attention model trained with SCST; inference with best‑of‑K or beam.

---

## 🖼️ Plots & Visualizations

- **Aggregate bar chart**: `paper/agg_bars_all.png` (from `aggregate_multi_seed.py` or `plot_results.py`).
- **Routes and timelines**: route overlays and Gantt‑style timelines can be generated from `solutions.json`.

---

## 📑 Cite

If you use this code, please cite the project (fill in details as appropriate):

```bibtex
@software{vrp_drone_hybrid_2025,
  title  = {A Simple and Reproducible Hybrid Solver for a Truck--Drone VRP with Recharge},
  author = {Meraliyev, Meraryslan and Turan, Cemil and Kadyrov, Shirali},
  year   = {2025},
  url    = {https://github.com/your-org/vrp-drone-hybrid},
  note   = {Hybrid RL + heuristics with feasibility-masked scheduling and exact simulation}
}
```

---

## 📄 License

**MIT** — see `LICENSE` for details.

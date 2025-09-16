# VRP with a Single Drone (Speed 2× Truck) — Strong Baseline + Hybrid RL + Plots

This repository implements a **single-truck, single-drone** VRP variant with:
- Truck speed `v_truck` (default 1.0) and **drone speed `2*v_truck`**.
- **Battery endurance** `E` (max flight time per sortie).
- **Recharge** `R` after each delivery; drone must recharge on the truck before next sortie (recharge overlaps with truck movement).
- Sorties launch/rendezvous only at **truck nodes**; one customer per sortie; one drone at a time.

## Components
- **Truck heuristics**: Clarke–Wright (capacity-aware) + **2-opt** + **Or-opt**.
- **Exact sequential simulator** for truck+drone timeline with endurance + recharge.
- **Greedy hill-climbing drone assignment** that **monotonically improves** makespan.
- **Pointer Network (attention) actor+critic** (REINFORCE), **Best‑of‑K** decoding, then LS + scheduling.
- **Plotting & visualization**:
  - `plot_training.py`: training curve (avg makespan vs. epoch)
  - `plot_results.py`: aggregate bar chart (mean ± s.e. across seeds)
  - `visualize_routes.py`: spatial plot of truck tour + drone sorties
  - `plot_timeline.py`: simple drone sorties timeline

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Single experiment (N=50) with logging + JSON for viz
python main.py --seed 1 --N 50 --epochs 2000 --K 32 --endurance 0.7 --recharge 0.1 --save_json

# Heuristic-only evaluation (no RL)
python main.py --seed 2 --N 60 --epochs 0 --K 0 --eval_only --save_json

# Training curve
python plot_training.py train_seed1_N50.csv

# Aggregate bar chart (use multiple seeds)
python plot_results.py "results_seed*_N50.csv"

# Visualize routes (RL/CW/NN)
python visualize_routes.py solutions_seed1_N50.json --method RL
python visualize_routes.py solutions_seed1_N50.json --method CW

# Drone sorties timeline
python plot_timeline.py solutions_seed1_N50.json --method RL
```


## Advanced research features

- **ALNS (Adaptive Large Neighborhood Search)** for truck tours: random/shaw/worst removal + greedy insertion; SA-style acceptance and operator adaptation.
- **Beam-search drone scheduler** with candidate pruning (top-k per edge), followed by exact resimulation for final events/makespan.
- **Assignment-level VNS** refinement on top of scheduler output (relocate, reassign, drop/add).
- **Transformer-based RL with SCST** (self-critical) to train a sequence policy; greedy decode + LS + beam scheduler at inference.
- **Batch runner** to replicate across seeds and configs; **LaTeX table** generator; **Wilcoxon** paired test script.

### One-click advanced runs (Spyder-friendly)

- **Pointer RL baseline (advanced)**:
  ```bash
  # Uses ALNS backbone + beam scheduler + pointer RL (best-of-K)
  python main.py --config config_adv_pointer.json
  ```

- **Transformer RL + SCST**:
  ```bash
  python main.py --config config_adv_transformer.json
  ```

- **Batch over configs and seeds**:
  ```bash
  python batch.py --configs config_adv_pointer.json config_adv_transformer.json --seeds 1 2 3 4 5
  ```

- **LaTeX table** from all runs:
  ```bash
  python latex_table.py "runs/*/results.csv" --out table_results.tex
  ```

- **Wilcoxon test** (paired):
  ```bash
  python stat_tests.py "runs/*/results.csv"
  ```

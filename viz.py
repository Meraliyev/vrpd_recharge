# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 09:53:38 2025

@author: sduai
"""

from plot_training import main as plot_train
from plot_results import main as plot_agg
from visualize_routes import main as plot_routes
from plot_timeline import main as plot_time

# training curve
#plot_train("runs/exp_seed1_N50/train.csv")

# aggregated bar chart across seeds
plot_agg("runs/exp_seed1_N50/results.csv")

# routes
plot_routes("runs/exp_seed1_N50/solutions.json", method="RL")

# timeline
plot_time("runs/exp_seed1_N50/solutions.json", method="RL")
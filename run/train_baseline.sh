#!/bin/bash
python trainer.py \
-n_iterations 15000 \
-n_agents 10 \
-T 15 \
-seed 11 \
-experiment_name 'baseline' \
-n_experiments 1 \
-network_file social_network_barabasi_albert_graph.txt
#!/bin/bash
python trainer_adv.py \
-n_iterations 15000 \
-n_agents 10 \
-T 15 \
-seed 11 \
-attack \
-n_attack 1 \
-experiment_name 'takeover_attack' \
-n_experiments 1 \
-network_file social_network_barabasi_albert_graph.txt
#!/bin/bash
python trainer_adv.py \
-n_iterations 15000 \
-n_agents 10 \
-T 15 \
-seed 12 \
-attack \
-n_attack 1 \
-random_attacker \
-experiment_name 'random_attack' \
-n_experiments 1 \
-network_file social_network_barabasi_albert_graph.txt

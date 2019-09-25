#!/bin/bash
python trainer.py \
-restore \
-evaluate \
-attack \
-n_attack 1 \
-bias_per_agent 3. \
-restore_run_id 'run_id' \
-restore_file 'filename' \
-experiment_name 'exp_name'

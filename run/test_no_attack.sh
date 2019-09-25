#!/bin/bash
python trainer.py \
-restore \
-evaluate \
-restore_run_id 'run_id' \
-restore_file 'filename' \
-experiment_name 'exp_name'

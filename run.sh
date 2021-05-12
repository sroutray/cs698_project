#!/bin/bash

# set up environment
export PATH=/h/sroutray/.conda/envs/dfl/bin:$PATH

# symlink checkpoint directory to run directory
#ln -s /checkpoint/$USER/$SLURM_JOB_ID /runs/checkpoint

# (while true; do nvidia-smi; top -b -n 1 | head -6; sleep 10; done) &
python train_m2.py
#!/bin/sh
#SBATCH -c 1                # Request 1 CPU core
#SBATCH --gres=gpu:1        # Request one GPUs
#SBATCH -t 0-02:00          # Runtime in D-HH:MM, minimum of 10 mins (this requests 2 hours)
#SBATCH --partition=gpmoo-b # Partition to submit to
#SBATCH --mem=2G            # Request 2G of memory
#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written (%j inserts jobid)
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written (%j inserts jobid)

# Command you want to run on the cluster
# Notice, you must set-up testEval correctly as a conda virtual environment
# Calling this full path makes sure you are running the correct package versions
~/.conda/envs/testEnv/bin/python3.9 train_net_w_gpus.py

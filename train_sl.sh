#!/bin/bash
#SBATCH -J sudoku_policy# Job name
#SBATCH -o sudoku_policy.o%j    # Name of stdout output file (%j expands to jobId)
#SBATCH -e sudoku_policys%j    # Name of stderr output file
#SBATCH -N 1   # Total number of CPU nodes requested
#SBATCH -n 1   # Total number of CPU cores requrested
#SBATCH -t 48:00:00    # Run time (hh:mm:ss)
#SBATCH -p aida --gres=gpu:4 -c 24  # Which queue to run on, and what resources to use


python3 -u sudoku_sl.py --log policy --type sudoku_model --model_type 1 --model_path ./modelsave_policy --gpu_list 0,1,2,3 --train_gpu 0,1,2,3 | tee master_policy.log


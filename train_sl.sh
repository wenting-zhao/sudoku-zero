#!/bin/bash
#SBATCH -J sudoku_pv# Job name
#SBATCH -o sudoku_pvo%j    # Name of stdout output file (%j expands to jobId)
#SBATCH -e sudoku_pvv%j    # Name of stderr output file
#SBATCH -N 1   # Total number of CPU nodes requested
#SBATCH -n 1   # Total number of CPU cores requrested
#SBATCH -t 48:00:00    # Run time (hh:mm:ss)
#SBATCH -p aida --gres=gpu:2 -c 24  # Which queue to run on, and what resources to use


python3 -u sudoku_sl.py --log pv --type sudoku_model --model_type 3 --model_path ./modelsave_pv --gpu_list 0,1 --train_gpu 0,1 | tee master_pv.log


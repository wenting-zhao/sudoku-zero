python3 -u sudoku_zero.py --mode slave --type sudoku_model --model_path ./modelsave --host 127.0.0.1 --gpu_list 0, 1, 2, 3, 4, 5, 6, 7 --num_thread 24 | tee slave.log

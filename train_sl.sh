#python3 -u sudoku_sl.py --type sudoku_model --model_path ./modelsave --gpu_list 0 --train_gpu 0 | tee master.log
python3 -u sudoku_sl.py --type sudoku_model --model_path ./modelsave --gpu_list 0,1,2,3 --train_gpu 0,1,2,3 | tee master.log

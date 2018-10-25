import numpy as np
from subprocess import Popen, PIPE, STDOUT

varmap = dict()
backmap = dict()
i = 1
for x in range(1, 10):
    for y in range(1, 10):
        for z in range(1, 10):
            varmap[(x,y,z)] = i
            backmap[i] = (x,y,z)
            i += 1

data = np.load("hardest500.npy")

with open('plain_sudoku.cnf', 'r') as f:
    sudoku_constraints = f.read()

for sudoku_idx in range(data.shape[0]):
    assumptions = ""
    for i in range(9):
        for j in range(9):
            current_bit = data[sudoku_idx][0][i][j]
            if current_bit != 0:
                assumptions += "{} 0\n".format(str(varmap[(i+1,j+1,current_bit)]))
    p = Popen(['./minicard'], stdout=PIPE, stdin=PIPE, stderr=PIPE)
    result = p.communicate(input=(sudoku_constraints+assumptions).encode())[0].decode()
    print(result)
    result = result.split('\n')[-2].split()[1:]
    solved = np.zeros((9,9), dtype=int)
    for var in result:
        var = var.replace("x", "")
        var = int(var)
        if var > 0:
            x, y, val = backmap[var]
            solved[x-1, y-1] = val
    #print(solved)
    assert np.array_equal(solved, data[sudoku_idx][1])
    print(sudoku_idx, "checked")

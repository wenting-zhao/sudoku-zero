import sys
import math


class Sudoku:
    def __init__(self, n):
        self.n = n
        self.varmap = dict()

        # build a vertex map and an edge map (for mapping vertices and edges to their variables)
        i = 1
        for x in range(1, self.n+1):
            for y in range(1, self.n+1):
                for z in range(1, n+1):
                    self.varmap[(x,y,z)] = i
                    i += 1
        self.nvars = i - 1

    def _getvar(self, x,y,z):
        return self.varmap[(x,y,z)]

    def i_per_row(self):
        for z in range(1, self.n+1):
            for x in range(1, self.n+1):
                i_in_row = [self._getvar(x,y,z) for y in range(1, self.n+1)]
                yield (i_in_row, "<= 1")
                yield (i_in_row, "0")

    def i_per_column(self):
        for z in range(1, self.n+1):
            for y in range(1, self.n+1):
                i_in_column = [self._getvar(x,y,z) for x in range(1, self.n+1)]
                yield (i_in_column, "<= 1")
                yield (i_in_column, "0")

    def i_per_square(self):
        sqrt_n = math.sqrt(self.n)
        assert sqrt_n == int(sqrt_n)
        sqrt_n = int(sqrt_n)
        for z in range(1, self.n+1):
            for x in range(1, self.n+1, sqrt_n):
                for y in range(1, self.n+1, sqrt_n):
                    i_in_square = [self._getvar(p,q,z) for p in range(x, x+sqrt_n) for q in range(y, y+sqrt_n)]
                    yield (i_in_square, "<= 1")
                    yield (i_in_square, "0")

    def i_per_cell(self):
        for x in range(1, self.n+1):
            for y in range(1, self.n+1):
                i_in_cell = [self._getvar(x,y,z) for z in range(1, self.n+1)]
                yield (i_in_cell, "<= 1")
                yield (i_in_cell, "0")

    def make_cnf(self):
        for c in self.i_per_row(): yield c
        for c in self.i_per_column(): yield c
        for c in self.i_per_cell(): yield c
        for c in self.i_per_square(): yield c

    def print_cnf(self):
        print("p cnf+ %d %d" % (self.nvars, 1))
        for clause in self.make_cnf():
            print(" ".join([str(x) for x in clause[0]]) + " " + clause[1])


def main():
    n = int(sys.argv[1])
    d = Sudoku(n)
    d.print_cnf()

if __name__ == '__main__':
    main()

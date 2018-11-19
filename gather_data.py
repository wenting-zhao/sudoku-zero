from os import listdir
import numpy as np
import pickle

data_dir = "./sudoku_data_1"

def _extract_feature(history, pos):
        n_board = history.shape[0]
        ret = np.zeros((n_board, n_board, n_board + 2))
        for (x, y) in zip(range(n_board), range(n_board)):
            ret[x, y, int(history[x, y])] = 1.0
        for (x, y) in pos:
            if x == -1 and y == -1:
                break
            ret[x, y, n_board + 1] = 1.0
        return ret


if __name__ == "__main__":
    f = listdir(data_dir)
    cnt = -1
    X = []
    Y = []
    for item in f:
        if item[0] == 'd' and item[1] == 'a':
            with open(data_dir + "/" + item, "rb") as handle:
                print (item)
                sudoku_data = pickle.load(handle)
                tmp_Y = []
                for ii, one in enumerate(sudoku_data):
                    for (history, pos, nxt_move) in one[0]:
                        if ii == 1 or ii == 3:
                            print (history)
                            print (nxt_move)
                            
                        feature = _extract_feature(history, [x[0] for x in pos])
                        label = np.zeros(16 * 16)
                        label[int(nxt_move[0] * 16 + nxt_move[1])] = 1.0
                        X.append(feature)
                        Y.append(label)
                        tmp_Y.append(label)
                print (np.argmax(tmp_Y, 1))
        cnt += 1
        if cnt >= 10:
            break
    print (np.array(X).shape)
    print (np.array(Y).shape)
    print (np.argmax(np.array(Y)[-3:-1], 1).shape)
    np.save("sudoku_sl_data_test", np.array(X))
    np.save("sudoku_sl_label_test", np.array(Y))
    print (cnt)

from os import listdir
import numpy as np
import pickle

data_dir = "./new_sudoku_data"

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
    value = []
    miao = 0
    for item in f:
        if item[0] == 'n' and item[1] == 'e':
            with open(data_dir + "/" + item, "rb") as handle:
                print (item)
                sudoku_data = pickle.load(handle)
                tmp_Y = []
                for ii, one in enumerate(sudoku_data):
                    for (history, pos, (nxt_move, nxt_value)) in one[0]:
                        #value.append(#TODO:)
                        if type(nxt_move) == int:
                            continue
                        feature = _extract_feature(history, [x[0] for x in pos])
                        label = np.zeros(16 * 16)
                        label[int(nxt_move[0] * 16 + nxt_move[1])] = 1.0
                        X.append(feature)
                        value_label = np.zeros(16)
                        label[int(nxt_value)] = 1.0
                        Y.append(label)
                        value.append(value_label)
                        #tmp_Y.append(label)
        cnt += 1
        break
    print (np.array(X).shape)
    print (np.array(Y).shape)
    print (np.array(value).shape)
    #print (np.array(X).shape)
    #print (np.argmax(np.array(Y)[-3:-1], 1).shape)
    np.save("new_sudoku_sl_data_test", np.array(X))
    np.save("new_sudoku_sl_label_test", np.array(Y))
    np.save("new_sudoku_sl_value_test", np.array(value))
    print (cnt)

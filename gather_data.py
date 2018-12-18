from os import listdir
import numpy as np
import pickle

data_dir = "./final_sudoku_data"

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

def _extract_feature4value(history, pos):
    n_board = history.shape[0]
    ret = np.zeros((n_board, n_board, n_board + 2))
    for (x, y) in zip(range(n_board), range(n_board)):
        ret[x, y, int(history[x, y])] = 1.0
    ret[pos[0], pos[1], n_board + 1] = 1.0

    return ret



if __name__ == "__main__":
    f = listdir(data_dir)
    cnt = -1
    X = []
    Y = []
    value = []
    position = []
    stat = {}
    stat_value = {}
    miao = 0
    for item in f:
        if item[0] == 'n' and item[1] == 'e':
            with open(data_dir + "/" + item, "rb") as handle:
                print (item)
                sudoku_data = pickle.load(handle)
                tmp_Y = []
                for ii, one in enumerate(sudoku_data):
                    for (history, pos, (nxt_move, nxt_value)) in one[0]:
                        #print (history)
                        #print ("miao: ", pos)
                        #print (nxt_move, " ", nxt_value)
                        #value.append(#TODO:)
                        if type(nxt_move) == int:
                            print ("shit!")
                            continue
                        pospos = nxt_move[0] * 16 + nxt_move[1]
                        if pospos in stat:
                            stat[pospos] += 1
                        else:
                            stat[pospos] = 1
                        vv = int(nxt_value) - 1
                        if vv in stat_value:
                            stat_value[vv] += 1
                        else:
                            stat_value[vv] = 1
                        cnt += 1
                        #feature = _extract_feature(history, [x[0] for x in pos])
                        feature4value = _extract_feature4value(history, nxt_move)
                        label = np.zeros(16 * 16)
                        label[int(nxt_move[0] * 16 + nxt_move[1])] = 1.0
                        value_label = np.zeros(16)
                        value_label[int(nxt_value) - 1] = 1.0
                        X.append(feature4value)
                        #X.append(feature)
                        Y.append(label)
                        value.append(value_label)
                        #tmp_Y.append(label)
    print (np.array(X).shape)
    #print (np.array(Y).shape)
    #print (np.array(value).shape)
    #print (np.array(X).shape)
    #print (np.argmax(np.array(Y)[-3:-1], 1).shape)
    #np.save("final_sudoku_sl_data_train", np.array(X))
    np.save("final_sudoku_sl_data4value", np.array(X))
    np.save("final_sudoku_sl_label", np.array(Y))
    np.save("final_sudoku_sl_value", np.array(value))
    print (cnt)
    #for k, v in stat.items():
    #    print (k, 1.0 * v / (1.0 * cnt))
    #print ("value")
    #for k, v in stat_value.items():
    #    print (k, 1.0 * v / (1.0 * cnt))

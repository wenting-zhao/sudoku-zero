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
    data = []
    label = []
    for item in f:
        if item[0] == 'd' and item[1] == 'a':
            with open(data_dir + "/" + item, "rb") as handle:
                data = pickle.load(handle)
                for one in data:
                    print (one)
                    for (history, pos, nxt_move) in one[0]:
                        print (nxt_move)
                        feature = _extract_feature(history, [x[0] for x in pos])
                        data.append(feature)
                        label.append(nxt_move)
    print (np.array(data).shape)
    print (np.array(label).shape)
    
                
    #with open("sudoku_data_1", "wb") as handle:
    #    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #with open("sudoku_label_1", "wb") as handle:
    #    pickle.dump(label, handle, protocol=pickle.HIGHEST_PROTOCOL)


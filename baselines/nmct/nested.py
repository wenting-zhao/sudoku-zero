import numpy as np
from copy import deepcopy
import random

class NestedMCTS():
    def __init__(self, sudoku_size):
        self.sudoku_size = sudoku_size
        self.max_depth = self.sudoku_size ** 2
        self.box_group, self.which_group = self._get_box_group(self.sudoku_size)
        self.root = Node(parent=None, action=None, pos="root")

    # def iterativeNested(self, position, level):
    #   best_score = -1
    #   while True:
    #       score = nested(position, level)
    #       best_score = max(best_score, score)
    #   return best_score
    def __call__(self, sudoku_state, level=1):
        self.sudoku = sudoku_state
        self.explored_nodes = self._get_explored_nodes(sudoku_state)
        self.constraints = self._get_constraints(sudoku_state)
        # self.search_order : [((pos-x, pos-y), [move1, move2, ...]), ...]
        self.search_order = self._get_search_order(self.constraints, self.explored_nodes)
        all_minimum = self._get_all_minimum(self.search_order)
        pos, possible_values = random.choice(all_minimum)
        self.root.depth = np.count_nonzero(self.explored_nodes) - 1
        self._create_leaves(self.root, pos, possible_values)
        curr_node = self.root

        ancestors = []
        best_sequence = []
        while True:
            res = self.nested(curr_node, level, ancestors)
            if res[-1] == "complete":
                return self.new_state(res[0])
            if len(res[0]) > len(best_sequence):
                best_sequence = deepcopy(res)
            next_move = best_sequence.pop(0)
            ancestors.append(next_move)
            for i in curr_node.children:
                if i.action == next_move[1]:
                    curr_node = i
                    break
            else:
                # if running nested at this iteration didn't yield a better results
                curr_node.children.clear()
                self._create_leaves(curr_node, next_move[0], [next_move[1]])
                curr_node = next(iter(curr_node.children))
            new_constraints = self._update_constraints(ancestors)
            explored = self._update_explored(ancestors)
            search_order = self._get_search_order(new_constraints, explored)
            if len(search_order[0][1]) == 0 or len(search_order) == 0:
                break
            all_minimum = self._get_all_minimum(search_order)
            pos, possible_values = random.choice(all_minimum)
            self._create_leaves(curr_node, pos, possible_values)
        return self.new_state(ancestors)

    def nested(self, node, level, ancestors):
        best_score = -1
        if level == 1:
            for child in node.children:
                child.reward, child.move_sequence = self._roll_out(child, ancestors+[((child.pos), child.action)])
                if child.reward == self.max_depth:
                    return child.move_sequence, "complete"
            best_child = sorted(node.children, key=lambda e: e.reward, reverse=True)[0]
        else:
            best_candidates = []
            for child in node.children:
                candidate = nested(child, level-1)
                best_candidates.append(candidate)
            best_child = sorted(best_candidates, key=lambda e: e.reward, reverse=True)[0]
        if best_child.reward > best_score:
            best_score = best_child.reward
            best_sequence = deepcopy(best_child.move_sequence)
        return best_sequence

    def _roll_out(self, node, ancestors):
        depth = node.depth
        # make sure we make the right initial depth
        assert depth == len(ancestors) + self.root.depth
        # record move sequence in case this rollout find a sol'n
        move_sequence = [(node.pos, node.action)]
        new_constraints = self._update_constraints(ancestors)
        new_explored = self._update_explored(ancestors)
        cell_possible_actions = self._get_search_order(new_constraints, new_explored)
        cell_possible_actions = {i[0]: i[1] for i in cell_possible_actions}
        while depth < self.max_depth:
            depth += 1
            if len(cell_possible_actions) == 0:
                break
            for i in range(self.sudoku_size):
                if (i, node.pos[1]) in cell_possible_actions:
                    if node.action in cell_possible_actions[(i, node.pos[1])]:
                        cell_possible_actions[(i, node.pos[1])].remove(node.action)
            for j in range(self.sudoku_size):
                if (node.pos[0], j) in cell_possible_actions:
                    if node.action in cell_possible_actions[(node.pos[0], j)]:
                        cell_possible_actions[(node.pos[0], j)].remove(node.action)
            for (i, j) in self.box_group[self.which_group[node.pos]]:
                if (i, j) in cell_possible_actions:
                    if node.action in cell_possible_actions[(i, j)]:
                        cell_possible_actions[(i, j)].remove(node.action)
            search_order = sorted(cell_possible_actions.items(), key=lambda kv: len(kv[1]))
            all_minimum = self._get_all_minimum(search_order)
            pos, actions = random.choice(all_minimum)
            del cell_possible_actions[pos]
            if len(actions) == 0:
                break
            else:
                node = Node(node, random.choice(list(actions)), pos)
                move_sequence.append((node.pos, node.action))
        assert depth == len(move_sequence)+len(ancestors)+self.root.depth
        return depth, move_sequence

    def new_state(self, ancestors):
        new = deepcopy(self.sudoku)
        for move in ancestors:
            (x, y), action = move[:2]
            new[x, y] = action
        return new

    def _create_leaves(self, node, pos, available_moves):
        for move in available_moves:
            new_child = Node(parent=node, action=move, pos=pos)
            node.children.add(new_child)

    def _get_constraints(self, sudoku):
        constraints = []

        assert self.sudoku_size == sudoku.shape[0], self.sudoku_size == sudoku.shape[1]

        # get row constraints, each element in the constraints list is also a list,
        # representing which elements are still available for that row
        for i in range(self.sudoku_size):
            constraints.append(set(range(1,self.sudoku_size+1)) - set(sudoku[i, :]))

        # column constraints
        for i in range(self.sudoku_size):
            constraints.append(set(range(1,self.sudoku_size+1)) - set(sudoku[:, i]))

        # box constraints
        for key in self.box_group.keys():
            box_number = set([sudoku[x, y] for (x, y) in self.box_group[key]])
            constraints.append(set(range(1,self.sudoku_size+1)) - set(box_number))
        return constraints

    def _get_search_order(self, constraints, explored):
        possible_values = dict()
        for i in range(self.sudoku_size):
            for j in range(self.sudoku_size):
                if explored[i, j] == 0:
                    possible_values[(i, j)] = constraints[i] & constraints[j+self.sudoku_size] & constraints[self.which_group[(i, j)]+2*self.sudoku_size]
        return sorted(possible_values.items(), key=lambda kv: len(kv[1]))

    def _get_filled_cells(self, sudoku):
        explored = deepcopy(sudoku)
        for (i, j), x in np.ndenumerate(explored):
            if x == 0:
                explored[i, j] = False
            else:
                explored[i, j] = True
        return explored

    def _update_explored(self, additional_nodes):
        new_explored = deepcopy(self.explored_nodes)
        for pseudo_node in additional_nodes:
            new_explored[pseudo_node[0]] = 1
        return new_explored

    def _get_box_group(self, sudoku_size):
        i = 0
        box_group = dict()
        which_group = dict()
        sqrt_n = int(np.sqrt(sudoku_size))
        for x in range(0, sudoku_size, sqrt_n):
            for y in range(0, sudoku_size, sqrt_n):
                box_group[i] = [(p, q) for p in range(x, x+sqrt_n) for q in range(y, y+sqrt_n)]
                for z in box_group[i]:
                    which_group[z] = i
                i += 1
        return box_group, which_group

    def _get_explored_nodes(self, sudoku):
        explored = deepcopy(sudoku)
        for (i, j), x in np.ndenumerate(explored):
            if x == 0:
                explored[i, j] = False
            else:
                explored[i, j] = True
        return explored

    def _get_all_minimum(self, search_order):
        all_minimum = []
        minimum = len(search_order[0][1])
        for cell in search_order:
            if len(cell[1]) == minimum:
                all_minimum.append(cell)
        return all_minimum

    # compute next level that has fewest available actions
    def _update_constraints(self, additional_nodes):
        new_constraints = deepcopy(self.constraints)
        for node in additional_nodes:
            (x, y), action = node[:2]
            try:
                new_constraints[x].remove(action)
                new_constraints[y+self.sudoku_size].remove(action)
                new_constraints[self.which_group[(x, y)]+2*self.sudoku_size].remove(action)
            # sometimes there is unit propagation between mcts calls, and thus the constraints
            # are not updated in time, and thus there will be situations where an action is no
            # longer possible but still in constraints
            except KeyError:
                pass
        return new_constraints

	# def sample(self, position):
	# 	while not end_game:
	# 		position = play(position, random_move)
	# 	return score


class Node():
    """
    A node holding a state in the tree.
    """
    def __init__(self, parent, action, pos):
        self.parent = parent
        self.children = set()
        self.pos = pos
        self.action = action  # a number from 1-n
        self.reward = 0
        self.move_sequence = []

        if self.parent is not None:
            self.depth = parent.depth + 1


def main():
    sudoku = np.asarray([[ 5,  0,  0, 13,  2,  0,  0, 14,  0,  0,  0, 15,  4,  0,  0,  0],
                        [ 2,  6, 10,  0,  1,  5,  0,  0,  0,  8,  0,  0,  0,  0,  0, 15],
                        [ 3,  0,  0,  0,  0,  0,  0, 16,  0,  0,  0,  0,  0,  0, 10, 14],
                        [ 4,  8,  0,  0,  0,  0,  0,  0,  0,  6, 10,  0,  0,  5,  0,  0],
                        [ 0,  5,  0,  0,  0,  2, 14,  0,  0,  0, 15,  0,  0,  0, 16,  0],
                        [ 6,  2,  0,  0,  0,  0, 13,  0,  0,  4, 16,  0,  0,  3, 15,  0],
                        [ 7,  0,  0, 11,  0,  0,  0,  0,  5,  1,  0,  9,  6,  0, 14, 10],
                        [ 0,  0,  0, 12,  0,  3,  0,  0,  6,  0,  0,  0,  5,  1, 13,  0],
                        [ 0,  0,  1,  0, 10,  0,  0,  0, 11,  0,  0,  7, 12,  0,  4,  0],
                        [ 0,  0,  2,  6,  9, 13,  1,  0,  0,  0,  0,  0, 11, 15,  0,  0],
                        [11,  0,  0,  0, 12,  0,  4,  0,  9,  0,  0,  5,  0,  0,  0,  0],
                        [ 0, 16,  0,  0,  0, 15,  3,  0,  0,  0,  2,  0,  0,  0,  1,  0],
                        [13,  0,  5,  1,  0,  0,  0,  2, 15, 11,  7,  3,  0, 12,  0,  0],
                        [ 0,  0,  0,  0, 13,  0,  0,  0, 16,  0,  0,  0,  0, 11,  0,  0],
                        [ 0,  0,  7,  0, 16,  0,  0,  0,  0,  0,  5,  0, 14,  0,  6,  0],
                        [ 0, 12,  0,  0,  0, 11,  7,  3,  0,  0,  0,  2,  0,  0,  0,  0]])
    #sudoku = np.zeros((n,n)).astype(int)
    n = sudoku.shape[0]
    nmct = NestedMCTS(n)
    cnt = 0
    while True:
        cnt += 1
        res = nmct(sudoku)
        if np.count_nonzero(res) == n*n:
            print(res, cnt)
            break

if __name__ == '__main__':
    main()
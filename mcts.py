import random
import numpy as np
import copy


class MCTS():
    """
    The central MCTS class, which performs the tree search. It gets a
    tree policy, a default policy, and a backup strategy.
    See e.g. Browne et al. (2012) for a survey on monte carlo tree search
    """
    def __init__(self, model, sudoku_size, gen_data=False, infer=False, rollout=10, ucb1_confidence=1.41):
        self.model = model
        self.infer = infer
        self.gen_data = gen_data
        self.sudoku_size = sudoku_size
        self.rollout = rollout
        self.max_depth = self.sudoku_size ** 2
        self.box_group, self.which_group = self._get_box_group(self.sudoku_size)
        self.ucb1_confidence = ucb1_confidence
        self.root = Node(parent=None, action=None, pos="root")
        self.sample = 0

    def __call__(self, sudoku_state, n=10000):
        """
        Run the monte carlo tree search.

        :param n: The number of roll-outs to be performed
        :return:
        """
        self.sudoku = sudoku_state
        self.explored_nodes = self._get_explored_nodes(sudoku_state)
        self.constraints = self._get_constraints(sudoku_state)
        # self.search_order : [((pos-x, pos-y), [move1, move2, ...]), ...]
        self.search_order = self._get_search_order(self.constraints, self.explored_nodes)
        all_minimum = self._get_all_minimum(self.search_order)
        if self.infer:
            features = np.zeros((1, 16, 16, 18))
            features[0] = self.model._extract_feature(self.sudoku, [x[0] for x in all_minimum])
            max_prob = np.argmax(self.model.predict(features))
            pos = divmod(max_prob, 16)
            for elm in all_minimum:
                if elm[0] == pos:
                    possible_values = elm[1]
                    break
            assert possible_values is not None
        else:
            pos, possible_values = random.choice(all_minimum)
        self.root.depth = np.count_nonzero(self.explored_nodes) - 1
        self._create_leaves(self.root, pos, possible_values)

        for i in range(n):
            res = self._get_next_node(self.root)
            assert (len(res) != 0)
            if res[0] == "unsat":
                node.reward = 0
                self.backup(node)
                continue
            else:
                node, ancestors = res
            reward = []
            if_found = False
            for _ in range(self.rollout):
                depth, _ = self._roll_out(node, ancestors)
                reward.append(depth / self.max_depth)
                self.sample += 1
                if depth == self.max_depth:
                    if_found = True
                    break
            node.reward = max(reward)
            if if_found:
                if self.gen_data:
                    return self._get_search_info(ancestors), i
                else:
                    return i
            self.backup(node)
        return None

    def _get_search_info(self, ancestors):
        search_info = []
        running_sudoku = copy.deepcopy(self.sudoku)
        cell_possible_actions = {i[0]: i[1] for i in self.search_order}
        for (pos, action) in ancestors:
            search_order = sorted(cell_possible_actions.items(), key=lambda kv: len(kv[1]))
            all_minimum = self._get_all_minimum(search_order)
            search_info.append((copy.deepcopy(running_sudoku), copy.deepcopy(all_minimum), pos))
            running_sudoku[pos] = action
            for i in range(self.sudoku_size):
                if (i, pos[1]) in cell_possible_actions:
                    if action in cell_possible_actions[(i, pos[1])]:
                        cell_possible_actions[(i, pos[1])].remove(action)
            for j in range(self.sudoku_size):
                if (pos[0], j) in cell_possible_actions:
                    if action in cell_possible_actions[(pos[0], j)]:
                        cell_possible_actions[(pos[0], j)].remove(action)
            for (i, j) in self.box_group[self.which_group[pos]]:
                if (i, j) in cell_possible_actions:
                    if action in cell_possible_actions[(i, j)]:
                        cell_possible_actions[(i, j)].remove(action)
            del cell_possible_actions[pos]
        return search_info

    def reset_sample(self):
        self.sample = 0

    def _compute_softmax(self, node):
        # note that this is only correct with 1d array
        x = [0] * self.sudoku_size
        for child in node.children:
            x[child.action-1] = child.visited
        refined_x = [i for i in x if i > 0]
        e_x = np.exp(refined_x - np.max(refined_x))
        refined_x = e_x / e_x.sum()
        idx = 0
        for i in range(len(x)):
            if x[i] > 0:
                x[i] = refined_x[idx]
                idx += 1
        return x

    def _best_child(self, node):
        most_promising_node = None
        most_promising = float("-inf")
        #deepest_node = 0
        #deepest = 0
        x = np.zeros(len(node.children))
        children = list(node.children)
        for i in range(len(children)):
            x[i] = self.compute_tree_policy(children[i])
        e_x = np.exp(x - np.max(x))
        distribution = e_x / e_x.sum()
        res = np.random.choice(children, 1, p=distribution)
        return res[0]

    def _get_next_node(self, node):
        ancestors = []
        while node.depth < self.max_depth:
            untried_nodes = node.untried_nodes()
            if len(untried_nodes) > 0:
                untried = random.choice(untried_nodes)
                ancestors.append(((untried.pos), untried.action))
                return untried, ancestors
            else:
                # no possible actions at this point
                if len(node.children) == 0:
                    return "unsat", node
                node = self._best_child(node)
                ancestors.append(((node.pos), node.action))
                if len(node.children) == 0:
                    new_constraints = self._update_constraints(ancestors)
                    explored = self._update_explored(ancestors)
                    if np.count_nonzero(explored) == self.sudoku_size ** 2:
                        return None, ancestors
                    search_order = self._get_search_order(new_constraints, explored)
                    all_minimum = self._get_all_minimum(search_order)
                    if self.infer:
                        features = np.zeros((1, 16, 16, 18))
                        features[0] = self.model._extract_feature(self.new_state(ancestors), [x[0] for x in all_minimum])
                        max_prob = np.argmax(self.model.predict(features))
                        pos = divmod(max_prob, 16)
                        for elm in all_minimum:
                            if elm[0] == pos:
                                possible_values = elm[1]
                                break
                        assert possible_values is not None
                    else:
                        pos, possible_values = random.choice(all_minimum)
                    self._create_leaves(node, pos, possible_values)
        return random.choice(list(node.children)), ancestors

    # compute next level that has fewest available actions
    def _update_constraints(self, additional_nodes):
        new_constraints = copy.deepcopy(self.constraints)
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
        new = copy.deepcopy(self.sudoku)
        for move in ancestors:
            (x, y), action = move[:2]
            new[x, y] = action
        return new

    def compute_tree_policy(self, node):
        res = (node.score + self.ucb1_confidence * np.sqrt(2 * np.log(node.parent.visited) / node.visited))
        return res

    def backup(self, node):
        """
        A monte carlo update as in classical UCT.

        See feldman amd Domshlak (2014) for reference.
        :param node: The node to start the backup from
        """
        r = node.reward
        while node is not None:
            node.visited += 1
            node.score = ((node.visited - 1)/node.visited) * node.score + 1/node.visited * r
            node = node.parent

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

    def _get_search_order(self, constraints, explored):
        possible_values = dict()
        for i in range(self.sudoku_size):
            for j in range(self.sudoku_size):
                if explored[i, j] == 0:
                    possible_values[(i, j)] = constraints[i] & constraints[j+self.sudoku_size] & constraints[self.which_group[(i, j)]+2*self.sudoku_size]
        return sorted(possible_values.items(), key=lambda kv: len(kv[1]))

    def _get_all_minimum(self, search_order):
        all_minimum = []
        minimum = len(search_order[0][1])
        for cell in search_order:
            if len(cell[1]) == minimum:
                all_minimum.append(cell)
        return all_minimum

    def _get_explored_nodes(self, sudoku):
        explored = copy.deepcopy(sudoku)
        for (i, j), x in np.ndenumerate(explored):
            if x == 0:
                explored[i, j] = False
            else:
                explored[i, j] = True
        return explored

    def _update_explored(self, additional_nodes):
        new_explored = copy.deepcopy(self.explored_nodes)
        for pseudo_node in additional_nodes:
            new_explored[pseudo_node[0]] = 1
        return new_explored


class Node():
    """
    A node holding a state in the tree.
    """
    def __init__(self, parent, action, pos):
        self.parent = parent
        self.children = set()
        self.action = action  # a number from 1-9
        self.visited = 0
        self.score = 0
        self.reward = 0
        self.prior = 0
        self.pos = pos

        if self.parent is not None:
            self.depth = parent.depth + 1

    def untried_nodes(self):
        return [node for node in self.children if node.visited == 0]

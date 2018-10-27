import random
import numpy as np
import copy


class MCTS():
    """
    The central MCTS class, which performs the tree search. It gets a
    tree policy, a default policy, and a backup strategy.
    See e.g. Browne et al. (2012) for a survey on monte carlo tree search
    """
    def __init__(self, sudoku_size, ucb1_confidence=1.41):
        self.sudoku_size = sudoku_size
        self.max_depth = self.sudoku_size ** 2
        self.box_group, self.which_group = self._get_box_group(self.sudoku_size)
        self.ucb1_confidence = ucb1_confidence

    def __call__(self, sudoku_state, n=1500):
        """
        Run the monte carlo tree search.

        :param root: The StateNode
        :param n: The number of roll-outs to be performed
        :return:
        """

        self.constraints = self._get_constraints(sudoku_state)
        # self.search_order : [((pos-x, pos-y), [move1, move2, ...]), ...]
        self.search_order = self._get_search_order()

        root = Node(parent=None, action=None, pos=None)
        root.depth = 0
        if len(self.search_order[0][1]) == 1:
            print("Next node is determined without running MCTS.")
            return self.search_order[0][1][0]
        else:
            pos, possible_values = self.search_order.pop()
            self._create_leaves(root, pos, possible_values)

        for _ in range(n):
            node = self._get_next_node(root, self.tree_policy)
            node.reward, move_sequence = self._roll_out(node)
            # when a solution is found in rollout...
            if node.reward == self.max_depth:
                while node.parent is not None:
                    move_sequence.append(((node.pos), node.action))
                return move_sequence
            backup(node)

        return _best_child(root)

    def _best_child(self, node):
        most_promising_node = 0
        most_promising = 0
        deepest_node = 0
        deepest = 0
        for child in node.children:
            ucb1 = self.UCB1(child)
            if ucb1 > most_promising:
                most_promising = ucb1
                most_promising_node = child
            if child.depth > deepest:
                deepest = child.depth
                deepest_node = child
            if most_promising_node.action != deepest_node.action:
                print("deepest node isn't most promising node.")
        return most_promising_node

    def _get_next_node(self, node):
        all_parents = []
        while node.depth < self.max_depth:
            untried_nodes = node.untried_nodes()
            if len(untried_nodes) > 0:
                untried = random.choice(untried_nodes)
                all_parents.append(((untried.pos), untried.action))
                return untried, all_parents
            else:
                node = self._best_child(node)
                all_parents.append(((node.pos), node.action))
                if len(node.children) == 0:
                    pos, possible_values = self._next_level(all_parents)
                    self._create_leaves(node, pos, possible_values)
        return random.choice(node.children), all_parents

    # compute next level that has fewest available actions
    def _next_level(self, additional_nodes):
        new_constraints = copy.deepcopy(self.constraints)
        for node in additional_nodes:
            x, y = node.pos
            new_constraints[x].remove(node.action)
            new_constraints[y+self.sudoku_size].remove(node.action)
            new_constraints[self.which_group[(x, y)]+2*self.sudoku_size].remove(node.action)
        return self._get_search_order(new_constraints).pop()

    def _roll_out(self, node, cell_possible_actions):
        depth = node.depth
        move_sequence = []
        cell_possible_actions = copy.deepcopy(self.search_order)
        while depth < self.max_depth:
            depth += 1
            move_sequence.append((node.pos, node.action))
            for i in range(self.sudoku_size):
                if cell_possible_actions[(i, node.pos[1])]:
                    cell_possible_actions[(i, node.pos[1])].remove(node.action)
            for j in range(self.sudoku_size):
                if cell_possible_actions[(node.pos[0], j)]:
                    cell_possible_actions[(node.pos[0], j)].remove(node.action)
            pos, actions = sorted(cell_possible_actions.items(), key=lambda kv: len(kv[1])).pop()
            if len(actions) == 0:
                break
            else:
                node = Node(node, random.choice(actions), pos)
        return depth, move_sequence

    def UCB1(self, node):
        return (node.score +
                self.ucb1_confidence * np.sqrt(2 * np.log(node.parent.visited) / node.visited))

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
        sudoku = np.asarray(sudoku)
        constraints = []

        assert sudoku_size == sudoku.shape[0], sudoku_size == sudoku.shape[1]

        # get row constraints, each element in the constraints list is also a list,
        # representing which elements are still available for that row
        for i in range(self.sudoku_size):
            constraints.append(set(range(1,10)) - set(sudoku[i, :]))

        # column constraints
        for i in range(self.sudoku_size):
            constraints.append(set(range(1,10)) - set(sudoku[:, i]))

        # box constraints
        for key in self.box_group.keys():
            box_number = set([sudoku[x, y] for (x, y) in self.box_group[key]])
            constraints.append(set(range(1,10)) - set(box_number))
        return constraints

    def _get_box_group(self, sudoku_size):
        i = 0
        sqrt_n = int(np.sqrt(sudoku_size))
        for x in range(0, sudoku_size, sqrt_n):
            for y in range(0, sudoku_size, sqrt_n):
                box_group[i] = [(p, q) for p in range(x, x+sqrt_n) for q in range(y, y+sqrt_n)]
                for z in box_group[i]:
                    which_group[z] = i
                i += 1
        return box_group, which_group

    def _get_search_order(self, constraints):
        possible_values = dict()
        for i in range(len(sudoku)):
            for j in range(len(sudoku)):
                possible_values[(i, j)] = constraints[i] & constraints[j+self.sudoku_size] & constraints[self.which_group[(i, j)]+2*self.sudoku_size]
        return sorted(possible_values.items(), key=lambda kv: len(kv[1]))


class Node(Node):
    """
    A node holding a state in the tree.
    """
    def __init__(self, parent, action, pos):
        self.parent = parent
        self.children = {}
        self.action = action  # a number from 1-9
        self.depth = parent.depth + 1
        self.visited = 0
        self.score = 0
        self.reward = 0
        self.pos = pos

    def untried_nodes(self):
        return [node for node in self.children if node.visited == 0]

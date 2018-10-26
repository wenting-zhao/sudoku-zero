import random
import numpy as np


class MCTS():
    """
    The central MCTS class, which performs the tree search. It gets a
    tree policy, a default policy, and a backup strategy.
    See e.g. Browne et al. (2012) for a survey on monte carlo tree search
    """
    def __init__(self, sudoku_size, ucb1_confidence=1.41):
        self.sudoku_size = sudoku_size
        self.box_group, self.which_group = self._get_box_group(self.sudoku_size)
        self.ucb1_confidence = ucb1_confidence

    def __call__(self, root, sudoku_state, n=1500):
        """
        Run the monte carlo tree search.

        :param root: The StateNode
        :param n: The number of roll-outs to be performed
        :return:
        """

        self.constraints = self._get_constraints(sudoku_state)
        self.search_order = self._get_search_order()

        if root.parent is not None:
            raise ValueError("Root's parent must be None.")

        for _ in range(n):
            node = _get_next_node(root, self.tree_policy)
            node.score = self._roll_out(node)
            backup(node)

        return _best_child(root)

    def _expand(state_node):
        return random.choice(state_node.untried_actions)

    def _best_child(state_node):
        most_promising_node = 0
        most_promising = 0
        deepest_node = 0
        deepest = 0
        for child in state_node.children:
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

    def _get_next_node(state_node):
        # TODO: next node is selected from the next fewest actions node
        while len(state_node.children) != 0:
            if state_node.untried_actions:
                return _expand(state_node)
            else:
                state_node = _best_child(state_node)
        return state_node

    def _roll_out(state_node):
        depth = 0
        state = state_node.state
        parent = state_node.parent.parent.state
        action = state_node.parent.action
        # TODO: in each iteration, need to add new constraints for two purposes:
        #       1) know which node to choose next
        #       2) know when to stop
        #       3) when depth is n*n, then the search can stop because a solution has found
        while len(state_node.children) != 0:
            depth += 1
            action = random.choice(state_node.state.actions)
            parent = state
            state = parent.perform(action)

        return reward

    def UCB1(node):
        return (node.score +
                self.ucb1_confidence * np.sqrt(2 * np.log(node.parent.visited) / node.visited))

    def backup(node):
        """
        A monte carlo update as in classical UCT.

        See feldman amd Domshlak (2014) for reference.
        :param node: The node to start the backup from
        """
        r = node.score
        while node is not None:
            node.visited += 1
            node.score = ((node.visited - 1)/node.visited) * node.score + 1/node.visited * r
            node = node.parent
            # TODO: record best sequence

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

    def _get_search_order(self):
        possible_values = dict()
        for i in range(len(sudoku)):
            for j in range(len(sudoku)):
                possible_values[(i, j)] = len(constraints[i] & constraints[j+sudoku_size] & constraints[which_group[(i, j)]+2*sudoku_size])
                return sorted(possible_values.items(), key=lambda kv: kv[1])


class Node(Node):
    """
    A node holding a state in the tree.
    """
    def __init__(self, parent, action, pos, sudoku_state):
        self.parent = parent
        self.children = {}
        self.action = action  # a number from 1-9
        self.best_sequence = []  # [(state_0, action_0), ..., (state_n, action_n)]
        self.depth = len(best_sequence)
        self.visited = 0
        self.score = 0
        self.pos = pos
        # these are the constraints added during tree search in addition to the original constraints
        self.additional_constraints = []

        # TODO: add children

    def untried_actions(self):
        """
        All actions which have never be performed
        :return: A list of the untried actions.
        """
        return [a for a in self.children if self.children[a].visited == 0]

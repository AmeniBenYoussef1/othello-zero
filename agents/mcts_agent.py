import math
import random


class _Node:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = {}
        self.visits = 0
        self.wins = 0.0
        self.untried_moves = None


class MCTSAgent:
    def __init__(self, game, n_simulations=200, c=1.4, seed=None):
        self.game = game
        self.n_simulations = n_simulations
        self.c = c
        self.rng = random.Random(seed)

    def choose_move(self, game, state):
        root = _Node(state)
        root.untried_moves = self._moves_or_pass(state)
        for _ in range(self.n_simulations):
            node = self._select(root)
            result = self._rollout(node.state)
            self._backpropagate(node, result)
        best_child = max(root.children.values(), key=lambda n: n.visits)
        return best_child.move

    def _moves_or_pass(self, state):
        moves = self.game.legal_move_squares(state)
        return moves if moves else [-1]

    def _select(self, node):
        while not self.game.is_terminal(node.state):
            if node.untried_moves is None:
                node.untried_moves = self._moves_or_pass(node.state)
            if node.untried_moves:
                move = node.untried_moves.pop(self.rng.randrange(len(node.untried_moves)))
                if move == -1:
                    child_state = self.game.apply_pass(node.state)
                else:
                    child_state = self.game.apply_move(node.state, move)
                child = _Node(child_state, parent=node, move=move)
                node.children[move] = child
                return child
            node = self._best_uct_child(node)
        return node

    def _best_uct_child(self, node):
        log_n = math.log(node.visits)

        def uct(child):
            exploit = 1 - (child.wins / child.visits)
            explore = self.c * math.sqrt(log_n / child.visits)
            return exploit + explore

        return max(node.children.values(), key=uct)

    def _rollout(self, state):
        rollout_state = state
        while not self.game.is_terminal(rollout_state):
            moves = self.game.legal_move_squares(rollout_state)
            if moves:
                rollout_state = self.game.apply_move(rollout_state, self.rng.choice(moves))
            else:
                rollout_state = self.game.apply_pass(rollout_state)
        return self.game.result(rollout_state)

    def _backpropagate(self, node, result):
        while node is not None:
            node.visits += 1
            node.wins += (result + 1) / 2
            result = -result
            node = node.parent

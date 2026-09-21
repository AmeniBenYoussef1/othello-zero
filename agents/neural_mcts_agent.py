import math

import numpy as np
import torch

from model.encode import encode_state


class _Node:
    def __init__(self, state, prior=1.0, parent=None, move=None):
        self.state = state
        self.prior = prior
        self.parent = parent
        self.move = move
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.expanded = False

    @property
    def q(self):
        return self.value_sum / self.visits if self.visits else 0.0


class NeuralMCTSAgent:
    def __init__(self, game, network, n_simulations=200, c_puct=1.5,
                 dirichlet_alpha=0.3, dirichlet_eps=0.25, device="cpu"):
        self.game = game
        self.network = network.to(device).eval()
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.device = device

    def search(self, state, add_noise=False):
        root = _Node(state)
        self._expand(root, add_noise=add_noise)
        for _ in range(self.n_simulations):
            node = root
            path = [node]
            while node.expanded and node.children:
                move, node = self._select_child(node)
                path.append(node)
            if self.game.is_terminal(node.state):
                value = self.game.result(node.state)
            else:
                value = self._expand(node)
            self._backpropagate(path, value)
        return root

    def choose_move(self, game, state, temperature=0.0, add_noise=False):
        root = self.search(state, add_noise=add_noise)
        visit_counts = {move: child.visits for move, child in root.children.items()}
        if temperature <= 0:
            return max(visit_counts, key=visit_counts.get)
        moves = list(visit_counts.keys())
        weights = np.array([v ** (1.0 / temperature) for v in visit_counts.values()], dtype=np.float64)
        weights /= weights.sum()
        return np.random.choice(moves, p=weights)

    def _evaluate(self, state):
        planes = encode_state(self.game, state)
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.network(x)
        logits = logits.squeeze(0).cpu().numpy()
        value = float(value.item())

        moves = self.game.legal_move_squares(state)
        if moves:
            move_ids, keys = moves, moves
        else:
            move_ids, keys = [self.game.num_squares], [-1]

        selected = logits[move_ids]
        selected = selected - selected.max()
        probs = np.exp(selected)
        probs /= probs.sum()
        priors = {mv: float(p) for mv, p in zip(keys, probs)}
        return priors, value

    def _expand(self, node, add_noise=False):
        priors, value = self._evaluate(node.state)
        moves = list(priors.keys())
        if add_noise and moves:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(moves))
            priors = {
                mv: (1 - self.dirichlet_eps) * priors[mv] + self.dirichlet_eps * eps_noise
                for mv, eps_noise in zip(moves, noise)
            }
        for move, prior in priors.items():
            child_state = (
                self.game.apply_pass(node.state)
                if move == -1
                else self.game.apply_move(node.state, move)
            )
            node.children[move] = _Node(child_state, prior=prior, parent=node, move=move)
        node.expanded = True
        return value

    def _select_child(self, node):
        sqrt_total = math.sqrt(node.visits)
        best_score, best_move, best_child = -float("inf"), None, None
        for move, child in node.children.items():
            score = -child.q + self.c_puct * child.prior * sqrt_total / (1 + child.visits)
            if score > best_score:
                best_score, best_move, best_child = score, move, child
        return best_move, best_child

    def _backpropagate(self, path, value):
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            value = -value

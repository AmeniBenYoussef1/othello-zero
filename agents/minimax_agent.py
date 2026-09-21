from agents.positional_agent import build_weights


class MinimaxAgent:
    def __init__(self, game, depth=4, mobility_weight=5):
        self.game = game
        self.depth = depth
        self.weights = build_weights(game.n)
        self.mobility_weight = mobility_weight

    def choose_move(self, game, state):
        moves = game.legal_move_squares(state)
        best_move = moves[0]
        best_score = float("-inf")
        for sq in moves:
            child = game.apply_move(state, sq)
            score = -self._search(child, self.depth - 1, float("-inf"), float("inf"))
            if score > best_score:
                best_score = score
                best_move = sq
        return best_move

    def _search(self, state, depth, alpha, beta):
        if self.game.is_terminal(state):
            return self.game.result(state) * 100000
        if depth == 0:
            return self._evaluate(state)
        moves = self.game.legal_move_squares(state)
        if not moves:
            child = self.game.apply_pass(state)
            return -self._search(child, depth - 1, -beta, -alpha)
        best = float("-inf")
        for sq in moves:
            child = self.game.apply_move(state, sq)
            score = -self._search(child, depth - 1, -beta, -alpha)
            best = max(best, score)
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        return best

    def _evaluate(self, state):
        me, opp, _ = state
        score = 0
        for sq in range(self.game.num_squares):
            if (me >> sq) & 1:
                score += self.weights[sq]
            elif (opp >> sq) & 1:
                score -= self.weights[sq]
        my_mobility = len(self.game.legal_move_squares(state))
        opp_mobility = len(self.game.legal_move_squares((opp, me, state[2])))
        score += self.mobility_weight * (my_mobility - opp_mobility)
        return score

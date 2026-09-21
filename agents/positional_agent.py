def build_weights(n):
    weights = [[1] * n for _ in range(n)]
    corners = [(r, c) for r in (0, n - 1) for c in (0, n - 1)]
    for r in range(n):
        for c in range(n):
            if r in (0, n - 1) and c in (0, n - 1):
                weights[r][c] = 100
            elif r in (0, n - 1) or c in (0, n - 1):
                weights[r][c] = 10
    for r in range(n):
        for c in range(n):
            if weights[r][c] == 100:
                continue
            if any(abs(r - cr) <= 1 and abs(c - cc) <= 1 for cr, cc in corners):
                weights[r][c] = -20
    flat = [0] * (n * n)
    for r in range(n):
        for c in range(n):
            flat[r * n + c] = weights[r][c]
    return flat


class PositionalAgent:
    def __init__(self, game):
        self.weights = build_weights(game.n)

    def choose_move(self, game, state):
        moves = game.legal_move_squares(state)
        return max(moves, key=lambda sq: self.weights[sq])

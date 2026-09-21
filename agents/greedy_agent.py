class GreedyAgent:
    def choose_move(self, game, state):
        moves = game.legal_move_squares(state)
        return max(moves, key=lambda sq: game.count_flips(state, sq))

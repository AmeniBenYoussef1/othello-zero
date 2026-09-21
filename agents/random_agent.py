import random


class RandomAgent:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def choose_move(self, game, state):
        moves = game.legal_move_squares(state)
        return self.rng.choice(moves)

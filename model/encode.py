import numpy as np


def encode_state(game, state):
    n = game.n
    me, opp, _ = state
    legal = game.legal_moves(state)
    planes = np.zeros((3, n, n), dtype=np.float32)
    for sq in range(game.num_squares):
        r, c = divmod(sq, n)
        if (me >> sq) & 1:
            planes[0, r, c] = 1.0
        if (opp >> sq) & 1:
            planes[1, r, c] = 1.0
        if (legal >> sq) & 1:
            planes[2, r, c] = 1.0
    return planes

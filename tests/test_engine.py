import random

import pytest

from engine.othello import Othello



def perft(game: Othello, state, depth: int) -> int:
    """Count the number of positions reachable in exactly `depth` plies.

    A forced pass counts as one ply, matching how the reference Othello
    perft figures used in test_perft_8x8 are defined.
    """
    if depth == 0 or game.is_terminal(state):
        return 1
    moves = game.legal_move_squares(state)
    if not moves:
        return perft(game, game.apply_pass(state), depth - 1)
    return sum(perft(game, game.apply_move(state, sq), depth - 1) for sq in moves)


def play_random_game(game: Othello, rng: random.Random):
    """Play one full game with uniformly-random legal moves. Returns the
    final state and the number of plies played (including passes).
    """
    state = game.initial_state()
    plies = 0
    while not game.is_terminal(state):
        moves = game.legal_move_squares(state)
        if moves:
            state = game.apply_move(state, rng.choice(moves))
        else:
            state = game.apply_pass(state)
        plies += 1
        assert plies < 10_000, "game did not terminate (infinite loop?)"
    return state, plies


# --------------------------------------------------------------------------- #
# Initial position
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [6, 8])
def test_initial_position_has_four_discs_and_four_moves(n):
    game = Othello(n=n)
    state = game.initial_state()
    me, opp, passed_last = state
    assert game.popcount(me) == 2
    assert game.popcount(opp) == 2
    assert passed_last is False
    assert len(game.legal_move_squares(state)) == 4


def test_initial_position_layout_6x6():
    # Centre 2x2 block, diagonally opposed pairs of discs, "me" (Black)
    # to move.
    game = Othello(n=6)
    state = game.initial_state()
    assert game.render(state) == (
        ". . . . . .\n"
        ". . . . . .\n"
        ". . O X . .\n"
        ". . X O . .\n"
        ". . . . . .\n"
        ". . . . . ."
    )


# --------------------------------------------------------------------------- #
# Move legality and disc counts
# --------------------------------------------------------------------------- #


def test_applying_a_move_updates_disc_counts_consistently():
    game = Othello(n=6)
    state = game.initial_state()
    sq = game.legal_move_squares(state)[0]

    before_me, before_opp, _ = state
    discs_before = game.popcount(before_me) + game.popcount(before_opp)

    new_state = game.apply_move(state, sq)
    new_me, new_opp, passed_last = new_state

    # One new disc placed; total disc count grows by exactly one; no
    # disc simply disappears (every flip converts an opponent disc to
    # mine, it never removes one).
    assert game.popcount(new_me) + game.popcount(new_opp) == discs_before + 1
    assert passed_last is False

    # After the move it must be the other player's turn: what used to be
    # "opp" is now "me" state's "opp" side (i.e. players were swapped).
    # We can't compare bitboards directly across the swap (they refer to
    # different players), but disc totals per physical player must
    # still sum correctly: new_opp (mover's discs after swap) must
    # contain the disc just placed and any flips.
    assert (new_opp >> sq) & 1


def test_illegal_move_raises():
    game = Othello(n=6)
    state = game.initial_state()
    legal = set(game.legal_move_squares(state))
    illegal_sq = next(sq for sq in range(game.num_squares) if sq not in legal)
    with pytest.raises(ValueError):
        game.apply_move(state, illegal_sq)


def test_pass_only_legal_without_moves():
    game = Othello(n=6)
    state = game.initial_state()
    # There are legal moves in the initial position, so passing must fail.
    with pytest.raises(ValueError):
        game.apply_pass(state)


def test_square_out_of_range_raises():
    game = Othello(n=6)
    state = game.initial_state()
    with pytest.raises(ValueError):
        game.apply_move(state, game.num_squares)  # one past the last square
    with pytest.raises(ValueError):
        game.apply_move(state, -1)


# --------------------------------------------------------------------------- #
# Flips in each of the 8 directions, and no wrap-around
# --------------------------------------------------------------------------- #


def _state_from_ascii(game: Othello, ascii_board: str, mover: str = "X") -> tuple:
    """Build a state from an ASCII diagram like the one `render` prints.

    `mover` selects whether 'X' or 'O' is the player to move ("me").
    """
    rows = [line.split() for line in ascii_board.strip("\n").split("\n")]
    assert len(rows) == game.n and all(len(r) == game.n for r in rows)
    x_bb = o_bb = 0
    for row, cells in enumerate(rows):
        for col, ch in enumerate(cells):
            sq = row * game.n + col
            if ch == "X":
                x_bb |= 1 << sq
            elif ch == "O":
                o_bb |= 1 << sq
    if mover == "X":
        return (x_bb, o_bb, False)
    return (o_bb, x_bb, False)


def test_flip_horizontal_run():
    game = Othello(n=6)
    # Row 2: ". X O O . X" (cols 0..5). The only empty square adjacent to
    # the O-O run is col 4 (square 16); playing there sandwiches both O's
    # between the new disc and the X at col 1, so both must flip.
    board = (
        ". . . . . .\n"
        ". . . . . .\n"
        ". X O O . X\n"
        ". . . . . .\n"
        ". . . . . .\n"
        ". . . . . ."
    )
    state = _state_from_ascii(game, board, mover="X")
    assert game.legal_move_squares(state) == [16]
    flips = game._flips(state, 16)
    assert flips == (1 << 14) | (1 << 15)  # row2,col2 and row2,col3


def test_flip_all_eight_directions_from_a_single_move():
    game = Othello(n=6)
    # Place a ring of O's around the centre square (2,2)=sq14 in all 8
    # directions, each backed by an X two steps further out, then have X
    # play the centre square. Every O in the ring should flip.
    n = game.n
    center_row, center_col = 2, 2
    deltas = [
        (0, 1), (0, -1), (1, 0), (-1, 0),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    ]
    x_bb = 0
    o_bb = 0
    for dr, dc in deltas:
        or_, oc = center_row + dr, center_col + dc
        xr, xc = center_row + 2 * dr, center_col + 2 * dc
        assert 0 <= or_ < n and 0 <= oc < n
        assert 0 <= xr < n and 0 <= xc < n
        o_bb |= 1 << (or_ * n + oc)
        x_bb |= 1 << (xr * n + xc)
    state = (x_bb, o_bb, False)  # X ("me") to move
    center_sq = center_row * n + center_col
    assert (game.legal_moves(state) >> center_sq) & 1, "centre move should be legal"

    new_state = game.apply_move(state, center_sq)
    new_me, new_opp, _ = new_state  # new_me == old opponent's (O's) perspective now
    # From the *new* state's point of view, the player who just moved is
    # now "opp". All 8 ring discs plus the centre disc must belong to them.
    for dr, dc in deltas:
        or_, oc = center_row + dr, center_col + dc
        assert (new_opp >> (or_ * n + oc)) & 1, f"disc at ({or_},{oc}) should have flipped"
    assert (new_opp >> center_sq) & 1


def test_no_wraparound_east():
    game = Othello(n=6)
    # X at the end of row 0 (col 5), O's filling the rest of row 0 to its
    # left would only be a valid flip if there's an X further left; here
    # we specifically check that a disc in the last column of one row can
    # NOT combine with the first column of the next row.
    n = game.n
    # Row 0: O at col 5 (last column). Row 1: X at col 0 (first column).
    # These are adjacent in memory (sq 5 and sq 6) but NOT adjacent on the
    # board (different rows) — no move should treat them as a flip pair.
    o_bb = 1 << (0 * n + 5)
    x_bb = 1 << (1 * n + 0)
    state = (x_bb, o_bb, False)
    # "East" from X at (1,0): the immediate neighbour bitboard-wise would
    # be square 6+1=7 = (1,1), not square 5. So this lone O at (0,5) must
    # not be flippable by any move here, and in particular no bug should
    # make square 6 (row1,col0 + 1 naive) alias into row 0.
    legal = game.legal_move_squares(state)
    for sq in legal:
        flips = game._flips(state, sq)
        assert not (flips >> (0 * n + 5)) & 1, "wrap-around bug: flipped a disc across a row boundary"


def test_no_wraparound_west():
    game = Othello(n=6)
    n = game.n
    # O at col 0 of row 1, X at col 5 (last) of row 0. Bitboard-adjacent
    # (indices n-1 and n) but not board-adjacent.
    o_bb = 1 << (1 * n + 0)
    x_bb = 1 << (0 * n + 5)
    state = (x_bb, o_bb, False)
    legal = game.legal_move_squares(state)
    for sq in legal:
        flips = game._flips(state, sq)
        assert not (flips >> (1 * n + 0)) & 1, "wrap-around bug: flipped a disc across a row boundary"


# --------------------------------------------------------------------------- #
# Passing and termination
# --------------------------------------------------------------------------- #


def test_two_consecutive_passes_end_the_game():
    game = Othello(n=6)
    # A tiny hand-built position where neither side has any legal move:
    # isolated discs far enough apart that no flip is possible, board not
    # full.
    n = game.n
    x_bb = 1 << (0 * n + 0)
    o_bb = 1 << (5 * n + 5)
    state = (x_bb, o_bb, False)
    assert game.legal_move_squares(state) == []
    assert not game.is_terminal(state), "no legal move yet, but only one pass so far"

    state = game.apply_pass(state)
    assert state[2] is True  # passed_last
    assert game.legal_move_squares(state) == [], "opponent also has no moves"
    assert game.is_terminal(state), "two consecutive passes must end the game"


def test_full_board_is_terminal():
    game = Othello(n=4)
    # Fill every square (values/ownership don't matter for this check).
    full = (1 << game.num_squares) - 1
    half = full & 0x5555  # arbitrary split, doesn't need to be realistic
    state = (half, full & ~half, False)
    assert (state[0] | state[1]) == game.board_mask
    assert game.is_terminal(state)


# --------------------------------------------------------------------------- #
# Perft: the strongest correctness check
# --------------------------------------------------------------------------- #


def test_perft_8x8_matches_known_reference_values():
    """Depths 1-7 from the standard 8x8 Othello opening are well-known
    reference values. Matching them is strong evidence that legal-move
    generation and flipping are both correct across all 8 directions.
    """
    game = Othello(n=8)
    state = game.initial_state()
    expected = {1: 4, 2: 12, 3: 56, 4: 244, 5: 1396, 6: 8200, 7: 55092}
    for depth, expected_count in expected.items():
        assert perft(game, state, depth) == expected_count, f"perft({depth}) mismatch"


def test_perft_6x6_is_internally_consistent():
    """No external reference table for 6x6 is used here; this test instead
    locks in the engine's own output as a regression baseline, and checks
    the sanity property that perft(d) strictly grows with d in this range
    (true until the board starts filling up).
    """
    game = Othello(n=6)
    state = game.initial_state()
    counts = [perft(game, state, d) for d in range(1, 6)]
    assert counts == [4, 12, 56, 244, 1364]
    assert all(b > a for a, b in zip(counts, counts[1:]))


# --------------------------------------------------------------------------- #
# Full random games: stress test
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [6, 8])
def test_many_random_games_terminate_and_balance(n):
    game = Othello(n=n)
    rng = random.Random(42)
    for _ in range(200):
        state, plies = play_random_game(game, rng)
        assert game.is_terminal(state)
        me, opp, _ = state
        # No square is ever claimed by both players.
        assert (me & opp) == 0
        # Every disc on the board belongs to exactly one player and the
        # board never exceeds its own capacity.
        assert game.popcount(me | opp) <= game.num_squares
        # The game must have produced a well-defined result.
        assert game.result(state) in (-1, 0, 1)

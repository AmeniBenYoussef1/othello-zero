"""Othello (Reversi) game engine, implemented with bitboards.

Design notes
------------
- The board is N x N (N even, N >= 4). Squares are numbered 0..N*N-1 in
  row-major order: square = row * N + col.
- A "bitboard" is a plain Python int used as a bitmask over the N*N squares.
  Bit `sq` of a bitboard is set iff there is a disc on that square.
- A game state is the tuple (me, opp, passed_last):
    * me:  bitboard of the discs belonging to the player who is about to move
    * opp: bitboard of the opponent's discs
    * passed_last: True if the *previous* ply was a forced pass
  Storing the state from the mover's point of view (instead of "black" /
  "white") means the neural network we add later always sees the same
  "me vs. opponent" framing, and the win/loss sign never needs to track
  which physical colour is which.
- apply_move / apply_pass both return a new state with (me, opp) *swapped*,
  because after a ply it is the other player's turn.

The core trick for legal-move generation and disc-flipping is the classic
"dumb7fill" bitboard technique: for each of the 8 directions, repeatedly
slide across a run of opponent discs, then check what lies just beyond it.
"""

from dataclasses import dataclass

# A state is (me: int, opp: int, passed_last: bool). Kept as a plain tuple
# (not a dataclass) because states are created by the millions during
# self-play and tuples are the cheapest thing Python has.
State = tuple


@dataclass(frozen=True)
class Move:
    """A move is either a square (0..N*N-1) or a pass, represented as -1."""

    square: int

    PASS = -1

    def __repr__(self) -> str:
        return "Move(PASS)" if self.square == Move.PASS else f"Move({self.square})"


class Othello:
    """Rules engine for N x N Othello. Stateless: all state lives in the
    (me, opp, passed_last) tuples passed into each method.
    """

    def __init__(self, n: int = 6):
        if n < 4 or n % 2 != 0:
            raise ValueError("board size must be even and >= 4")
        self.n = n
        self.num_squares = n * n
        self.board_mask = (1 << self.num_squares) - 1
        self.file_a = self._file_mask(0)  # leftmost column
        self.file_last = self._file_mask(n - 1)  # rightmost column

        # One "slide" function per direction. Each takes a bitboard and
        # returns that bitboard shifted one step in the given direction,
        # with squares that would wrap around an edge cleared to 0.
        self._directions = [
            self._make_slider(shift=+1, pre_mask=~self.file_last),       # East
            self._make_slider(shift=-1, pre_mask=~self.file_a),          # West
            self._make_slider(shift=+n, pre_mask=~0),                    # South
            self._make_slider(shift=-n, pre_mask=~0),                    # North
            self._make_slider(shift=+(n + 1), pre_mask=~self.file_last), # South-East
            self._make_slider(shift=+(n - 1), pre_mask=~self.file_a),    # South-West
            self._make_slider(shift=-(n - 1), pre_mask=~self.file_last), # North-East
            self._make_slider(shift=-(n + 1), pre_mask=~self.file_a),    # North-West
        ]

    # ------------------------------------------------------------------ #
    # Bit-twiddling helpers
    # ------------------------------------------------------------------ #

    def _file_mask(self, col: int) -> int:
        """Bitmask with every square in the given column set."""
        mask = 0
        for row in range(self.n):
            mask |= 1 << (row * self.n + col)
        return mask

    def _make_slider(self, shift: int, pre_mask: int):
        """Build a function bb -> bb shifted by `shift` squares in the
        board's index space, masking out any bits that would otherwise
        wrap around a board edge.

        `pre_mask` clears, *before* shifting, the squares that must not
        participate (e.g. the rightmost column before shifting "east",
        since those discs have nowhere valid to slide to).
        """
        board_mask = self.board_mask
        pre_mask &= board_mask
        if shift >= 0:
            def slide(bb: int) -> int:
                return ((bb & pre_mask) << shift) & board_mask
        else:
            s = -shift

            def slide(bb: int) -> int:
                return ((bb & pre_mask) >> s) & board_mask

        return slide

    @staticmethod
    def popcount(bb: int) -> int:
        return bin(bb).count("1")

    # ------------------------------------------------------------------ #
    # Game state
    # ------------------------------------------------------------------ #

    def initial_state(self) -> State:
        """Standard Othello opening: two diagonally-opposed pairs of discs
        in the four centre squares. The player to move ("me") is Black,
        matching standard Othello, where Black always plays first.
        """
        n = self.n
        c0, c1 = n // 2 - 1, n // 2
        r0, r1 = n // 2 - 1, n // 2
        black = (1 << (r0 * n + c1)) | (1 << (r1 * n + c0))
        white = (1 << (r0 * n + c0)) | (1 << (r1 * n + c1))
        return (black, white, False)

    def legal_moves(self, state: State) -> int:
        """Bitboard of every square `me` may legally play on."""
        me, opp, _ = state
        empty = self.board_mask & ~(me | opp)
        legal = 0
        for slide in self._directions:
            # Opponent discs directly adjacent to one of my discs...
            run = slide(me) & opp
            # ...extended through the rest of a contiguous opponent run.
            for _ in range(self.n - 2):
                run |= slide(run) & opp
            # An empty square just past such a run is a legal move: playing
            # there would sandwich the whole run between two of my discs.
            legal |= slide(run) & empty
        return legal

    def legal_move_squares(self, state: State) -> list:
        """Legal moves as a sorted list of square indices (empty if none)."""
        moves = self.legal_moves(state)
        squares = []
        sq = 0
        while moves:
            if moves & 1:
                squares.append(sq)
            moves >>= 1
            sq += 1
        return squares

    def _flips(self, state: State, sq: int) -> int:
        """Bitboard of every opponent disc that playing at `sq` would flip."""
        me, opp, _ = state
        new_disc = 1 << sq
        flips = 0
        for slide in self._directions:
            run = slide(new_disc) & opp
            for _ in range(self.n - 2):
                run |= slide(run) & opp
            # The run only gets flipped if it is capped by one of my discs
            # (otherwise it's a run of opponent discs running off the edge
            # or into an empty square, which flips nothing).
            if slide(run) & me:
                flips |= run
        return flips

    def count_flips(self, state: State, sq: int) -> int:
        return self.popcount(self._flips(state, sq))

    def apply_move(self, state: State, sq: int) -> State:
        """Play at `sq` (must be a legal move) and return the resulting
        state, with (me, opp) swapped since it is now the other player's
        turn.
        """
        me, opp, _ = state
        if not (0 <= sq < self.num_squares):
            raise ValueError(f"square out of range: {sq}")
        if not (self.legal_moves(state) >> sq) & 1:
            raise ValueError(f"illegal move: {sq}")
        flips = self._flips(state, sq)
        assert flips, "a legal move must flip at least one disc"
        new_me = me | flips | (1 << sq)
        new_opp = opp & ~flips
        return (new_opp, new_me, False)

    def apply_pass(self, state: State) -> State:
        """Pass turn (only legal when there is no legal move)."""
        me, opp, _ = state
        if self.legal_moves(state) != 0:
            raise ValueError("cannot pass: a legal move exists")
        return (opp, me, True)

    def is_terminal(self, state: State) -> bool:
        me, opp, passed_last = state
        if (me | opp) == self.board_mask:
            return True  # board full
        if passed_last and self.legal_moves(state) == 0:
            return True  # two consecutive passes
        return False

    def result(self, state: State) -> int:
        """Outcome for the player to move: +1 win, -1 loss, 0 draw.

        Only meaningful once `is_terminal(state)` is True.
        """
        me, opp, _ = state
        me_count = self.popcount(me)
        opp_count = self.popcount(opp)
        if me_count > opp_count:
            return 1
        if me_count < opp_count:
            return -1
        return 0

    # ------------------------------------------------------------------ #
    # Debug / display helpers
    # ------------------------------------------------------------------ #

    def render(self, state: State) -> str:
        """ASCII board. 'X' = player to move, 'O' = opponent, '.' = empty."""
        me, opp, _ = state
        n = self.n
        rows = []
        for row in range(n):
            cells = []
            for col in range(n):
                sq = row * n + col
                if (me >> sq) & 1:
                    cells.append("X")
                elif (opp >> sq) & 1:
                    cells.append("O")
                else:
                    cells.append(".")
            rows.append(" ".join(cells))
        return "\n".join(rows)

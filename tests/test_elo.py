"""Tests for the Elo fitting and bootstrap confidence interval used to
evaluate agents. No torch needed: uses only the reference agents.
"""

from engine.othello import Othello
from agents import RandomAgent, GreedyAgent, MinimaxAgent
from eval.elo import elo_diff_from_winrate, bootstrap_winrate_ci, fit_bradley_terry
from eval.round_robin import run_round_robin


def test_elo_diff_zero_at_fifty_percent():
    assert abs(elo_diff_from_winrate(0.5)) < 1e-9


def test_elo_diff_positive_above_fifty_percent():
    assert elo_diff_from_winrate(0.75) > 0
    assert elo_diff_from_winrate(0.25) < 0


def test_bootstrap_ci_contains_observed_winrate():
    lo, hi = bootstrap_winrate_ci(n_wins=70, n_losses=30, n_draws=0, seed=0)
    assert lo <= 0.7 <= hi
    assert lo < hi


def test_bootstrap_ci_narrows_with_more_games():
    lo_small, hi_small = bootstrap_winrate_ci(7, 3, 0, seed=0)
    lo_big, hi_big = bootstrap_winrate_ci(700, 300, 0, seed=0)
    assert (hi_big - lo_big) < (hi_small - lo_small)


def test_bradley_terry_orders_a_dominant_player_first():
    # Total wins must actually differ (not just pairwise results) for a
    # strict Elo ordering: A crushes both B and C, and B also clearly
    # beats C, so A's and B's total win counts are unambiguously higher.
    pairwise = {
        ("A", "B"): {"wins_a": 90, "wins_b": 10, "draws": 0},
        ("A", "C"): {"wins_a": 95, "wins_b": 5, "draws": 0},
        ("B", "C"): {"wins_a": 70, "wins_b": 30, "draws": 0},
    }
    elo = fit_bradley_terry(pairwise)
    assert elo["A"] > elo["B"] > elo["C"]


def test_round_robin_produces_a_sorted_table():
    game = Othello(n=6)
    agents = {
        "Random": RandomAgent(seed=1),
        "Greedy": GreedyAgent(),
        "Minimax": MinimaxAgent(game, depth=2),
    }
    table, pairwise = run_round_robin(game, agents, n_games=6, seed=0)
    assert len(table) == 3
    elos = [row["elo"] for row in table]
    assert elos == sorted(elos, reverse=True)
    assert len(pairwise) == 3  # 3 choose 2

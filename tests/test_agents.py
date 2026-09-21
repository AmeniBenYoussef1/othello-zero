import random

from engine.othello import Othello
from agents import RandomAgent, GreedyAgent, PositionalAgent, MinimaxAgent, MCTSAgent


def _sample_states(game, n_states, seed):
    rng = random.Random(seed)
    states = []
    state = game.initial_state()
    while len(states) < n_states:
        if game.is_terminal(state):
            state = game.initial_state()
            continue
        moves = game.legal_move_squares(state)
        if moves:
            if game.legal_moves(state):
                states.append(state)
            state = game.apply_move(state, rng.choice(moves))
        else:
            state = game.apply_pass(state)
    return states


def _assert_agent_always_legal(agent, game, states):
    for state in states:
        sq = agent.choose_move(game, state)
        assert (game.legal_moves(state) >> sq) & 1, f"{agent} chose illegal move {sq}"


def test_random_agent_always_legal():
    game = Othello(n=6)
    states = _sample_states(game, 30, seed=1)
    _assert_agent_always_legal(RandomAgent(seed=1), game, states)


def test_greedy_agent_always_legal():
    game = Othello(n=6)
    states = _sample_states(game, 30, seed=2)
    _assert_agent_always_legal(GreedyAgent(), game, states)


def test_positional_agent_always_legal():
    game = Othello(n=6)
    states = _sample_states(game, 30, seed=3)
    _assert_agent_always_legal(PositionalAgent(game), game, states)


def test_minimax_agent_always_legal():
    game = Othello(n=6)
    states = _sample_states(game, 10, seed=4)
    _assert_agent_always_legal(MinimaxAgent(game, depth=2), game, states)


def test_mcts_agent_always_legal():
    game = Othello(n=6)
    states = _sample_states(game, 10, seed=5)
    _assert_agent_always_legal(MCTSAgent(game, n_simulations=30, seed=5), game, states)


def test_greedy_maximizes_flip_count():
    game = Othello(n=6)
    states = _sample_states(game, 20, seed=6)
    agent = GreedyAgent()
    for state in states:
        sq = agent.choose_move(game, state)
        best_possible = max(
            game.count_flips(state, m) for m in game.legal_move_squares(state)
        )
        assert game.count_flips(state, sq) == best_possible

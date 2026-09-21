import math


def play_game(game, first_agent, second_agent):
    state = game.initial_state()
    agents = (first_agent, second_agent)
    turn = 0
    while not game.is_terminal(state):
        moves = game.legal_move_squares(state)
        if moves:
            sq = agents[turn].choose_move(game, state)
            state = game.apply_move(state, sq)
        else:
            state = game.apply_pass(state)
        turn = 1 - turn
    return state


def play_match(game, agent_a, agent_b, n_games=100):
    wins_a = wins_b = draws = 0
    for i in range(n_games):
        a_plays_first = i % 2 == 0
        first, second = (agent_a, agent_b) if a_plays_first else (agent_b, agent_a)
        final_state = play_game(game, first, second)
        me, opp, _ = final_state
        first_discs, second_discs = game.popcount(me), game.popcount(opp)
        a_discs, b_discs = (
            (first_discs, second_discs) if a_plays_first else (second_discs, first_discs)
        )
        if a_discs == b_discs:
            draws += 1
        elif a_discs > b_discs:
            wins_a += 1
        else:
            wins_b += 1
    return {"wins_a": wins_a, "wins_b": wins_b, "draws": draws, "n_games": n_games}


def elo_diff_from_winrate(win_rate, eps=1e-6):
    p = min(max(win_rate, eps), 1 - eps)
    return -400 * math.log10(1 / p - 1)

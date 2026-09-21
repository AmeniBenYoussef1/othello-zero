import numpy as np

from model.encode import encode_state


def play_self_play_game(game, agent, temperature_moves=10, temperature=1.0, add_noise=True):
    state = game.initial_state()
    recorded = []
    ply = 0
    while not game.is_terminal(state):
        moves = game.legal_move_squares(state)
        if not moves:
            state = game.apply_pass(state)
            ply += 1
            continue

        root = agent.search(state, add_noise=add_noise)
        visit_counts = {move: child.visits for move, child in root.children.items()}
        total_visits = sum(visit_counts.values())

        policy_target = np.zeros(game.num_squares + 1, dtype=np.float32)
        for move, visits in visit_counts.items():
            policy_target[move] = visits / total_visits

        temp = temperature if ply < temperature_moves else 0.0
        if temp > 0:
            move_list = list(visit_counts.keys())
            weights = np.array([v ** (1.0 / temp) for v in visit_counts.values()], dtype=np.float64)
            weights /= weights.sum()
            move = move_list[np.random.choice(len(move_list), p=weights)]
        else:
            move = max(visit_counts, key=visit_counts.get)

        recorded.append((encode_state(game, state), policy_target, ply))
        state = game.apply_move(state, move)
        ply += 1

    final_result = game.result(state)
    final_parity = ply % 2
    examples = []
    for planes, policy_target, move_ply in recorded:
        z = final_result if (move_ply % 2) == final_parity else -final_result
        examples.append((planes, policy_target, z))
    return examples

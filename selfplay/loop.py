import copy

from agents.neural_mcts_agent import NeuralMCTSAgent
from eval.tournament import play_match
from model.train import train_on_buffer
from selfplay.self_play import play_self_play_game


def run_generation(
    game,
    current_net,
    buffer,
    games_per_generation=20,
    n_simulations=100,
    temperature_moves=10,
    train_steps=200,
    batch_size=64,
    lr=1e-3,
    eval_games=40,
    eval_simulations=100,
    promotion_threshold=0.55,
    device="cpu",
    use_augmentation=True,
    add_root_noise=True,
):
    agent = NeuralMCTSAgent(game, current_net, n_simulations=n_simulations, device=device)
    for _ in range(games_per_generation):
        examples = play_self_play_game(
            game, agent, temperature_moves=temperature_moves, add_noise=add_root_noise
        )
        buffer.add_game(examples, n=game.n, augment=use_augmentation)

    challenger = copy.deepcopy(current_net)
    history = train_on_buffer(
        challenger, buffer, n_steps=train_steps, batch_size=batch_size, lr=lr, device=device
    )

    challenger_agent = NeuralMCTSAgent(game, challenger, n_simulations=eval_simulations, device=device)
    current_agent = NeuralMCTSAgent(game, current_net, n_simulations=eval_simulations, device=device)
    result = play_match(game, challenger_agent, current_agent, n_games=eval_games)

    decided = result["wins_a"] + result["wins_b"]
    win_rate = result["wins_a"] / decided if decided else 0.5
    promoted = win_rate >= promotion_threshold

    return {
        "network": challenger if promoted else current_net,
        "promoted": promoted,
        "win_rate": win_rate,
        "match_result": result,
        "loss_history": history,
    }

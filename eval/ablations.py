from model.network import OthelloNet
from selfplay.replay_buffer import ReplayBuffer
from selfplay.loop import run_generation
from agents.neural_mcts_agent import NeuralMCTSAgent
from eval.round_robin import run_round_robin

DEFAULT_CONFIG = {
    "channels": 64,
    "n_blocks": 5,
    "n_simulations": 100,
    "eval_simulations": 100,
    "games_per_generation": 20,
    "temperature_moves": 10,
    "train_steps": 200,
    "batch_size": 64,
    "lr": 1e-3,
    "eval_games": 40,
    "promotion_threshold": 0.55,
    "use_augmentation": True,
    "add_root_noise": True,
    "c_puct": 1.5,
}


def merge_config(overrides):
    config = dict(DEFAULT_CONFIG)
    config.update(overrides)
    return config


def train_config(game, config, n_generations=1, device="cpu"):
    network = OthelloNet(n=game.n, channels=config["channels"], n_blocks=config["n_blocks"])
    buffer = ReplayBuffer()
    history = []
    for _ in range(n_generations):
        result = run_generation(
            game,
            network,
            buffer,
            games_per_generation=config["games_per_generation"],
            n_simulations=config["n_simulations"],
            temperature_moves=config["temperature_moves"],
            train_steps=config["train_steps"],
            batch_size=config["batch_size"],
            lr=config["lr"],
            eval_games=config["eval_games"],
            eval_simulations=config["eval_simulations"],
            promotion_threshold=config["promotion_threshold"],
            device=device,
            use_augmentation=config["use_augmentation"],
            add_root_noise=config["add_root_noise"],
        )
        network = result["network"]
        history.append(result)
    return network, history


def make_agent(game, network, config, device="cpu"):
    return NeuralMCTSAgent(
        game,
        network,
        n_simulations=config["eval_simulations"],
        c_puct=config["c_puct"],
        device=device,
    )


def run_ablation_study(game, variations, n_generations=1, n_eval_games=40, device="cpu", seed=None):
    agents_by_name = {}
    train_histories = {}
    for name, overrides in variations.items():
        config = merge_config(overrides)
        network, history = train_config(game, config, n_generations=n_generations, device=device)
        agents_by_name[name] = make_agent(game, network, config, device=device)
        train_histories[name] = history

    table, pairwise = run_round_robin(game, agents_by_name, n_games=n_eval_games, seed=seed)
    return table, pairwise, train_histories

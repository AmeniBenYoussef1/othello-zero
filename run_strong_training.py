import os
import time

import torch

from engine.othello import Othello
from model.network import OthelloNet
from model.checkpoint import save_checkpoint, load_checkpoint
from selfplay.replay_buffer import ReplayBuffer
from selfplay.loop import run_generation

CHECKPOINT_PATH = "checkpoint_strong.pt"
LOG_PATH = "training_log.txt"

CONFIG = {
    "channels": 64,
    "n_blocks": 5,
    "games_per_generation": 40,
    "n_simulations": 200,
    "temperature_moves": 12,
    "train_steps": 300,
    "batch_size": 128,
    "lr": 5e-4,
    "eval_games": 40,
    "eval_simulations": 150,
    "promotion_threshold": 0.55,
}

N_GENERATIONS = 50


def log(message):
    print(message, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(message + "\n")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device: {device}")

    game = Othello(n=6)
    buffer = ReplayBuffer()

    if os.path.exists(CHECKPOINT_PATH):
        network, payload = load_checkpoint(CHECKPOINT_PATH, device=device)
        start_gen = payload.get("generation", 0) + 1
        log(f"resuming from generation {start_gen}")
    else:
        network = OthelloNet(n=6, channels=CONFIG["channels"], n_blocks=CONFIG["n_blocks"])
        start_gen = 1

    network = network.to(device)

    for gen in range(start_gen, N_GENERATIONS + 1):
        start = time.time()
        result = run_generation(
            game,
            network,
            buffer,
            games_per_generation=CONFIG["games_per_generation"],
            n_simulations=CONFIG["n_simulations"],
            temperature_moves=CONFIG["temperature_moves"],
            train_steps=CONFIG["train_steps"],
            batch_size=CONFIG["batch_size"],
            lr=CONFIG["lr"],
            eval_games=CONFIG["eval_games"],
            eval_simulations=CONFIG["eval_simulations"],
            promotion_threshold=CONFIG["promotion_threshold"],
            device=device,
        )
        network = result["network"]
        elapsed = time.time() - start
        last_losses = result["loss_history"][-1]
        log(
            f"gen {gen}/{N_GENERATIONS} | {elapsed:.0f}s | "
            f"buffer={len(buffer)} | win_rate={result['win_rate']:.2f} | "
            f"promoted={result['promoted']} | "
            f"policy_loss={last_losses[0]:.3f} | value_loss={last_losses[1]:.3f}"
        )
        save_checkpoint(
            network,
            CHECKPOINT_PATH,
            channels=CONFIG["channels"],
            n_blocks=CONFIG["n_blocks"],
            extra={"generation": gen},
        )

    log("training finished")


if __name__ == "__main__":
    main()
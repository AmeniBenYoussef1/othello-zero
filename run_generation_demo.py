import time

from engine.othello import Othello
from model.network import OthelloNet
from model.checkpoint import save_checkpoint
from selfplay.replay_buffer import ReplayBuffer
from selfplay.loop import run_generation

game = Othello(n=6)
network = OthelloNet(n=6, channels=32, n_blocks=3)
buffer = ReplayBuffer()

start = time.time()
result = run_generation(
    game,
    network,
    buffer,
    games_per_generation=10,
    n_simulations=50,
    temperature_moves=6,
    train_steps=100,
    batch_size=32,
    lr=1e-3,
    eval_games=10,
    eval_simulations=50,
    promotion_threshold=0.55,
)
elapsed = time.time() - start

print(f"generation finished in {elapsed:.1f}s")
print(f"replay buffer size: {len(buffer)}")
print(f"match result: {result['match_result']}")
print(f"win rate (challenger): {result['win_rate']:.2f}")
print(f"promoted: {result['promoted']}")

first_losses = result["loss_history"][0]
last_losses = result["loss_history"][-1]
print(f"loss at step 0:   policy={first_losses[0]:.4f}  value={first_losses[1]:.4f}")
print(f"loss at step {len(result['loss_history']) - 1}: policy={last_losses[0]:.4f}  value={last_losses[1]:.4f}")

save_checkpoint(result["network"], "checkpoint_gen1.pt", channels=32, n_blocks=3)
print("saved checkpoint_gen1.pt")

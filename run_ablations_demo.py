from engine.othello import Othello
from eval.ablations import run_ablation_study
from eval.round_robin import print_table

game = Othello(n=6)

variations = {
    "baseline": {},
    "few_simulations": {"n_simulations": 20, "eval_simulations": 20},
    "small_net": {"channels": 16, "n_blocks": 2},
    "no_augmentation": {"use_augmentation": False},
    "no_root_noise": {"add_root_noise": False},
}

table, pairwise, histories = run_ablation_study(
    game,
    variations,
    n_generations=1,
    n_eval_games=10,
    seed=0,
)

print_table(table)

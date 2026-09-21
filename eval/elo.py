import math
import random
from collections import defaultdict


def elo_diff_from_winrate(win_rate, eps=1e-6):
    p = min(max(win_rate, eps), 1 - eps)
    return 400 * math.log10(p / (1 - p))


def bootstrap_winrate_ci(n_wins, n_losses, n_draws, n_resamples=2000, ci=0.95, seed=None):
    rng = random.Random(seed)
    outcomes = [1.0] * n_wins + [0.0] * n_losses + [0.5] * n_draws
    n = len(outcomes)
    if n == 0:
        return (float("nan"), float("nan"))
    samples = []
    for _ in range(n_resamples):
        resample_sum = sum(outcomes[rng.randrange(n)] for _ in range(n))
        samples.append(resample_sum / n)
    samples.sort()
    lo_idx = int((1 - ci) / 2 * n_resamples)
    hi_idx = int((1 + ci) / 2 * n_resamples) - 1
    return samples[lo_idx], samples[hi_idx]


def fit_bradley_terry(pairwise_results, n_iters=500, eps=1e-9):
    players = set()
    for a, b in pairwise_results:
        players.add(a)
        players.add(b)
    players = sorted(players)
    strength = {p: 1.0 for p in players}

    wins = defaultdict(float)
    games = {}
    for (a, b), res in pairwise_results.items():
        draws = res.get("draws", 0)
        wa = res["wins_a"] + 0.5 * draws
        wb = res["wins_b"] + 0.5 * draws
        n = res["wins_a"] + res["wins_b"] + draws
        wins[a] += wa
        wins[b] += wb
        games[(a, b)] = n
        games[(b, a)] = n

    for _ in range(n_iters):
        new_strength = {}
        for p in players:
            denom = 0.0
            for q in players:
                if p == q:
                    continue
                n_pq = games.get((p, q), 0.0)
                if n_pq == 0:
                    continue
                denom += n_pq / (strength[p] + strength[q])
            new_strength[p] = wins[p] / denom if denom > eps else strength[p]
        norm = sum(new_strength.values()) / len(players)
        strength = {p: v / norm for p, v in new_strength.items()}

    elo = {p: 400 * math.log10(max(s, eps)) for p, s in strength.items()}
    baseline = elo[players[0]]
    return {p: v - baseline for p, v in elo.items()}

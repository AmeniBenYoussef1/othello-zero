from itertools import combinations

from eval.tournament import play_match
from eval.elo import fit_bradley_terry, bootstrap_winrate_ci


def run_round_robin(game, agents_by_name, n_games=100, seed=None):
    pairwise_results = {}
    names = list(agents_by_name.keys())
    for name_a, name_b in combinations(names, 2):
        result = play_match(game, agents_by_name[name_a], agents_by_name[name_b], n_games=n_games)
        pairwise_results[(name_a, name_b)] = result

    ratings = fit_bradley_terry(pairwise_results)

    aggregate = {name: {"wins": 0, "losses": 0, "draws": 0} for name in names}
    for (name_a, name_b), result in pairwise_results.items():
        aggregate[name_a]["wins"] += result["wins_a"]
        aggregate[name_a]["losses"] += result["wins_b"]
        aggregate[name_a]["draws"] += result["draws"]
        aggregate[name_b]["wins"] += result["wins_b"]
        aggregate[name_b]["losses"] += result["wins_a"]
        aggregate[name_b]["draws"] += result["draws"]

    table = []
    for name in names:
        agg = aggregate[name]
        lo, hi = bootstrap_winrate_ci(agg["wins"], agg["losses"], agg["draws"], seed=seed)
        table.append({
            "name": name,
            "elo": ratings[name],
            "wins": agg["wins"],
            "losses": agg["losses"],
            "draws": agg["draws"],
            "winrate_ci": (lo, hi),
        })
    table.sort(key=lambda row: row["elo"], reverse=True)
    return table, pairwise_results


def print_table(table):
    print(f"{'agent':<14}{'elo':>8}{'W-L-D':>14}{'winrate 95% CI':>20}")
    for row in table:
        wld = f"{row['wins']}-{row['losses']}-{row['draws']}"
        lo, hi = row["winrate_ci"]
        ci_str = f"[{lo:.2f}, {hi:.2f}]"
        print(f"{row['name']:<14}{row['elo']:>8.0f}{wld:>14}{ci_str:>20}")

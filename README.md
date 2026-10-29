# OthelloZero

Agent Othello par auto-apprentissage : Monte-Carlo Tree Search guidée par
un réseau de neurones résiduel, entraîné par auto-jeu (self-play), à la
manière d'AlphaZero.


## Fonctionnalités

- **Moteur de jeu** en bitboards, paramétrable en taille de plateau (6x6
  par défaut, 8x8 supporté), validé par un perft conforme aux valeurs de
  référence connues.
- **Agents de référence** : aléatoire, glouton, positionnel, alpha-bêta,
  MCTS classique — pour évaluer et comparer l'agent neuronal.
- **Réseau de neurones résiduel** (PyTorch) avec tête de politique et tête
  de valeur, couplé à une recherche MCTS guidée par PUCT.
- **Boucle d'auto-apprentissage** : parties d'auto-jeu, tampon de rejeu
  avec augmentation par symétrie diédrale (8 transformations), entraînement,
  évaluation challenger vs courant, promotion automatique.
- **Évaluation Elo** : classement par régression Bradley-Terry (algorithme
  MM), intervalles de confiance par bootstrap, tournoi round-robin entre
  agents.
- **Étude d'ablation** quantifiant l'apport de chaque composant du pipeline
  (bruit d'exploration, augmentation de données, taille du réseau,
  profondeur de recherche MCTS).
- **Démonstrateur interactif** (Gradio) pour jouer contre l'IA, avec
  affichage en temps réel de la probabilité de victoire.

## Résultats

Comparaison round-robin + classement Elo de 5 configurations (plateau
6x6, 1 génération d'entraînement) :

| agent | elo | W-L-D | IC 95% winrate |
|---|---|---|---|
| baseline | 0 | 25-10-5 | [0.55, 0.82] |
| no_root_noise | -76 | 20-15-5 | [0.42, 0.71] |
| no_augmentation | -112 | 20-20-0 | [0.35, 0.65] |
| small_net | -185 | 10-20-10 | [0.25, 0.50] |
| few_simulations | -185 | 15-25-0 | [0.23, 0.53] |

Chaque composant du pipeline contribue positivement au niveau de jeu ;
réduire la recherche MCTS ou la taille du réseau est ce qui coûte le plus
cher en performance.

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

**Lancer les tests**

```bash
pytest tests/ -v
```

**Jouer contre l'IA**

```bash
python demo/app.py
```

Ouvre le lien local affiché dans le terminal. Utilise le réseau entraîné
présent dans `checkpoint_gen1.pt` (variable d'environnement
`OTHELLOZERO_CHECKPOINT` pour pointer vers un autre checkpoint), sinon
retombe automatiquement sur un agent alpha-bêta.

**Lancer un entraînement**

```bash
python run_generation_demo.py       # démo rapide, une génération
python run_strong_training.py       # entraînement long, plusieurs générations, reprise automatique
```

**Reproduire l'étude d'ablation**

```bash
python run_ablations_demo.py
```

## Structure du dépôt

```
engine/      moteur de jeu (bitboards)
agents/      agents de référence et agent MCTS neuronal
model/       réseau PyTorch, boucle d'entraînement
selfplay/    génération de parties, tampon de rejeu
eval/        tournois, classement Elo, ablations
demo/        démonstrateur interactif (Gradio)
tests/       tests unitaires
```
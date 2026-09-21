import numpy as np

from engine.othello import Othello
from model.network import OthelloNet
from agents.neural_mcts_agent import NeuralMCTSAgent
from selfplay.self_play import play_self_play_game
from selfplay.replay_buffer import ReplayBuffer, augment_example


def _tiny_agent(n=6, simulations=16):
    game = Othello(n=n)
    net = OthelloNet(n=n, channels=8, n_blocks=1)
    return game, NeuralMCTSAgent(game, net, n_simulations=simulations)


def test_self_play_game_produces_valid_examples():
    game, agent = _tiny_agent()
    examples = play_self_play_game(game, agent, temperature_moves=4)
    assert len(examples) > 0
    for planes, policy, z in examples:
        assert planes.shape == (3, game.n, game.n)
        assert policy.shape == (game.num_squares + 1,)
        assert abs(float(policy.sum()) - 1.0) < 1e-4
        assert z in (-1, 0, 1)


def test_replay_buffer_augmentation_multiplies_by_eight():
    game, agent = _tiny_agent()
    examples = play_self_play_game(game, agent, temperature_moves=4)
    buffer = ReplayBuffer()
    buffer.add_game(examples, n=game.n, augment=True)
    assert len(buffer) == len(examples) * 8


def test_augmentation_preserves_policy_mass():
    n = 6
    policy = np.zeros(n * n + 1, dtype=np.float32)
    policy[5] = 0.7
    policy[n * n] = 0.3  
    planes = np.random.rand(3, n, n).astype(np.float32)
    for aug_planes, aug_policy in augment_example(planes, policy, n):
        assert abs(float(aug_policy.sum()) - 1.0) < 1e-5
        assert aug_policy[n * n] == 0.3  

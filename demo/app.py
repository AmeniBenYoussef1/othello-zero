import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr

from engine.othello import Othello
from model.checkpoint import load_checkpoint
from agents.neural_mcts_agent import NeuralMCTSAgent
from agents.minimax_agent import MinimaxAgent

BOARD_SIZE = 6
CHECKPOINT_PATH = os.environ.get("OTHELLOZERO_CHECKPOINT", "checkpoint_gen1.pt")

BOARD_CSS = """
.score-card {
    text-align: center; padding: 14px; border-radius: 10px;
    font-size: 22px; font-weight: 700;
}
.score-black { background: #111111 !important; color: #f5f5f5 !important; border: 2px solid #2dd4bf; }
.score-white { background: #f5f5f5 !important; color: #111111 !important; border: 2px solid #d4d4d4; }
.status-bar {
    text-align: center !important; padding: 10px !important; border-radius: 8px !important;
    background: #1a1a1a !important; color: #eee !important; font-weight: 600 !important;
}
.board-row { gap: 0px !important; justify-content: center !important; }
.cell {
    min-width: 64px !important; width: 64px !important; height: 64px !important; max-width: 64px !important;
    font-size: 34px !important; padding: 0 !important; border-radius: 0 !important;
    background: #e8c547 !important; border: 1px solid #b89530 !important;
}
.cell-black {
    color: #111111 !important;
    text-shadow: -2px -2px 3px rgba(255,255,255,0.25), 2px 2px 4px rgba(0,0,0,0.6) !important;
}
.cell-white {
    color: #fdfdfd !important;
    text-shadow: -2px -2px 3px rgba(255,255,255,0.9), 2px 2px 4px rgba(0,0,0,0.4) !important;
}
.cell-legal {
    background: #f3dd84 !important; color: #a5842a !important; font-size: 16px !important;
}
.cell-legal:hover { background: #f7e6a0 !important; }
.cell-last-black { box-shadow: inset 0 0 0 4px #38bdf8 !important; }
.cell-last-white { box-shadow: inset 0 0 0 4px #c084fc !important; }
"""

game = Othello(n=BOARD_SIZE)

try:
    network, _ = load_checkpoint(CHECKPOINT_PATH)
    agent = NeuralMCTSAgent(game, network, n_simulations=200)
    HAS_NETWORK = True
except FileNotFoundError:
    agent = MinimaxAgent(game, depth=4)
    HAS_NETWORK = False


def split_colors(state, ply):
    a, b, _ = state
    return (a, b) if ply % 2 == 0 else (b, a)


def cell_content(state, ply, sq, last_black, last_white):
    black_bits, white_bits = split_colors(state, ply)
    classes = ["cell"]
    label = ""
    if (black_bits >> sq) & 1:
        label, classes = "⚫", classes + ["cell-black"]
    elif (white_bits >> sq) & 1:
        label, classes = "⚪", classes + ["cell-white"]
    elif sq in game.legal_move_squares(state):
        label, classes = "•", classes + ["cell-legal"]
    if sq == last_black:
        classes.append("cell-last-black")
    if sq == last_white:
        classes.append("cell-last-white")
    return label, classes


def board_updates(state, ply, last_black, last_white):
    updates = []
    for sq in range(game.num_squares):
        label, classes = cell_content(state, ply, sq, last_black, last_white)
        updates.append(gr.update(value=label, elem_classes=classes))
    return updates


def score_text(state, ply):
    black_bits, white_bits = split_colors(state, ply)
    return f"Noir\n{game.popcount(black_bits)}", f"Blanc\n{game.popcount(white_bits)}"


def win_probability(state):
    if not HAS_NETWORK or game.is_terminal(state):
        return None
    root = agent.search(state, add_noise=False)
    return (root.q + 1) / 2


def status_text(state, ply):
    if game.is_terminal(state):
        black_bits, white_bits = split_colors(state, ply)
        black_count = game.popcount(black_bits)
        white_count = game.popcount(white_bits)
        if black_count > white_count:
            return "Tu gagnes !"
        if black_count < white_count:
            return "L'IA gagne."
        return "Egalite."
    prob_human = win_probability(state)
    if prob_human is None:
        return "A toi de jouer (noir) -- clique une case en surbrillance."
    prob_ai = 1 - prob_human
    return f"A toi de jouer (noir). Probabilite de victoire IA : {prob_ai:.0%}"


def advance_passes(state, ply):
    while not game.is_terminal(state) and not game.legal_move_squares(state):
        state = game.apply_pass(state)
        ply += 1
    return state, ply


def ai_reply(state, ply):
    if game.is_terminal(state):
        return state, ply, None
    if HAS_NETWORK:
        move = agent.choose_move(game, state, temperature=0.0, add_noise=False)
    else:
        move = agent.choose_move(game, state)
    state = game.apply_move(state, move)
    ply += 1
    state, ply = advance_passes(state, ply)
    return state, ply, move


def play(sq, state, ply, last_black, last_white):
    if not game.is_terminal(state) and sq in game.legal_move_squares(state):
        state = game.apply_move(state, sq)
        ply += 1
        state, ply = advance_passes(state, ply)
        last_black = sq
        state, ply, ai_move = ai_reply(state, ply)
        if ai_move is not None:
            last_white = ai_move
    black_score, white_score = score_text(state, ply)
    return (
        state, ply, last_black, last_white,
        *board_updates(state, ply, last_black, last_white),
        status_text(state, ply), black_score, white_score,
    )


def new_game():
    state, ply = advance_passes(game.initial_state(), 0)
    last_black, last_white = None, None
    black_score, white_score = score_text(state, ply)
    return (
        state, ply, last_black, last_white,
        *board_updates(state, ply, last_black, last_white),
        status_text(state, ply), black_score, white_score,
    )


with gr.Blocks(title="OthelloZero", css=BOARD_CSS) as demo:
    gr.Markdown("# OthelloZero")
    initial_state, initial_ply = advance_passes(game.initial_state(), 0)
    state = gr.State(initial_state)
    ply_state = gr.State(initial_ply)
    last_black_state = gr.State(None)
    last_white_state = gr.State(None)

    init_black_score, init_white_score = score_text(initial_state, initial_ply)
    with gr.Row():
        black_score = gr.Markdown(init_black_score, elem_classes=["score-card", "score-black"])
        white_score = gr.Markdown(init_white_score, elem_classes=["score-card", "score-white"])

    status = gr.Markdown(status_text(initial_state, initial_ply), elem_classes=["status-bar"])

    buttons = []
    for r in range(BOARD_SIZE):
        with gr.Row(elem_classes=["board-row"]):
            for c in range(BOARD_SIZE):
                sq = r * BOARD_SIZE + c
                label, classes = cell_content(initial_state, initial_ply, sq, None, None)
                buttons.append(gr.Button(label, elem_classes=classes))

    reset_btn = gr.Button("Nouvelle partie")

    outputs = [
        state, ply_state, last_black_state, last_white_state,
        *buttons, status, black_score, white_score,
    ]
    inputs_common = [state, ply_state, last_black_state, last_white_state]
    for sq, btn in enumerate(buttons):
        btn.click(lambda s, p, lb, lw, sq=sq: play(sq, s, p, lb, lw), inputs=inputs_common, outputs=outputs)
    reset_btn.click(new_game, inputs=[], outputs=outputs)

if __name__ == "__main__":
    demo.launch()

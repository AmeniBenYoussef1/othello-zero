import torch

from model.network import OthelloNet


def save_checkpoint(network, path, channels, n_blocks, extra=None):
    payload = {
        "state_dict": network.state_dict(),
        "n": network.n,
        "channels": channels,
        "n_blocks": n_blocks,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location=device)
    network = OthelloNet(n=payload["n"], channels=payload["channels"], n_blocks=payload["n_blocks"])
    network.load_state_dict(payload["state_dict"])
    return network, payload

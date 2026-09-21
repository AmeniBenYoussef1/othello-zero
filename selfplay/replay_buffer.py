import random
from collections import deque

import numpy as np


def dihedral_transforms():
    def rot(a, k):
        return np.rot90(a, k, axes=(-2, -1))

    fns = []
    for k in range(4):
        fns.append(lambda a, k=k: rot(a, k))
    for k in range(4):
        fns.append(lambda a, k=k: rot(a, k)[..., ::-1])
    return fns


def augment_example(planes, policy, n):
    policy_board = policy[: n * n].reshape(n, n)
    pass_prob = policy[n * n]
    out = []
    for fn in dihedral_transforms():
        p = np.ascontiguousarray(fn(planes))
        pb = np.ascontiguousarray(fn(policy_board))
        new_policy = np.concatenate([pb.reshape(-1), [pass_prob]]).astype(np.float32)
        out.append((p, new_policy))
    return out


class ReplayBuffer:
    def __init__(self, capacity=200_000):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def add_game(self, examples, n, augment=True):
        for planes, policy, z in examples:
            if augment:
                for p, pol in augment_example(planes, policy, n):
                    self.buffer.append((p, pol, z))
            else:
                self.buffer.append((planes, policy, z))

    def sample(self, batch_size):
        batch = random.choices(self.buffer, k=batch_size)
        planes = np.stack([b[0] for b in batch])
        policies = np.stack([b[1] for b in batch])
        values = np.array([b[2] for b in batch], dtype=np.float32)
        return planes, policies, values

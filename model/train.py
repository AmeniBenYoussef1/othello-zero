import torch
import torch.nn.functional as F


def train_step(network, optimizer, batch, device="cpu"):
    planes, policies, values = batch
    x = torch.from_numpy(planes).to(device)
    target_policy = torch.from_numpy(policies).to(device)
    target_value = torch.from_numpy(values).to(device)

    network.train()
    logits, value_pred = network(x)
    log_probs = F.log_softmax(logits, dim=1)
    policy_loss = -(target_policy * log_probs).sum(dim=1).mean()
    value_loss = F.mse_loss(value_pred, target_value)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    network.eval()
    return float(policy_loss.item()), float(value_loss.item())


def train_on_buffer(network, buffer, n_steps=200, batch_size=64, lr=1e-3, device="cpu"):
    network.to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    history = []
    for _ in range(n_steps):
        batch = buffer.sample(batch_size)
        policy_loss, value_loss = train_step(network, optimizer, batch, device=device)
        history.append((policy_loss, value_loss))
    return history

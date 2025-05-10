import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x
    
## Step 1
def shard_data(batch, rank, world_size):
    total_samples = len(batch)
    samples_per_rank = total_samples // world_size
    start_idx = rank * samples_per_rank
    end_idx = start_idx + samples_per_rank
    return batch[start_idx:end_idx]


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def train_single_process(X, y):
    model = ToyModel(10, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    # one epoch, full‐batch
    optimizer.zero_grad()
    pred = model(X)
    loss = nn.MSELoss()(pred, y)
    loss.backward()
    optimizer.step()
    print(f"[Single] final loss = {loss.item():.6f}")

def ddp_training(rank, world_size, X, y, epochs=5):
    setup(rank, world_size)
    torch.manual_seed(42)
    model = ToyModel(10, 2)
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    x_shard = shard_data(X, rank, world_size)
    y_shard = shard_data(y, rank, world_size)
    for e in range(epochs):
        optimizer.zero_grad()
        pred = model(x_shard)
        loss = nn.MSELoss()(pred, y_shard)
        loss.backward()
        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size
        optimizer.step()
        print(f"[Rank {rank}] epoch {e} loss = {loss.item():.6f}")

if __name__ == "__main__":

    torch.manual_seed(42)
    X = torch.randn(64, 10)
    y = torch.randn(64, 2)
    train_single_process(X, y)

    world_size = 2
    mp.spawn(
        ddp_training,
        args=(world_size,X, y),
        nprocs=world_size,
        join=True,
    )
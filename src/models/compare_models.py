"""Train the 3D-CNN and the hybrid CNN+Transformer on the mock cube and compare."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from cnn3d import Small3DCNN
from hybrid import HybridCNNViT
from train_mock import make_patches

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def train_and_eval(model, x, y, train_idx, test_idx, epochs=5):
    # Stressed pixels are only 4%, so give them a higher weight
    weights = torch.tensor([1.0, (y == 0).sum().item() / (y == 1).sum().item()])
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        order = train_idx[torch.randperm(len(train_idx))]
        total = 0.0
        for k in range(0, len(order), 64):
            batch = order[k:k + 64]
            optimizer.zero_grad()
            loss = loss_fn(model(x[batch]), y[batch])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        print(f"  Epoch {epoch + 1}: loss {total / len(order):.4f}")

    model.eval()
    with torch.no_grad():
        pred = model(x[test_idx]).argmax(1)
    truth = y[test_idx]
    accuracy = (pred == truth).float().mean().item()
    recall = (pred[truth == 1] == 1).float().mean().item()
    return accuracy, recall


if __name__ == "__main__":
    cube = np.load(DATA_DIR / "mock_cube_pca.npy")
    labels = np.load(DATA_DIR / "mock_labels.npy")
    cube = (cube - cube.mean(axis=(0, 1))) / cube.std(axis=(0, 1))
    x, y = make_patches(cube, labels)

    # Same split for both models so the comparison is fair
    torch.manual_seed(42)
    perm = torch.randperm(len(y))
    split = int(0.8 * len(y))
    train_idx, test_idx = perm[:split], perm[split:]

    results = {}
    for name, build in [("3D-CNN", Small3DCNN), ("Hybrid CNN+Transformer", HybridCNNViT)]:
        print(name)
        torch.manual_seed(0)
        model = build()
        results[name] = train_and_eval(model, x, y, train_idx, test_idx)
        if name.startswith("Hybrid"):
            (PROJECT_ROOT / "models").mkdir(exist_ok=True)
            torch.save(model.state_dict(), PROJECT_ROOT / "models" / "hybrid_mock.pt")

    print("\nResults on mock data")
    for name, (acc, rec) in results.items():
        print(f"{name}: accuracy {acc:.3f}, stressed recall {rec:.3f}")
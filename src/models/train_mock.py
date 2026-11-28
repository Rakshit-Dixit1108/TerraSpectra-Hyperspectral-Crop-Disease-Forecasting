"""Train the 3D-CNN on the mock cube (a pipeline check, not a real result)."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from cnn3d import Small3DCNN

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
PATCH = 9


def make_patches(cube, labels):
    """One 9x9 patch around every pixel. Returns (N, 1, bands, 9, 9)."""
    half = PATCH // 2
    padded = np.pad(cube, ((half, half), (half, half), (0, 0)), mode="reflect")
    h, w, _ = cube.shape
    patches, ys = [], []
    for i in range(h):
        for j in range(w):
            p = padded[i:i + PATCH, j:j + PATCH, :]  # (9, 9, bands)
            patches.append(p.transpose(2, 0, 1))     # (bands, 9, 9)
            ys.append(labels[i, j])
    x = np.array(patches, dtype=np.float32)[:, None]
    return torch.from_numpy(x), torch.tensor(ys, dtype=torch.long)


if __name__ == "__main__":
    torch.manual_seed(42)
    cube = np.load(DATA_DIR / "mock_cube_pca.npy")
    labels = np.load(DATA_DIR / "mock_labels.npy")
    cube = (cube - cube.mean(axis=(0, 1))) / cube.std(axis=(0, 1))

    x, y = make_patches(cube, labels)

    perm = torch.randperm(len(y))
    split = int(0.8 * len(y))
    train_idx, test_idx = perm[:split], perm[split:]

    model = Small3DCNN()
    # Stressed pixels are only 4%, so give them a higher weight
    weights = torch.tensor([1.0, (y == 0).sum().item() / (y == 1).sum().item()])
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
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
        print(f"Epoch {epoch + 1}: loss {total / len(order):.4f}")

    model.eval()
    with torch.no_grad():
        pred = model(x[test_idx]).argmax(1)
    truth = y[test_idx]
    accuracy = (pred == truth).float().mean().item()
    stressed = truth == 1
    recall = (pred[stressed] == 1).float().mean().item()
    print(f"Test accuracy: {accuracy:.3f}")
    print(f"Stressed recall: {recall:.3f}")

    (PROJECT_ROOT / "models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), PROJECT_ROOT / "models" / "cnn3d_mock.pt")
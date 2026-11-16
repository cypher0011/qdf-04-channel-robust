"""Clean-trained vs channel-robust detector, evaluated under every channel.

    python scripts/02_train.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from channel_robust import data  # noqa: E402
from channel_robust.channels import CONDITIONS  # noqa: E402
from channel_robust.head import LayerWeightedHead  # noqa: E402
from channel_robust.metrics import summary  # noqa: E402

CACHE = data.DATA_ROOT.parent / "cache"
RES = ROOT / "results"
SEED = 0


def meta_of(name):
    return pd.DataFrame(json.loads(str(np.load(CACHE / f"{name}.npz")["meta"])))


def train_head(X, y, epochs=40, lr=1e-3, bs=128):
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X)); nv = len(X) // 10
    va, tr = perm[:nv], perm[nv:]
    head = LayerWeightedHead().set_norm(X[tr])
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-2)
    pw = torch.tensor((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1), dtype=torch.float32)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))
    best, state = 1.0, None
    for _ in range(epochs):
        head.train()
        order = rng.permutation(tr)
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            loss = F.binary_cross_entropy_with_logits(head(Xt[b] + 0.05 * torch.randn_like(Xt[b]) * head.sd[0]), yt[b], pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        v = summary(y[va], predict(head, X[va]))["eer"]
        if v <= best:
            best, state = v, {k: t.clone() for k, t in head.state_dict().items()}
    head.load_state_dict(state)
    return head


@torch.no_grad()
def predict(head, X):
    head.eval()
    return head(torch.from_numpy(X)).numpy()


def main():
    y_tr = meta_of("aradf_train")["is_fake"].values
    X_clean = np.load(CACHE / "ssl_aradf_train.npy").astype(np.float32)
    X_aug = np.load(CACHE / "ssl_aradf_train_aug.npy").astype(np.float32)

    models = {
        "clean_trained (= Model 3)": train_head(X_clean, y_tr),
        "channel_robust (Model 4)": train_head(np.concatenate([X_clean, X_aug]), np.r_[y_tr, y_tr]),
    }
    (RES / "models").mkdir(parents=True, exist_ok=True)
    torch.save(models["channel_robust (Model 4)"].state_dict(), RES / "models" / "head_channel_robust.pt")

    dev_meta = meta_of("aradf_dev")
    idx = json.load(open(CACHE / "aradf_dev_channel_subset.json"))
    y_dev = dev_meta["is_fake"].values[idx]
    metrics = {"per_condition": {}, "external": {}}
    for c in CONDITIONS:
        X = np.load(CACHE / f"ssl_aradf_dev_{c}.npy").astype(np.float32)
        metrics["per_condition"][c] = {k: summary(y_dev, predict(h, X)) for k, h in models.items()}
    for ext in ("aradf_track2", "arad"):
        p = CACHE / f"ssl_{ext}.npy"
        if p.exists():
            X, y = np.load(p).astype(np.float32), meta_of(ext)["is_fake"].values
            metrics["external"][ext] = {k: summary(y, predict(h, X)) for k, h in models.items()}
            for k, h in models.items():
                tag = "robust" if "robust" in k else "clean"
                pd.DataFrame({"is_fake": y, "score": predict(h, X)}).to_csv(RES / f"scores_{tag}_{ext}.csv", index=False)

    for c, d in {**metrics["per_condition"], **metrics["external"]}.items():
        print(f"{c:14s} " + "  ".join(f"{k.split(' ')[0]} {v['eer']*100:5.2f}%" for k, v in d.items()))
    for k in models:
        metrics[f"mean_eer_over_channels::{k}"] = float(np.mean([metrics["per_condition"][c][k]["eer"] for c in CONDITIONS]))
        metrics[f"worst_eer_over_channels::{k}"] = float(np.max([metrics["per_condition"][c][k]["eer"] for c in CONDITIONS]))
    json.dump(metrics, open(RES / "metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()

"""Degrade audio through simulated channels and extract frozen mHuBERT-147
embeddings for every version.

Training : every ArA-DF train utterance gets ONE random channel (25% clean)
Testing  : a fixed balanced subset of ArA-DF dev is pushed through EVERY channel

    python scripts/01_extract_channel_ssl.py
Needs ../data/cache/aradf_train.npz and aradf_dev.npz (built by Model 3's
scripts/01_build_cache.py).
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from channel_robust import data  # noqa: E402
from channel_robust.backbone import Backbone  # noqa: E402
from channel_robust.channels import CONDITIONS, apply  # noqa: E402

CACHE = data.DATA_ROOT.parent / "cache"
N_TEST = 1500
SEED = 0


def _degrade(args):
    w, cond, seed = args
    return apply(w.astype(np.float32) / 32767, cond, seed)


def extract(bb, wav_int16, conds, seeds, name, bs=16):
    dst = CACHE / f"ssl_{name}.npy"
    if dst.exists():
        print("exists", dst.name); return
    out = np.zeros((len(wav_int16), 13, 1536), dtype=np.float16)
    t0 = time.time()
    # threads, not processes: ffmpeg runs in its own process and numpy/scipy release the GIL,
    # and a forked/spawned Pool next to an MPS model deadlocks on macOS
    with ThreadPoolExecutor(8) as pool:
        for i in tqdm(range(0, len(wav_int16), bs * 8), desc=name):
            j = slice(i, i + bs * 8)
            degraded = list(pool.map(_degrade, zip(wav_int16[j], conds[j], seeds[j])))
            for k in range(0, len(degraded), bs):
                x = np.stack(degraded[k:k + bs])
                out[i + k:i + k + len(x)] = bb.pooled(__import__("torch").from_numpy(x)).float().cpu().numpy()
    np.save(dst, out)
    print(f"{name}: {len(out)} in {time.time()-t0:.0f}s")


def main():
    rng = np.random.default_rng(SEED)
    bb = Backbone()

    tr = np.load(CACHE / "aradf_train.npz")
    n = len(tr["wav"])
    p = np.array([0.25] + [0.75 / (len(CONDITIONS) - 1)] * (len(CONDITIONS) - 1))
    conds = rng.choice(CONDITIONS, size=n, p=p)
    json.dump(conds.tolist(), open(CACHE / "aradf_train_aug_conditions.json", "w"))
    extract(bb, tr["wav"], conds, np.arange(n) + 1000, "aradf_train_aug")

    dev = np.load(CACHE / "aradf_dev.npz")
    meta = json.loads(str(dev["meta"]))
    y = np.array([m["is_fake"] for m in meta])
    idx = np.sort(np.r_[rng.choice(np.where(y == 0)[0], N_TEST // 2, replace=False),
                        rng.choice(np.where(y == 1)[0], N_TEST // 2, replace=False)])
    json.dump(idx.tolist(), open(CACHE / "aradf_dev_channel_subset.json", "w"))
    for c in CONDITIONS:
        extract(bb, dev["wav"][idx], np.array([c] * len(idx)), idx + 7, f"aradf_dev_{c}")


if __name__ == "__main__":
    main()

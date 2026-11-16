"""Dataset readers.

Every reader yields dicts with a common schema so the rest of the pipeline
does not care where the audio came from:

    {"id": str, "wav": np.ndarray[float32] @16 kHz, "is_fake": 0|1,
     "attack": str, "speaker": str, "corpus": str, ...}

Convention used across the whole project: ``is_fake = 1`` means spoofed /
synthetic speech, and every detector outputs a score where *higher = more fake*.
"""
from __future__ import annotations

import io
import json
import os
import tarfile
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf

SR = 16000
DATA_ROOT = Path(os.environ.get("QDF_DATA", Path(__file__).resolve().parents[3] / "data" / "raw"))


def _decode(raw: bytes) -> np.ndarray:
    wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return wav


# --------------------------------------------------------------------------- #
# ASVspoof 2019 LA (English) - evaluation partition, attacks A07-A19
# --------------------------------------------------------------------------- #
def load_asvspoof_protocol(root: Path = DATA_ROOT / "asvspoof2019") -> pd.DataFrame:
    df = pd.read_csv(root / "protocol.txt", sep=" ", header=None,
                     names=["speaker", "id", "_", "attack", "key"])
    df["is_fake"] = (df["key"] == "spoof").astype(int)
    df.loc[df["attack"] == "-", "attack"] = "bonafide"
    return df[["id", "speaker", "attack", "is_fake"]].set_index("id")


def iter_asvspoof(root: Path = DATA_ROOT / "asvspoof2019", spoof_keep: float = 1.0,
                  seed: int = 0) -> Iterator[dict]:
    """spoof_keep < 1 randomly subsamples spoofed trials (bona fide are always kept)."""
    proto = load_asvspoof_protocol(root)
    rng = np.random.default_rng(seed)
    for shard in sorted(root.glob("test-*.parquet")):
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=256, columns=["path", "audio"]):
            for path, audio in zip(batch.column("path").to_pylist(), batch.column("audio").to_pylist()):
                uid = Path(path).stem
                meta = proto.loc[uid]
                if meta.is_fake and rng.random() > spoof_keep:
                    continue
                yield {"id": uid, "audio_bytes": audio["bytes"], "is_fake": int(meta.is_fake),
                       "attack": meta.attack, "speaker": meta.speaker, "corpus": "asvspoof2019"}


# --------------------------------------------------------------------------- #
# ArA-DF-2026 (Arabic, multi-dialect) - WebDataset tar shards
# --------------------------------------------------------------------------- #
def load_aradf_metadata(split: str, root: Path = DATA_ROOT / "aradf") -> pd.DataFrame:
    labeled = root / "metadata_labeled" / f"{split}.parquet"
    df = pd.read_parquet(labeled if labeled.exists() else root / "metadata" / f"{split}.parquet")
    df["is_fake"] = (df["label"] == 0).astype(int)  # ArA-DF: 1 = bona fide, 0 = spoof
    parts = df["id"].str.split("_")
    if split in ("train", "dev"):
        df["dialect"], df["gender"] = parts.str[0], parts.str[1]
    return df.set_index("id")


def iter_aradf(tar_glob: str, split: str, root: Path = DATA_ROOT / "aradf",
               max_per_shard: int | None = None, seed: int = 0) -> Iterator[dict]:
    """Shards are sorted by dialect, so we sample a fixed random subset from EACH
    shard to keep the dialect mix balanced."""
    meta = load_aradf_metadata(split, root)
    for tar_path in sorted(root.glob(tar_glob)):
        if not Path(str(tar_path) + ".done").exists():   # shard still downloading
            print(f"[data] skipping incomplete shard {tar_path.name}")
            continue
        keep = None
        if max_per_shard:
            shard_id = tar_path.stem.split("-")[-1]
            ids = meta.index[meta["file_name"].str.contains(f"-{shard_id}.tar", regex=False)]
            rng = np.random.default_rng(seed)
            keep = set(rng.choice(ids, size=min(max_per_shard, len(ids)), replace=False))
        with tarfile.open(tar_path) as tf:
            for m in tf:
                if not m.name.endswith(".flac"):
                    continue
                uid = Path(m.name).stem
                if uid not in meta.index or (keep is not None and uid not in keep):
                    continue
                row = meta.loc[uid]
                yield {"id": uid, "audio_bytes": tf.extractfile(m).read(), "is_fake": int(row.is_fake),
                       "attack": "spoof" if row.is_fake else "bonafide",
                       "speaker": f"{row.get('dialect', 'NA')}_{row.get('gender', 'NA')}",
                       "dialect": row.get("dialect", "NA"), "corpus": "aradf2026"}


def decode(item: dict) -> dict:
    item = dict(item)
    item["wav"] = _decode(item.pop("audio_bytes"))
    return item


# --------------------------------------------------------------------------- #
# ArAD (Arabic, mostly Levantine) - RVC v2 voice conversion, parquet
# --------------------------------------------------------------------------- #
def iter_arad(split: str = "test", root: Path = DATA_ROOT / "arad", max_items: int | None = None,
              seed: int = 0) -> Iterator[dict]:
    """ArAD labels: 0 = fake, 1 = real. Clips are <= 3 s."""
    path = root / f"{split}.parquet"
    if not Path(str(path) + ".done").exists():
        print(f"[data] {path.name} not downloaded yet")
        return
    t = pq.read_table(path)
    idx = np.arange(t.num_rows)
    if max_items and max_items < len(idx):
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_items, replace=False))
    audio, label = t.column("audio").to_pylist(), t.column("label").to_pylist()
    for i in idx:
        yield {"id": f"arad_{split}_{i:06d}", "audio_bytes": audio[i]["bytes"], "is_fake": int(label[i] == 0),
               "attack": "rvc" if label[i] == 0 else "bonafide", "speaker": "NA", "dialect": "LEV",
               "corpus": "arad"}

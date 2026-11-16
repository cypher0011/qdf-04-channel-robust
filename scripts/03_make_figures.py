"""README figures for Model 4: channel examples and robustness comparison."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from channel_robust import data  # noqa: E402
from channel_robust.channels import CONDITIONS, apply  # noqa: E402

RES, FIG, CACHE = ROOT / "results", ROOT / "assets" / "figures", data.DATA_ROOT.parent / "cache"
FIG.mkdir(parents=True, exist_ok=True)
CLEAN_C, ROBUST_C, GREY = "#94a3b8", "#16a34a", "#6b7280"
LABEL = {"clean": "Clean\n(studio)", "whatsapp_opus": "WhatsApp\nOpus 12-24k", "phone_g711": "Phone call\nG.711 8 kHz",
         "mp3_32k": "MP3\n32 kbit/s", "aac_32k": "AAC\n32 kbit/s", "replay": "Replay\nspeaker->room->mic",
         "noisy": "Background\nnoise 5-15 dB"}
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11})


def save(fig, name):
    fig.savefig(FIG / name, facecolor="white"); plt.close(fig); print("  ", name)


def fig_channel_gallery():
    z = np.load(CACHE / "aradf_dev.npz")
    meta = json.loads(str(z["meta"]))
    i = next(k for k, m in enumerate(meta) if m["is_fake"] == 1 and m.get("dialect") == "Gulf")
    w = z["wav"][i].astype(np.float32) / 32767
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.2), gridspec_kw={"hspace": 0.45})
    for ax, c in zip(axes.ravel(), CONDITIONS):
        y = apply(w, c, seed=3)
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=160)), ref=np.max)
        librosa.display.specshow(S, sr=16000, hop_length=160, y_axis="hz", x_axis="time", ax=ax, cmap="magma", vmin=-80)
        ax.set_title(LABEL[c].replace("\n", " · "), fontsize=9.5); ax.set_xlabel(""); ax.set_ylabel("")
    axes.ravel()[-1].axis("off")
    axes.ravel()[-1].text(0.05, 0.5, "The same Gulf-dialect\nFAKE utterance after\neach simulated channel.\n\nPhone calls erase\neverything above 3.4 kHz,\nwhere Model 1 found\nmost of its evidence.",
                          fontsize=10, va="center")
    fig.suptitle("What a scam voice looks like after it travels", x=0.01, ha="left", fontweight="bold")
    save(fig, "channel_gallery.png")


def fig_robustness(m):
    conds = list(m["per_condition"])
    keys = list(m["per_condition"][conds[0]])
    x = np.arange(len(conds)); w = 0.38
    fig, ax = plt.subplots(figsize=(12, 4.2))
    for j, (k, col) in enumerate(zip(keys, (CLEAN_C, ROBUST_C))):
        vals = [m["per_condition"][c][k]["eer"] * 100 for c in conds]
        bars = ax.bar(x + (j - 0.5) * w, vals, w, color=col, label=k)
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([LABEL[c] for c in conds], fontsize=8.5)
    ax.set_ylabel("EER (%)  - lower is better"); ax.legend(frameon=False)
    ax.set_title("Same Arabic test utterances, seven channels", loc="left")
    save(fig, "robustness_per_channel.png")


def fig_summary(m):
    keys = list(m["per_condition"]["clean"])
    rows = [("mean over 7\nsimulated channels", [m[f"mean_eer_over_channels::{k}"] for k in keys]),
            ("worst channel", [m[f"worst_eer_over_channels::{k}"] for k in keys])]
    for ext, lab in (("aradf_track2", "ArA-DF Track 2\n(REAL unseen channels)"), ("arad", "ArAD\n(RVC, unseen generator)")):
        if ext in m["external"]:
            rows.append((lab, [m["external"][ext][k]["eer"] for k in keys]))
    x = np.arange(len(rows)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for j, (k, col) in enumerate(zip(keys, (CLEAN_C, ROBUST_C))):
        vals = [r[1][j] * 100 for r in rows]
        bars = ax.bar(x + (j - 0.5) * w, vals, w, color=col, label=k)
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=9); ax.set_ylabel("EER (%)")
    ax.legend(frameon=False); ax.set_title("Robustness summary", loc="left")
    save(fig, "robustness_summary.png")


def main():
    fig_channel_gallery()
    p = RES / "metrics.json"
    if p.exists():
        m = json.load(open(p)); fig_robustness(m); fig_summary(m)


if __name__ == "__main__":
    main()

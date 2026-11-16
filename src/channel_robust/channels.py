"""Simulated real-world channels a scam voice goes through before it reaches
the victim (or the investigator).

Every function maps a 16 kHz float32 waveform to a 16 kHz float32 waveform
of the same length, so the detector always sees the same input format.

    clean          untouched
    whatsapp_opus  Opus VoIP codec at 16 kbit/s (WhatsApp / Telegram voice notes)
    phone_g711     8 kHz narrow-band telephone: 300-3400 Hz band-pass + G.711 mu-law
    mp3_32k        low-bit-rate MP3 (social-media re-uploads)
    aac_32k        AAC at 32 kbit/s (iMessage / Instagram / TikTok)
    replay         played through a loudspeaker and re-recorded in a room:
                   speaker band-limit + soft clipping + synthetic room impulse
                   response + microphone noise
    noisy          coloured background noise at 5-15 dB SNR
"""
from __future__ import annotations

import subprocess

import numpy as np
import scipy.signal as ss

SR = 16000
CONDITIONS = ("clean", "whatsapp_opus", "phone_g711", "mp3_32k", "aac_32k", "replay", "noisy")


def _ffmpeg_roundtrip(wav: np.ndarray, enc_args: list[str], fmt: str, dec_fmt: str | None = None) -> np.ndarray:
    enc = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
                          *enc_args, "-f", fmt, "pipe:1"], input=wav.astype(np.float32).tobytes(),
                         capture_output=True, check=True).stdout
    dec = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", dec_fmt or fmt, "-i", "pipe:0", "-f", "f32le", "-ar", str(SR),
                          "-ac", "1", "pipe:1"], input=enc, capture_output=True, check=True).stdout
    out = np.frombuffer(dec, dtype=np.float32)
    return _fit(_align(out, wav), len(wav))


def _align(out: np.ndarray, ref: np.ndarray, max_lag: int = 3000) -> np.ndarray:
    """Remove the codec's encoder delay (MP3/AAC pad ~1-2k samples) by cross-correlation."""
    n = min(len(ref), len(out)) - max_lag
    if n <= 0:
        return out
    c = np.correlate(out[: n + max_lag], ref[:n], mode="valid")
    lag = int(np.argmax(c))
    return out[lag:]


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))


def whatsapp_opus(wav, rng):
    return _ffmpeg_roundtrip(wav, ["-c:a", "libopus", "-b:a", f"{rng.choice([12, 16, 24])}k",
                                   "-application", "voip"], "ogg")


def mp3_32k(wav, rng):
    return _ffmpeg_roundtrip(wav, ["-c:a", "libmp3lame", "-b:a", "32k"], "mp3")


def aac_32k(wav, rng):
    return _ffmpeg_roundtrip(wav, ["-c:a", "aac", "-b:a", "32k"], "adts", "aac")


def phone_g711(wav, rng):
    x = ss.resample_poly(wav, 1, 2)                                   # 16k -> 8k
    sos = ss.butter(6, [300, 3400], btype="band", fs=8000, output="sos")
    x = np.clip(ss.sosfilt(sos, x), -1, 1)
    mu = 255.0                                                        # G.711 mu-law, 8-bit
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    y = np.round((y + 1) / 2 * 255) / 255 * 2 - 1
    x = np.sign(y) * ((1 + mu) ** np.abs(y) - 1) / mu
    return _fit(ss.resample_poly(x, 2, 1).astype(np.float32), len(wav))


def _synthetic_rir(rng, rt60: float) -> np.ndarray:
    n = int(SR * rt60)
    t = np.arange(n) / SR
    rir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)            # -60 dB at rt60
    rir[: int(0.002 * SR)] = 0
    rir[0] = 1.0                                                     # direct path
    return rir / np.abs(rir).sum() * 8


def replay(wav, rng):
    sos = ss.butter(4, [rng.uniform(120, 250), rng.uniform(6000, 7500)], btype="band", fs=SR, output="sos")
    x = ss.sosfilt(sos, wav)                                          # small loudspeaker
    x = np.tanh(rng.uniform(1.5, 3.0) * x) / np.tanh(1.5)             # speaker non-linearity
    x = ss.fftconvolve(x, _synthetic_rir(rng, rng.uniform(0.2, 0.7)))[: len(wav)]
    x = x / (np.abs(x).max() + 1e-9) * 0.9
    snr = rng.uniform(20, 35)
    x = x + rng.standard_normal(len(x)) * np.sqrt(np.mean(x ** 2) / 10 ** (snr / 10))
    return x.astype(np.float32)


def noisy(wav, rng):
    white = rng.standard_normal(len(wav))
    alpha = rng.uniform(0.5, 1.5)                                     # 1/f^alpha coloured noise
    f = np.fft.rfftfreq(len(wav)); f[0] = f[1]
    n = np.fft.irfft(np.fft.rfft(white) / f ** (alpha / 2), len(wav))
    snr = rng.uniform(5, 15)
    n = n * np.sqrt(np.mean(wav ** 2) / (np.mean(n ** 2) * 10 ** (snr / 10)))
    return (wav + n).astype(np.float32)


FUNCS = {"clean": lambda w, r: w, "whatsapp_opus": whatsapp_opus, "phone_g711": phone_g711,
         "mp3_32k": mp3_32k, "aac_32k": aac_32k, "replay": replay, "noisy": noisy}


def apply(wav: np.ndarray, condition: str, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = FUNCS[condition](wav.astype(np.float32), rng)
    peak = np.abs(out).max()
    return (out / peak * 0.9 if peak > 0 else out).astype(np.float32)

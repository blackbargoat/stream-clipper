#!/usr/bin/env python3
"""Exact A/V sync measurement against a reference copy of the same stream (YouTube's own, which is in sync).

Finds where our clip's VIDEO sits in the reference (frame-change pattern) and where our clip's AUDIO sits in the
reference (waveform), independently. In the reference both line up, so the difference is our clip's sync error.

  python3 sync_ref.py ours.mp4 reference.mp4
  -> {"video_at": 81.36, "audio_at": 79.36, "audio_early": 2.0, "video_match": 0.97, "audio_match": 0.91}

audio_early > 0 means the voice comes before the lips by that many seconds. Fix with render --audio-shift -audio_early.
Matches are correlation peaks (0..1); below ~0.5 the clip isn't in the reference window.
"""
import json, subprocess, sys
import numpy as np

AR = 8000   # audio rate for matching
VR = 30     # video frames per second for matching


def _audio(path, t=None):
    cmd = ["ffmpeg", "-v", "error", "-i", path] + (["-t", str(t)] if t else []) + ["-vn", "-ac", "1", "-ar", str(AR), "-f", "s16le", "-"]
    a = np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.int16).astype(np.float32)
    return a / (np.abs(a).max() + 1)


def _video(path, t=None):
    """Per-frame luminance change of the whole frame at 64x36: a fingerprint of cuts and motion."""
    cmd = ["ffmpeg", "-v", "error", "-i", path] + (["-t", str(t)] if t else []) + \
          ["-an", "-vf", f"fps={VR},scale=64:36,format=gray", "-f", "rawvideo", "-"]
    f = np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.uint8).reshape(-1, 36, 64).astype(np.float32)
    return np.r_[0, np.abs(np.diff(f, axis=0)).mean(axis=(1, 2))]


def _locate(needle, hay, rate):
    """Where `needle` starts inside `hay` (seconds), by normalized cross-correlation (FFT), sub-sample peak."""
    n = needle - needle.mean()
    L = len(hay) + len(n)
    N = 1 << (L - 1).bit_length()
    c = np.fft.irfft(np.fft.rfft(hay, N) * np.conj(np.fft.rfft(n, N)), N)[: len(hay) - len(n) + 1]
    # normalize by the energy of each hay window
    cs = np.cumsum(np.r_[0, hay]); cs2 = np.cumsum(np.r_[0, hay ** 2]); m = len(n)
    win_mean = (cs[m:] - cs[:-m]) / m
    win_var = (cs2[m:] - cs2[:-m]) - m * win_mean ** 2
    c = c / (np.sqrt(np.maximum(win_var, 1e-9)) * np.sqrt((n ** 2).sum()) + 1e-9)
    k = int(np.argmax(c)); frac = 0.0
    if 0 < k < len(c) - 1:
        y0, y1, y2 = c[k - 1], c[k], c[k + 1]; den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if den else 0.0
    return float(k + frac) / rate, float(c[k])


def measure(ours, ref, seconds=40):
    v_at, v_m = _locate(_video(ours, seconds), _video(ref), VR)
    a_at, a_m = _locate(_audio(ours, seconds), _audio(ref), AR)
    return {"video_at": round(v_at, 3), "audio_at": round(a_at, 3), "audio_early": round(a_at - v_at, 3),
            "video_match": round(v_m, 3), "audio_match": round(a_m, 3)}


if __name__ == "__main__":
    print(json.dumps(measure(sys.argv[1], sys.argv[2])))

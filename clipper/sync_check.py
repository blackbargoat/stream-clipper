#!/usr/bin/env python3
"""Audio/video sync check. Lines up mouth movement in the facecam with the loudness of the voice and returns the
shift (seconds) where they match best. Positive = the audio comes LATE (voice after the lips), negative = early.

  python3 sync_check.py clip_source.mp4      -> {"offset": 0.48, "confidence": 3.1}
Confidence is how far the best match stands above the rest (peak z-score); under ~2.2 means "can't tell".
"""
import json, subprocess, sys
import numpy as np

FPS = 25
MAX_LAG = 4.0  # seconds searched either way


def mouth_motion(src, face, dur):
    cx, cy, fw = face
    w, h = int(fw * 0.8) // 2 * 2, int(fw * 0.55) // 2 * 2
    x, y = int(cx - w / 2), int(cy + fw * 0.08)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-t", str(dur), "-vf", f"crop={w}:{h}:{max(x,0)}:{max(y,0)},scale=48:32,fps={FPS},format=gray",
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    f = np.frombuffer(raw, np.uint8).reshape(-1, 32, 48).astype(np.float32)
    return np.r_[0, np.abs(np.diff(f, axis=0)).mean(axis=(1, 2))]


def voice_level(src, dur):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-t", str(dur), "-vn", "-ac", "1", "-ar", "16000",
                          "-af", "highpass=f=150,lowpass=f=3500", "-f", "s16le", "-"], capture_output=True, check=True).stdout
    a = np.frombuffer(raw, np.int16).astype(np.float32)
    hop = 16000 // FPS
    n = len(a) // hop
    rms = np.sqrt((a[: n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1)
    return 20 * np.log10(rms)


def onsets(x):
    """Changes, not levels: smooth a little, take the rise, z-score."""
    x = np.convolve(x, np.ones(3) / 3, "same")
    d = np.clip(np.diff(x, prepend=x[0]), 0, None)
    return (d - d.mean()) / (d.std() + 1e-6)


def check(src, face=None, dur=60):
    if face is None:
        import render
        sw, sh_, d = render.probe(src)
        cam = render.find_cam(src, sw, sh_, d)
        face = render.LAST_FACE
        if not face:
            return {"offset": 0.0, "confidence": 0.0, "note": "no face found"}
        dur = min(dur, d)
    v, a = onsets(mouth_motion(src, face, dur)), onsets(voice_level(src, dur))
    n = min(len(v), len(a))
    v, a = v[:n], a[:n]
    L = int(MAX_LAG * FPS)
    lags = np.arange(-L, L + 1)
    # corr at lag k: audio shifted earlier by k frames vs video (k>0 = audio late)
    c = np.array([np.mean(v[max(0, -k): n - max(0, k)] * a[max(0, k): n - max(0, -k)]) for k in lags])
    best = int(np.argmax(c))
    conf = float((c[best] - np.median(c)) / (c.std() + 1e-6))
    frac = 0.0
    if 0 < best < len(c) - 1:  # parabolic peak: sub-frame precision
        y0, y1, y2 = c[best - 1], c[best], c[best + 1]
        den = y0 - 2 * y1 + y2
        frac = float(0.5 * (y0 - y2) / den) if den else 0.0
    return {"offset": round((float(lags[best]) + frac) / FPS, 2), "confidence": round(conf, 2)}


if __name__ == "__main__":
    print(json.dumps(check(sys.argv[1])))

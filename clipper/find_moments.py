#!/usr/bin/env python3
"""Find the clip-worthy moments in a stream.

Signals (each turned into a per-second score, z-scored, then combined):
  1. Audio: loudness above the stream's own running level. Screams and yelling sit 12-25 dB over normal talk;
     a sustained burst (2+ seconds) counts more than a single bang.
  2. Chat (optional): messages per second against the running rate, plus "hype" messages (💀 😂 LMAO W HELP ...).
     Works with yt-dlp's live_chat JSON, recorded live (Speed turns chat replay off) or downloaded as a replay.
  3. YouTube "most replayed" heatmap (optional, from yt-dlp metadata) when the VOD has one.

  python3 find_moments.py --wav audio16k.wav [--chat x.live_chat.json --start-ts <stream start unix s>] \
      [--meta meta.json] [--n 12] [--out moments.json]
Output: [{"t": peak second, "start": s, "end": s, "score": x, "why": [...]}] best first, at least 90 s apart.
"""
import argparse, json, re, wave
import numpy as np

HYPE = re.compile(r"(💀|😂|🤣|😭|lmao|lmfao|lol|w+\b|helps?\b|caught|ayo|bro what|nah|crazy|no way|omg|clip it|clip that|speed|kai|🔥|😱)", re.I)


def audio_score(path):
    with wave.open(path) as w:
        sr = w.getframerate(); a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    n = len(a) // sr
    rms = np.sqrt(np.mean(a[: n * sr].reshape(n, sr) ** 2, axis=1) + 1e-9)
    db = 20 * np.log10(rms)
    # running level: median of the surrounding 2 minutes, so a loud game section doesn't look like a reaction
    pad = 60
    base = np.array([np.median(db[max(0, i - pad): i + pad]) for i in range(n)])
    over = np.clip(db - base, 0, None)
    # reward sustained bursts: 3 s window mean
    k = np.ones(3) / 3
    return np.convolve(over, k, mode="same"), n


def chat_score(path, start_ts, n):
    rate = np.zeros(n); hype = np.zeros(n)
    for line in open(path, encoding="utf-8"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        act = d.get("replayChatItemAction", {})
        off = act.get("videoOffsetTimeMsec")
        for a in act.get("actions", []):
            item = a.get("addChatItemAction", {}).get("item", {})
            r = item.get("liveChatTextMessageRenderer") or item.get("liveChatPaidMessageRenderer")
            if not r:
                continue
            if off is not None and int(off) > 0:
                t = int(off) // 1000
            elif start_ts and r.get("timestampUsec"):
                t = int(int(r["timestampUsec"]) / 1e6 - start_ts)
            else:
                continue
            if not 0 <= t < n:
                continue
            text = "".join(x.get("text", "") or x.get("emoji", {}).get("emojiId", "") for x in r.get("message", {}).get("runs", []))
            rate[t] += 1
            if HYPE.search(text):
                hype[t] += 1
    # chat lags the moment by ~5-15 s: shift back 8 s
    lag = 8
    rate = np.roll(rate, -lag); hype = np.roll(hype, -lag)
    k = np.ones(10) / 10
    r, h = np.convolve(rate, k, "same"), np.convolve(hype, k, "same")
    base = np.array([np.median(r[max(0, i - 300): i + 300]) for i in range(n)]) + 0.5
    return (r / base) + 2 * (h / (base * 0.3 + 0.5)), rate.sum()


def heatmap_score(meta, n):
    s = np.zeros(n)
    for h in (meta or {}).get("heatmap") or []:
        a, b = int(h["start_time"]), min(n, int(h["end_time"]) + 1)
        s[a:b] = np.maximum(s[a:b], h["value"])
    return s


def z(x):
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x * 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wav", required=True); p.add_argument("--chat"); p.add_argument("--start-ts", type=float)
    p.add_argument("--meta"); p.add_argument("--n", type=int, default=12); p.add_argument("--gap", type=int, default=90)
    p.add_argument("--skip-start", type=int, default=120, help="ignore the 'starting soon' minutes")
    p.add_argument("--out", default="moments.json")
    a = p.parse_args()
    aud, n = audio_score(a.wav)
    parts = {"audio": (z(aud), 1.0)}
    if a.chat:
        c, total = chat_score(a.chat, a.start_ts, n)
        if total > 50:
            parts["chat"] = (z(c), 1.4)
    meta = json.load(open(a.meta)) if a.meta else None
    hm = heatmap_score(meta, n)
    if hm.any():
        parts["replayed"] = (z(hm), 1.6)
    score = sum(w * s for s, w in parts.values())
    score[: a.skip_start] = -1e9
    score[-30:] = -1e9
    picks = []
    order = np.argsort(-score)
    for t in order:
        if len(picks) >= a.n or score[t] < 1.0:
            break
        if all(abs(int(t) - q["t"]) >= a.gap for q in picks):
            why = [k for k, (s, w) in parts.items() if s[t] > 1.5]
            # window: build-up before the peak, reaction after; refine later on the transcript
            picks.append({"t": int(t), "start": max(0, int(t) - 32), "end": min(n, int(t) + 18),
                          "score": round(float(score[t]), 2), "why": why or ["audio"]})
    json.dump(picks, open(a.out, "w"), indent=1)
    for q in picks:
        print(f"{q['t']//3600}:{q['t']%3600//60:02d}:{q['t']%60:02d}  score {q['score']:5.2f}  {','.join(q['why'])}")


if __name__ == "__main__":
    main()

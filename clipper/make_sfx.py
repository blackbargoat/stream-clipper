"""Original sound effects, generated from scratch (no samples): riser, whoosh, impact, hit. 48 kHz stereo WAV."""
import numpy as np, wave, os
SR = 48000
rng = np.random.default_rng(7)
def save(name, x, width=0.25):
    x = x / (np.abs(x).max() + 1e-9) * 0.89
    # light stereo width: tiny delay on one side
    d = int(SR * 0.004); l = x; r = np.r_[np.zeros(d), x[:-d]] * (1 - width) + x * width
    st = (np.stack([l, r], 1) * 32767).astype(np.int16)
    with wave.open(os.path.join("sfx", name), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(st.tobytes())
def lowpass(x, a):   # one-pole, a in 0..1 (per-sample array allowed)
    y = np.zeros_like(x); s = 0.0
    a = np.broadcast_to(a, x.shape)
    for i in range(len(x)): s += a[i] * (x[i] - s); y[i] = s
    return y
t = lambda sec: np.arange(int(SR * sec)) / SR

# RISER (1.6 s): noise swept up through a filter + a rising tone, swelling into the cut
T = t(1.6); p = T / T[-1]
noise = rng.standard_normal(len(T))
sweep = lowpass(noise, 0.01 + 0.5 * p ** 2.2) - lowpass(noise, 0.004 + 0.05 * p ** 2)
tone = np.sin(2 * np.pi * np.cumsum(180 + 1500 * p ** 2.5) / SR) * 0.35
riser = (sweep * 1.2 + tone) * p ** 1.8
riser[-int(SR * 0.02):] *= np.linspace(1, 0, int(SR * 0.02))
save("riser.wav", riser)

# WHOOSH (0.55 s): band-passed noise with a fast-in, slow-out envelope and a falling filter
T = t(0.55); p = T / T[-1]
n = rng.standard_normal(len(T))
bp = lowpass(n, 0.35 - 0.3 * p) - lowpass(n, 0.03)
env = np.sin(np.pi * np.clip(p * 1.15, 0, 1)) ** 1.5
save("whoosh2.wav", bp * env, width=0.5)

# IMPACT (1.2 s): sub drop 70->38 Hz + a short noise crack + tail
T = t(1.2)
f = 38 + 32 * np.exp(-T * 9)
sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-T * 3.2)
crack = lowpass(rng.standard_normal(len(T)), 0.6) * np.exp(-T * 40) * 0.6
body = lowpass(rng.standard_normal(len(T)), 0.05) * np.exp(-T * 6) * 0.5
save("impact.wav", np.tanh((sub * 1.4 + crack + body) * 1.6), width=0.1)

# HIT (0.25 s): bright snap for the colour flash
T = t(0.25)
snap = (rng.standard_normal(len(T)) - lowpass(rng.standard_normal(len(T)), 0.2)) * np.exp(-T * 30)
ping = np.sin(2 * np.pi * 1900 * T) * np.exp(-T * 22) * 0.4
save("hit.wav", snap + ping, width=0.4)
print("ok")

# DING (0.9 s): bright bell for a green "good" flash: inharmonic partials with fast attack and long decay
T = t(0.9)
f0 = 1568.0
bell = sum(a * np.sin(2 * np.pi * f0 * r * T) * np.exp(-T * d) for r, a, d in ((1, 1.0, 5.0), (2.0, 0.45, 7.0), (2.76, 0.3, 9.0), (5.4, 0.12, 14.0)))
bell *= np.minimum(1, T / 0.004)
save("ding.wav", bell, width=0.3)

# BUZZ (0.45 s): low "wrong" buzzer for a red "bad" flash
T = t(0.45)
sq = np.sign(np.sin(2 * np.pi * 116 * T)) * 0.6 + np.sign(np.sin(2 * np.pi * 123 * T)) * 0.4
buzz = lowpass(sq, 0.25) * np.minimum(1, T / 0.01) * np.minimum(1, (T[-1] - T) / 0.06)
save("buzz.wav", buzz, width=0.15)
print("ding/buzz ok")

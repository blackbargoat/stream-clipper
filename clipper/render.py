#!/usr/bin/env python3
"""Turn one stream moment into a finished 9:16 Short.

  python3 render.py --src section.mp4 --peak 32 [--title "TEXT"] --out short.mp4 [--words-out words.json]

1. Transcribe (faster-whisper, word timestamps).
2. Pick the cut: start in a pause ~12-30 s before the peak, end in a pause after the reaction, 15-58 s long.
3. Find the facecam: faces that sit in the same small box across sampled frames. Facecam -> cam on top
   (38%), gameplay below. No facecam (IRL, full-cam) -> one vertical crop that follows the face.
4. Word-by-word captions (spoken word in yellow) on the seam, optional white title box at the top,
   audio leveled to -14 LUFS. One encode, H.264 CRF 18, 30 fps.
"""
import argparse, json, os, subprocess, sys, tempfile
import numpy as np

FONT_DIR = os.environ.get("CLIP_FONTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts"))
W, H = 1080, 1920


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def probe(p):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height:format=duration",
                          "-of", "json", p], capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    return d["streams"][0]["width"], d["streams"][0]["height"], float(d["format"]["duration"])


def transcribe(src, model_size):
    from faster_whisper import WhisperModel
    wav = tempfile.mktemp(suffix=".wav")
    sh(["ffmpeg", "-v", "error", "-y", "-i", src, "-ac", "1", "-ar", "16000", wav])
    m = WhisperModel(model_size, device="cpu", compute_type="int8")
    segs, _ = m.transcribe(wav, word_timestamps=True, vad_filter=True, language="en")
    words = [{"start": w.start, "end": w.end, "text": w.word.strip()} for s in segs for w in (s.words or []) if w.word.strip()]
    os.remove(wav)
    return words


def pick_cut(words, peak, dur, lo=15, hi=58):
    """Start in a gap between words 12-30 s before the peak; end in a gap 6-20 s after it."""
    gaps = [(0.0, words[0]["start"])] if words else [(0.0, dur)]
    for a, b in zip(words, words[1:]):
        gaps.append((a["end"], b["start"]))
    if words:
        gaps.append((words[-1]["end"], dur))
    mid = lambda g: (g[0] + g[1]) / 2
    starts = [g for g in gaps if peak - 30 <= mid(g) <= peak - 10] or [(max(0, peak - 22),) * 2]
    s = max(starts, key=lambda g: g[1] - g[0])  # the longest pause = a natural beginning
    ends = [g for g in gaps if peak + 6 <= mid(g) <= min(dur, peak + 22) and lo <= mid(g) - mid(s) <= hi]
    e = max(ends, key=lambda g: g[1] - g[0]) if ends else (min(dur, mid(s) + 40),) * 2
    start, end = max(0.0, mid(s) - 0.15), min(dur, mid(e) + 0.15)
    return round(start, 2), round(min(end, start + hi), 2)


def find_cam(src, sw, sh_, dur):
    """Return (x, y, w, h) of a facecam box, ('face', cx) for full-frame camera footage, or None.
    The real facecam sits in the same spot in every frame; faces in the game move around. So detect faces in
    16 frames (two cascades), cluster by position, and keep the spot that shows up in the most frames."""
    import cv2
    dets = [cv2.CascadeClassifier(cv2.data.haarcascades + f) for f in
            ("haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml")]
    cap = cv2.VideoCapture(src)
    frames = []
    for k in range(16):
        cap.set(cv2.CAP_PROP_POS_MSEC, 1000 * dur * (k + 0.5) / 16)
        ok, f = cap.read()
        if not ok:
            continue
        g = cv2.equalizeHist(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        hits = []
        for det in dets:
            hits += [tuple(map(int, h)) for h in det.detectMultiScale(g, 1.05, 5, minSize=(int(sh_ * 0.025),) * 2)]
        frames.append(hits)
    cap.release()
    allhits = [(fi, h) for fi, hs in enumerate(frames) for h in hs]
    if not allhits:
        return None
    best, best_n = None, 0
    for fi, (x, y, w, h) in allhits:
        cx, cy = x + w / 2, y + h / 2
        seen = {fj for fj, (x2, y2, w2, h2) in allhits
                if abs(x2 + w2 / 2 - cx) < w * 0.5 and abs(y2 + h2 / 2 - cy) < h * 0.5 and 0.6 < w2 / w < 1.6}
        if len(seen) > best_n:
            best_n, best = len(seen), (cx, cy, w)
    if best_n < max(4, len(frames) // 3):
        return None
    cx, cy, fw = best
    if fw > sh_ * 0.22:  # a big face = the camera is the whole frame (IRL)
        return ("face", cx)
    bw = min(fw * 4.6, sw * 0.5); bh = bw / (W / (H * 0.38))  # match the top area's shape so nothing squashes
    x = int(min(max(cx - bw / 2, 0), sw - bw)); y = int(min(max(cy - bh * 0.42, 0), sh_ - bh))
    return (x, y, int(bw), int(bh))


def caption_track(words, start, end, y, tmp):
    """Word-by-word captions as transparent PNGs + a concat list (groups of up to 3 words, spoken word yellow).
    Pure Pillow, so it works with any ffmpeg build (no libass needed)."""
    from PIL import Image, ImageDraw, ImageFont
    F = ImageFont.truetype(os.path.join(FONT_DIR, "TikTokSans-Black.ttf"), 86)
    ws = [w for w in words if w["end"] > start and w["start"] < end]
    BH = 260  # caption strip height; overlaid centred on y
    blank = os.path.join(tmp, "blank.png")
    Image.new("RGBA", (W, BH), (0, 0, 0, 0)).save(blank)
    events, t = [], 0.0
    for i in range(0, len(ws), 3):
        grp = ws[i:i + 3]
        for j, w in enumerate(grp):
            a = max(w["start"], start) - start
            b = min((grp[j + 1]["start"] if j + 1 < len(grp) else w["end"] + 0.25), end) - start
            if b <= a:
                continue
            im = Image.new("RGBA", (W, BH), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
            parts = [g["text"].upper().strip(",.?!") for g in grp]
            line = " ".join(parts)
            size = 86
            while d.textlength(line, font=F) > W - 120 and size > 50:
                size -= 6; F = ImageFont.truetype(os.path.join(FONT_DIR, "TikTokSans-Black.ttf"), size)
            x = (W - d.textlength(line, font=F)) / 2
            for k, ptxt in enumerate(parts):
                col = (255, 229, 0, 255) if k == j else (255, 255, 255, 255)
                d.text((x, BH / 2), ptxt, font=F, fill=col, stroke_width=8, stroke_fill=(0, 0, 0, 255), anchor="lm")
                x += d.textlength(ptxt + " ", font=F)
            p = os.path.join(tmp, f"c{len(events):04d}.png"); im.save(p)
            if a > t:
                events.append((blank, a - t))
            events.append((p, b - a)); t = b
    total = end - start
    if total > t:
        events.append((blank, total - t))
    lst = os.path.join(tmp, "captions.txt")
    with open(lst, "w") as f:
        for p, d in events:
            f.write(f"file '{p}'\nduration {d:.3f}\n")
        f.write(f"file '{events[-1][0]}'\n")
    return lst


def render(src, peak, out, title=None, model="base.en", words=None):
    sw, sh_, dur = probe(src)
    words = words if words is not None else transcribe(src, model)
    start, end = pick_cut(words, peak, dur)
    cam = find_cam(src, sw, sh_, dur)
    split = 0.38
    if cam and cam[0] != "face":
        x, y, bw, bh = cam
        ch = int(H * split)
        # gameplay: crop the frame to the bottom area's aspect, centred, avoiding the cam box where possible
        gh = H - ch
        gw_src = int(sh_ * W / gh)
        gx = int(min(max((sw - gw_src) / 2, 0), sw - gw_src))
        vf = (f"[0:v]split[c][g];[c]crop={bw}:{bh}:{x}:{y},scale={W}:{ch}:force_original_aspect_ratio=increase,crop={W}:{ch}[cam];"
              f"[g]crop={gw_src}:{sh_}:{gx}:0,scale={W}:{gh}[game];[cam][game]vstack")
        cap_y = ch + 230
        layout = "cam+game"
    else:
        cw = int(sh_ * W / H)
        cx = cam[1] if cam else sw / 2
        x = int(min(max(cx - cw / 2, 0), sw - cw))
        vf = f"[0:v]crop={cw}:{sh_}:{x}:0,scale={W}:{H}"
        cap_y = int(H * 0.68)
        layout = "vertical"
    tmp = tempfile.mkdtemp()
    lst = caption_track(words, start, end, cap_y, tmp)
    inputs = ["-ss", str(start), "-to", str(end), "-i", src, "-f", "concat", "-safe", "0", "-i", lst]
    vf += f",fps=30[base0];[base0][1:v]overlay=0:{cap_y - 130}:format=auto:eof_action=pass"
    if title:
        # white title box (Joe Bart style), drawn as a transparent PNG so emoji render; sits on the seam or near the top
        png = os.path.join(tmp, "title.png")
        seam = int(H * split) if layout == "cam+game" else 260
        sh([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_title.py"), title, png, str(seam)])
        inputs += ["-i", png]
        vf += "[base];[base][2:v]overlay=0:0:format=auto"
    vf += ",format=yuv420p[v]"
    sh(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", vf, "-map", "[v]", "-map", "0:a",
        "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-c:a", "aac", "-b:a", "192k",
        "-ar", "48000", "-movflags", "+faststart", out])
    text = " ".join(w["text"] for w in words if start <= w["start"] <= end)
    return {"start": start, "end": end, "duration": round(end - start, 1), "layout": layout, "cam": cam if cam and cam[0] != "face" else None, "transcript": text}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True); p.add_argument("--peak", type=float, required=True); p.add_argument("--out", required=True)
    p.add_argument("--title"); p.add_argument("--model", default="base.en"); p.add_argument("--words-out")
    a = p.parse_args()
    info = render(a.src, a.peak, a.out, a.title, a.model)
    print(json.dumps(info, indent=1))

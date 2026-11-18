#!/usr/bin/env python3
"""Clip a stream while it's live (or a finished VOD), straight into the clip_clips queue.

  SUPABASE_SERVICE_KEY=... python3 live.py --stream <clip_streams.id> [--max-minutes 340]

Live:
  * ffmpeg records the stream's HLS feed into 60-second .ts pieces (piece i = capture seconds i*60 .. i*60+60).
  * Chat is recorded alongside (YouTube: yt-dlp live_chat; Twitch: anonymous IRC), timestamped by wall clock.
  * Every 4 minutes the newest audio + chat is scored (find_moments). Each strong moment is cut from the pieces,
    rendered (render.py), uploaded to the 'clips' bucket and added as a clip row. A moment must score high
    against the whole stream so far, so the quiet start doesn't flood the queue.
  * When the stream ends (or this job hits its time limit while still live, so the next job takes over) it stops.
VOD (not live): downloads the audio (+ chat replay when there is one), finds the best moments over the whole
stream, downloads just those sections and renders them.
"""
import argparse, glob, json, os, re, shutil, socket, subprocess, sys, threading, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import log, rest, upload, title_for, review_for, ytdlp_json  # noqa: E402
import find_moments as fm  # noqa: E402
import render as rd  # noqa: E402
import sync_audit  # noqa: E402

WORK = os.environ.get("CLIP_WORK", "/tmp/clipwork")
SEG = 60
MIN_SCORE = float(os.environ.get("CLIP_MIN_SCORE", "3.2"))   # combined z-score a moment needs
KEEP_SCORE = float(os.environ.get("CLIP_KEEP_SCORE", 6))
AUTO_APPROVE_SCORE = 8.0
MAX_PER_HOUR = int(os.environ.get("CLIP_MAX_PER_HOUR", "6"))


def stream_row(sid):
    rows = rest("GET", f"clip_streams?id=eq.{sid}&select=*,clip_sources(name,platform,handle,hashtags)")
    return rows[0]


def heartbeat(sid, **kw):
    rest("PATCH", f"clip_streams?id=eq.{sid}", {"runner_seen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kw}, prefer="return=minimal")


# ---------- chat ----------
def twitch_chat(channel, out, stop):
    s = socket.socket(); s.settimeout(30); s.connect(("irc.chat.twitch.tv", 6667))
    s.send(b"NICK justinfan%d\r\n" % int(time.time() % 100000)); s.send(f"JOIN #{channel.lower()}\r\n".encode())
    buf = ""
    with open(out, "a", encoding="utf-8") as f:
        while not stop.is_set():
            try:
                buf += s.recv(4096).decode("utf-8", "ignore")
            except socket.timeout:
                continue
            except OSError:
                break
            *lines, buf = buf.split("\r\n")
            for ln in lines:
                if ln.startswith("PING"):
                    s.send(b"PONG :tmi.twitch.tv\r\n")
                m = re.match(r":\w+!\w+@\S+ PRIVMSG #\w+ :(.*)", ln)
                if m:  # same shape find_moments reads: a live chat action with a wall-clock timestamp
                    f.write(json.dumps({"replayChatItemAction": {"actions": [{"addChatItemAction": {"item": {"liveChatTextMessageRenderer": {
                        "timestampUsec": str(int(time.time() * 1e6)), "message": {"runs": [{"text": m.group(1)}]}}}}}]}}) + "\n")
                    f.flush()


def start_chat(platform, handle, url, out, stop):
    if platform == "twitch":
        t = threading.Thread(target=twitch_chat, args=(handle, out, stop), daemon=True); t.start(); return None
    # yt-dlp writes <out>.live_chat.json; rename handled by reading the glob
    return subprocess.Popen(["yt-dlp", "--no-warnings", "-q", "--skip-download", "--write-subs", "--sub-langs", "live_chat",
                             "-o", out.replace(".jsonl", ""), url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def chat_file(base):
    c = glob.glob(base.replace(".jsonl", "") + "*live_chat.json*") + glob.glob(base)
    return max(c, key=os.path.getsize) if c else None


# ---------- moments ----------
def audio_wav(pieces, dst):
    lst = dst + ".txt"
    with open(lst, "w") as f:
        for p in pieces:
            f.write(f"file '{p}'\n")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-vn", "-ac", "1", "-ar", "16000", dst], check=True)


def pcm_to_wav(pcm, dst):
    import wave
    data = open(pcm, "rb").read()
    data = data[: len(data) // 2 * 2]
    with wave.open(dst, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(data)


def score(wav, chat, t0):
    aud, n = fm.audio_score(wav)
    total = fm.z(aud)
    why = {"audio": fm.z(aud)}
    if chat and os.path.exists(chat):
        c, cnt = fm.chat_score(chat, t0, n)
        if cnt > 50:
            why["chat"] = fm.z(c); total = total + 1.4 * why["chat"]
    return total, why, n


def cut(pieces_dir, a, b, dst):
    """Concatenate the pieces that cover capture seconds a..b, then trim exactly."""
    i0, i1 = int(a // SEG), int(b // SEG)
    files = [os.path.join(pieces_dir, f"p{i:05d}.ts") for i in range(i0, i1 + 1)]
    files = [f for f in files if os.path.exists(f)]
    lst = dst + ".txt"
    with open(lst, "w") as f:
        for p in files:
            f.write(f"file '{p}'\n")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-ss", str(a - i0 * SEG), "-t", str(b - a),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-c:a", "aac", "-b:a", "192k", dst], check=True)


def make_clip(row, src, peak_in_src, t_abs, sc, why, capture_start=None):
    sid = row["id"]; who = (row.get("clip_sources") or {}).get("name") or "streamer"
    sync = None
    if row.get("_edge") and capture_start is not None:  # YouTube: verify lip sync against YouTube's own copy
        try:
            src, trim, sync = sync_audit.audit(src, row["_edge"], capture_start, os.path.join(WORK, f"ref_{t_abs}"))
            peak_in_src -= trim
            log(f"  sync {t_abs}s: {sync}")
            if sync["status"] == "still off":
                log(f"  skip {t_abs}s: lip sync could not be repaired"); return None
        except Exception as e:  # noqa: BLE001
            sync = {"status": "unverified", "error": str(e)[:120]}; log(f"  sync check failed: {e}")
    words = rd.transcribe(src, os.environ.get("CLIP_WHISPER", "base.en"))
    start, end = rd.pick_cut(words, peak_in_src, rd.probe(src)[2])
    text = " ".join(w["text"] for w in words if start <= w["start"] <= end)
    if len(text.split()) < 6:
        log(f"  skip {t_abs}s: almost no talking"); return None
    title = title_for(text, who, row.get("title"))
    # Review: is it worth keeping? Clips under KEEP_SCORE go to Skipped (you can bring them back on the site).
    rv = review_for(text, title, who, row.get("title"), sc)
    if rv and rv.get("better_title"):
        title = rv["better_title"]
    settings = {r["key"]: r["value"] for r in (rest("GET", "clip_settings?select=key,value") or [])}
    keep = (rv is None) or (rv.get("keep") and float(rv.get("score", 0)) >= KEEP_SCORE)
    status = "skipped" if not keep else ("approved" if settings.get("auto_approve") is True and float(rv.get("score", 0)) >= AUTO_APPROVE_SCORE else "ready")
    out = os.path.join(WORK, f"clip_{t_abs}.mp4")
    info = rd.render(src, peak_in_src, out, title, words=words)
    thumb = out.replace(".mp4", ".jpg")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "1.5", "-i", out, "-frames:v", "1", "-vf", "scale=540:-2", "-q:v", "4", thumb], check=True)
    key = f"{sid[:8]}/{t_abs}"
    v = upload(out, key + ".mp4", "video/mp4"); th = upload(thumb, key + ".jpg", "image/jpeg")
    rest("POST", "clip_clips?on_conflict=stream_id,t", [{"stream_id": sid, "t": t_abs, "score": round(sc, 2), "why": why, "title": title,
        "transcript": info["transcript"][:4000], "duration": info["duration"], "layout": info["layout"], "video_url": v, "thumb_url": th, "sync": sync,
        "review_score": rv.get("score") if rv else None, "review": rv, "status": status, "stage": "edited"}],
        prefer="resolution=merge-duplicates,return=minimal")
    log(f"  clip {t_abs}s → {title} ({info['duration']}s, {info['layout']}) review {rv.get('score') if rv else '-'} → {status}")
    return v


def feed_gap(logf, two_feeds, wait=90):
    """Seconds between the video feed's start and the audio feed's start, read from ffmpeg's 'Input #N ... start:' lines."""
    if not two_feeds:
        return 0.0
    t_end = time.time() + wait
    while time.time() < t_end:
        txt = open(logf, errors="ignore").read()
        st = {}
        for m in re.finditer(r"Input #(\d+),.*?start: ([\d.]+)", txt, re.S):
            st[int(m.group(1))] = float(m.group(2))
        if 0 in st and 1 in st:
            g = st[1] - st[0]
            return g if abs(g) < 30 else 0.0
        time.sleep(2)
    log("couldn't read feed start times; assuming no gap")
    return 0.0


def run_live(row, max_minutes):
    sid, url = row["id"], row["url"]
    src = row.get("clip_sources") or {}
    os.makedirs(WORK, exist_ok=True); pdir = os.path.join(WORK, "pieces"); os.makedirs(pdir, exist_ok=True)
    # YouTube lives come as separate video and audio HLS feeds; Twitch as one combined feed
    feed = subprocess.run(["yt-dlp", "--no-warnings", "-g", "-f", "bv*[height<=1080]+ba/b[height<=1080]/best", url],
                          capture_output=True, text=True, timeout=120).stdout.strip().splitlines()
    if not feed:
        raise RuntimeError("no live feed (stream may have ended)")
    t0 = time.time()
    pcm = os.path.join(WORK, "audio.pcm")  # continuous 16 kHz mono audio: the timeline every moment is measured on
    rc = ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_on_network_error", "1", "-reconnect_delay_max", "10"]
    ins = [*rc, "-i", feed[0]] + ([*rc, "-i", feed[1]] if len(feed) > 1 else [])
    a_in = "1:a:0" if len(feed) > 1 else "0:a:0"
    # YouTube's video and audio feeds start at different points on the stream clock (about 2 s apart). -copyts keeps
    # that clock in the video pieces so voice and lips stay together. The audio timeline (audio.pcm) starts when the
    # audio feed starts, so its second 0 sits `gap` seconds into the video; every cut adds the gap back.
    flog = open(os.path.join(WORK, "ffmpeg.log"), "w")
    rec = subprocess.Popen(["ffmpeg", "-hide_banner", "-v", "info", "-nostats", "-y", "-copyts", *ins,
                            "-map", "0:v:0", "-map", a_in, "-c", "copy", "-f", "segment", "-segment_time", str(SEG),
                            "-reset_timestamps", "1", os.path.join(pdir, "p%05d.ts"),
                            "-map", a_in, "-ac", "1", "-ar", "16000", "-f", "s16le", pcm], stderr=flog)
    gap = feed_gap(os.path.join(WORK, "ffmpeg.log"), len(feed) > 1)
    if len(feed) > 1 and "googlevideo" in feed[0]:
        try:
            row["_edge"] = sync_audit.live_edge(feed)
        except Exception as e:  # noqa: BLE001
            log("no YouTube piece numbers, sync audit off:", e)
    log(f"audio starts {gap:+.3f}s after video on the stream clock")
    stop = threading.Event(); chat = os.path.join(WORK, "chat.jsonl")
    cp = start_chat(src.get("platform"), src.get("handle", ""), url, chat, stop)
    heartbeat(sid, status="capturing")
    made, last_scan = [], 0
    try:
        while rec.poll() is None and (time.time() - t0) < max_minutes * 60:
            time.sleep(240)
            pieces = sorted(glob.glob(os.path.join(pdir, "p*.ts")))[:-1]  # the last one is still being written
            if len(pieces) < 3:
                heartbeat(sid); continue
            wav = os.path.join(WORK, "a.wav"); pcm_to_wav(pcm, wav)
            total, why, n = score(wav, chat_file(chat), t0)
            new = range(max(last_scan, 90), n - 25)
            hour_cap = MAX_PER_HOUR * max(1, n / 3600)
            for t in sorted(new, key=lambda i: -total[i]):
                if total[t] < MIN_SCORE or len(made) >= hour_cap:
                    break
                if any(abs(t - m) < 90 for m in made):
                    continue
                a, b = max(0, t - 40), min(n - 1, t + 25)
                clip_src = os.path.join(WORK, f"src_{t}.mp4")
                try:
                    cut(pdir, a + gap, b + gap, clip_src)
                    if make_clip(row, clip_src, t - a, t, float(total[t]), [k for k, s in why.items() if s[t] > 1.5] or ["audio"], capture_start=a + gap):
                        made.append(t)
                except Exception as e:  # noqa: BLE001
                    log(f"  clip at {t}s failed: {e}")
            last_scan = max(last_scan, n - 25)
            heartbeat(sid, checked_to=n)
            # keep disk in check: drop pieces older than 15 minutes that can't be part of a new clip
            for p in pieces[: max(0, len(pieces) - 15)]:
                os.remove(p)
    finally:
        stop.set()
        if cp: cp.terminate()
        rec.terminate()
    still_live = (ytdlp_json(url) or {}).get("live_status") == "is_live"
    heartbeat(sid, status="live" if still_live else "done")
    log(f"done: {len(made)} clips, still live: {still_live}")


def run_vod(row, n_clips=10):
    sid, url = row["id"], row["url"]
    os.makedirs(WORK, exist_ok=True)
    heartbeat(sid, status="processing")
    meta = ytdlp_json(url) or {}
    rest("PATCH", f"clip_streams?id=eq.{sid}", {"title": meta.get("title")}, prefer="return=minimal"); row["title"] = meta.get("title")
    subprocess.run(["yt-dlp", "--no-warnings", "-q", "-f", "ba[ext=m4a]/ba", "-o", os.path.join(WORK, "vod.%(ext)s"), url], check=True)
    aud = glob.glob(os.path.join(WORK, "vod.*"))[0]; wav = os.path.join(WORK, "vod.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", aud, "-ac", "1", "-ar", "16000", wav], check=True)
    subprocess.run(["yt-dlp", "--no-warnings", "-q", "--skip-download", "--write-subs", "--sub-langs", "live_chat", "-o", os.path.join(WORK, "chat"), url])
    chat = (glob.glob(os.path.join(WORK, "chat*.json")) or [None])[0]
    json.dump(meta, open(os.path.join(WORK, "meta.json"), "w"))
    picks_file = os.path.join(WORK, "moments.json")
    args = ["--wav", wav, "--meta", os.path.join(WORK, "meta.json"), "--n", str(n_clips), "--out", picks_file]
    if chat:
        args += ["--chat", chat]
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "find_moments.py"), *args], check=True)
    for m in json.load(open(picks_file)):
        a, b = max(0, m["t"] - 40), m["t"] + 25
        src = os.path.join(WORK, f"src_{m['t']}.mp4")
        try:
            subprocess.run(["yt-dlp", "--no-warnings", "-q", "-f", "bv*[height<=1080]+ba/b[height<=1080]", "--download-sections", f"*{a}-{b}",
                            "--force-keyframes-at-cuts", "--merge-output-format", "mp4", "-o", src, url], check=True, timeout=900)
            make_clip(row, src, m["t"] - a, m["t"], m["score"], m["why"])
        except Exception as e:  # noqa: BLE001
            log(f"  moment {m['t']} failed: {e}")
    heartbeat(sid, status="done")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--stream", required=True); p.add_argument("--max-minutes", type=float, default=340)
    a = p.parse_args()
    row = stream_row(a.stream)
    meta = ytdlp_json(row["url"]) or {}
    if meta.get("title") and not row.get("title"):
        rest("PATCH", f"clip_streams?id=eq.{row['id']}", {"title": meta["title"]}, prefer="return=minimal"); row["title"] = meta["title"]
    try:
        if meta.get("live_status") == "is_live" or meta.get("is_live"):
            run_live(row, a.max_minutes)
        else:
            run_vod(row)
    except Exception as e:  # noqa: BLE001
        log("failed:", e)
        rest("PATCH", f"clip_streams?id=eq.{row['id']}", {"status": "failed", "error": str(e)[:900]}, prefer="return=minimal")
        raise

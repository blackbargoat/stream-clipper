#!/usr/bin/env python3
"""Clipora on this Mac: runs in the background (launchd), no one needs to start it.

Every 5 minutes:
  - reads the Clipping switch from the website (clip_settings.clipping_enabled)
  - if on: checks each creator you watch (one light request each); for whoever is live, starts a clipper
    (run-local.sh) unless one is already running for that stream
  - if off: stops any clippers it started
  - if YouTube shows the bot check, waits 30 minutes before asking again (never tries to get around it)
Keys come from .env.local next to this file (not in git).
"""
import json, os, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
for line in open(os.path.join(HERE, ".env.local")):
    if "=" in line and not line.startswith("#"):
        k, v = line.strip().split("=", 1)
        if k == "PATH": os.environ["PATH"] = v          # background services start with a bare PATH
        else: os.environ.setdefault(k, v)
sys.path.insert(0, os.path.join(HERE, "clipper"))
from common import log, rest  # noqa: E402

CHECK_EVERY = 300
BACKOFF = 1800
procs = {}           # stream id -> Popen of run-local.sh
blocked_until = 0.0


def setting(key, default=None):
    rows = rest("GET", f"clip_settings?key=eq.{key}&select=value") or []
    return rows[0]["value"] if rows else default


def live_url(src):
    """(url, title) if live, None if not, 'blocked' if YouTube shows the bot check."""
    if src["platform"] == "twitch":
        h = src["handle"].lstrip("@")
        r = subprocess.run(["yt-dlp", "--no-warnings", "-J", f"https://www.twitch.tv/{h}"], capture_output=True, text=True, timeout=120)
        if r.returncode == 0 and r.stdout.strip():
            d = json.loads(r.stdout)
            if d.get("is_live"): return f"https://www.twitch.tv/{h}#{time.strftime('%Y%m%d')}", d.get("description") or d.get("title")
        return None
    h = src["handle"] if src["handle"].startswith("@") else "@" + src["handle"]
    r = subprocess.run(["yt-dlp", "--no-warnings", "-J", f"https://www.youtube.com/{h}/live"], capture_output=True, text=True, timeout=120)
    if "confirm you" in r.stderr and "not a bot" in r.stderr:
        return "blocked"
    if r.returncode == 0 and r.stdout.strip():
        d = json.loads(r.stdout)
        if d.get("live_status") == "is_live": return f"https://www.youtube.com/watch?v={d['id']}", d.get("title")
    return None


def stop_all():
    for sid, p in list(procs.items()):
        if p.poll() is None:
            try: os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError: pass
        procs.pop(sid, None)
    subprocess.run(["pkill", "-f", "ffmpeg.*copyts"], capture_output=True)


def tick():
    global blocked_until
    if setting("clipping_enabled", False) is not True:
        if procs: log("clipping switched off: stopping clippers"); stop_all()
        return
    for sid, p in list(procs.items()):          # forget clippers that finished (stream ended)
        if p.poll() is not None: procs.pop(sid, None)
    if time.time() < blocked_until:
        return
    for src in rest("GET", "clip_sources?enabled=eq.true&select=*") or []:
        res = live_url(src)
        if res == "blocked":
            blocked_until = time.time() + BACKOFF; log("YouTube bot check: waiting 30 min"); return
        if not res:
            continue
        url, title = res
        rest("POST", "clip_streams?on_conflict=url", [{"source_id": src["id"], "url": url, "title": title, "status": "live"}],
             prefer="resolution=ignore-duplicates,return=minimal")
        row = (rest("GET", f"clip_streams?url=eq.{url.replace('#', '%23')}&select=id,status") or [None])[0]
        if not row or row["id"] in procs:
            continue
        rest("PATCH", f"clip_streams?id=eq.{row['id']}", {"status": "capturing", "error": None}, prefer="return=minimal")
        logf = open(os.path.expanduser(f"~/Library/Logs/clipora-{row['id'][:8]}.log"), "a")
        procs[row["id"]] = subprocess.Popen(["/bin/bash", os.path.join(HERE, "run-local.sh"), row["id"]], stdout=logf, stderr=logf,
                                            start_new_session=True, env=os.environ.copy())
        log(f"{src['name']} is live: clipping {url}")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: (stop_all(), sys.exit(0)))
    log("clipora daemon started")
    while True:
        try:
            tick()
        except Exception as e:  # noqa: BLE001
            log("tick failed:", e)
        time.sleep(CHECK_EVERY)

#!/usr/bin/env python3
"""Every few minutes: is any watched creator live? Then make sure a capture job is running for each live stream
and for every link pasted on the website (status 'queued').

  SUPABASE_SERVICE_KEY=... GH_TOKEN=... python3 watch.py
Starts .github/workflows/capture.yml with the stream id (one job per stream; jobs hand over every ~5.5 h).
"""
import os, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import log, rest, ytdlp_json  # noqa: E402

REPO = os.environ.get("GITHUB_REPOSITORY", "blackbargoat/stream-clipper")


def live_url(src):
    if src["platform"] == "youtube":
        h = src["handle"] if src["handle"].startswith("@") else "@" + src["handle"]
        d = ytdlp_json(f"https://www.youtube.com/{h}/live")
        if d and d.get("live_status") == "is_live":
            return f"https://www.youtube.com/watch?v={d['id']}", d.get("title")
    else:
        d = ytdlp_json(f"https://www.twitch.tv/{src['handle']}")
        if d and d.get("is_live"):
            return f"https://www.twitch.tv/{src['handle']}#{time.strftime('%Y%m%d')}", d.get("description") or d.get("title")
    return None, None


def dispatch(stream_id):
    r = subprocess.run(["gh", "workflow", "run", "capture.yml", "-R", REPO, "-f", f"stream={stream_id}"], capture_output=True, text=True)
    log("start capture", stream_id, r.returncode, r.stderr.strip()[:200])
    return r.returncode == 0


def main():
    for src in rest("GET", "clip_sources?enabled=eq.true&select=*"):
        try:
            url, title = live_url(src)
        except Exception as e:  # noqa: BLE001
            log(src["name"], "check failed:", e); continue
        log(src["name"], "LIVE" if url else "offline")
        if url:
            rest("POST", "clip_streams?on_conflict=url", [{"source_id": src["id"], "url": url, "title": title, "status": "live"}],
                 prefer="resolution=ignore-duplicates,return=minimal")
    stale = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 15 * 60))
    todo = rest("GET", f"clip_streams?select=id,url,status&or=(status.in.(queued,live),and(status.eq.capturing,runner_seen_at.lt.{stale}))")
    for s in todo:
        if dispatch(s["id"]):
            # mark it taken so the next check doesn't start a second job; the job's heartbeat keeps it fresh
            rest("PATCH", f"clip_streams?id=eq.{s['id']}", {"status": "capturing", "runner_seen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                 prefer="return=minimal")


if __name__ == "__main__":
    main()

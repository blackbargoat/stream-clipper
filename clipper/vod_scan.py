#!/usr/bin/env python3
"""Find each watched creator's recent finished streams (VODs) on YouTube and clip their best moments.

VODs have signals a live stream doesn't: YouTube's "most replayed" heatmap and the full chat replay, so the
moments are usually much stronger than what live scoring catches.

  SUPABASE_SERVICE_KEY=... CLIP_WORKER_KEY=... python3 vod_scan.py [--per-creator 2] [--clips 8] [--max-hours 14]
Skips VODs already in clip_streams. Needs a normal (home) connection: YouTube blocks data-center servers.
"""
import argparse, json, os, re, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import log, rest  # noqa: E402


# Only GTA 6 content for now (the channel's focus while GTA 6 is out). Matches "GTA 6", "GTA6", "GTA VI", "Grand Theft Auto 6/VI".
TOPIC = re.compile(r"\bgta\s*(6|vi)\b|grand\s+theft\s+auto\s*(6|vi)\b", re.I)


def recent_vods(handle, n):
    h = handle if handle.startswith("@") else "@" + handle
    out = subprocess.run(["yt-dlp", "--no-warnings", "--flat-playlist", "-J", "--playlist-end", str(max(15, n * 6)), f"https://www.youtube.com/{h}/streams"],
                         capture_output=True, text=True, timeout=180)
    if out.returncode != 0 or not out.stdout.strip():
        log(handle, "couldn't list streams:", out.stderr.strip()[-200:]); return []
    vods = []
    for e in json.loads(out.stdout).get("entries") or []:
        if e.get("live_status") in ("is_live", "is_upcoming"):
            continue  # live ones are handled by the live clipper
        if (e.get("duration") or 0) < 20 * 60:
            continue
        if not TOPIC.search(e.get("title") or ""):
            continue  # not GTA 6
        vods.append({"url": f"https://www.youtube.com/watch?v={e['id']}", "title": e.get("title"), "duration": e.get("duration")})
        if len(vods) >= n:
            break
    return vods


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-creator", type=int, default=2); p.add_argument("--clips", type=int, default=8)
    p.add_argument("--max-hours", type=float, default=14)
    a = p.parse_args()
    import live  # noqa: E402  (imports render + whisper; only needed once there is work)
    seen = {r["url"] for r in rest("GET", "clip_streams?select=url")}
    for src in rest("GET", "clip_sources?enabled=eq.true&platform=eq.youtube&select=*"):
        for v in recent_vods(src["handle"], a.per_creator):
            if v["url"] in seen:
                log(src["name"], "already clipped:", (v["title"] or "")[:60]); continue
            if (v["duration"] or 0) > a.max_hours * 3600:
                log(src["name"], "too long, skipped:", v["title"]); continue
            row = rest("POST", "clip_streams", [{"source_id": src["id"], "url": v["url"], "title": v["title"], "status": "queued"}])[0]
            row["clip_sources"] = src
            seen.add(v["url"])
            log(f"VOD {src['name']}: {v['title']} ({(v['duration'] or 0) / 3600:.1f} h)")
            os.environ["CLIP_WORK"] = os.path.join(os.environ.get("CLIP_WORK_ROOT", "/tmp/clipora"), row["id"])
            live.WORK = os.environ["CLIP_WORK"]
            try:
                live.run_vod(row, n_clips=a.clips)
            except Exception as e:  # noqa: BLE001
                log("  failed:", e)
                rest("PATCH", f"clip_streams?id=eq.{row['id']}", {"status": "failed", "error": str(e)[:900]}, prefer="return=minimal")


if __name__ == "__main__":
    main()

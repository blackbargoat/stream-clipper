"""Per-clip lip-sync audit for YouTube lives, measured against YouTube's own copy (not guessed).

YouTube numbers every 2-second piece of a live (`.../sq/N/...` in the HLS piece URLs) and keeps serving old pieces,
so for any clip we can fetch the matching stretch, join video and audio on YouTube's own timestamps (an in-sync
reference), and measure our clip against it with sync_ref. If the voice is off by more than TOLERANCE, the source is
repaired by the measured amount and measured again. The result is stored with the clip.
"""
import os, re, subprocess, urllib.request
from concurrent.futures import ThreadPoolExecutor

import sync_ref

TOLERANCE = 0.045  # seconds; people notice lip sync errors from roughly 45 ms


def _first_piece(playlist_url):
    txt = urllib.request.urlopen(playlist_url, timeout=30).read().decode()
    lines = [l for l in txt.splitlines() if l and not l.startswith("#")]
    seq = int(re.search(r"#EXT-X-MEDIA-SEQUENCE:(\d+)", txt).group(1))
    return lines[-1], seq + len(lines) - 1  # newest piece URL and its number


def live_edge(feed):
    """Called when recording starts: remembers the piece-URL templates and the newest piece number."""
    v_url, sq = _first_piece(feed[0])
    a_url, _ = _first_piece(feed[1])
    return {"v": v_url, "a": a_url, "sq": sq}


def _id3_ts(path):
    b = open(path, "rb").read(256)
    k = b.find(b"com.apple.streaming.transportStreamTimestamp\x00")
    return (int.from_bytes(b[k + 45:k + 53], "big") & ((1 << 33) - 1)) / 90000 if k >= 0 else None


def reference(edge, sq_from, sq_to, work):
    os.makedirs(work, exist_ok=True)
    def get(kind, i):
        p = os.path.join(work, f"{kind}{i}.ts")
        url = re.sub(r"sq/\d+", f"sq/{i}", edge[kind])
        urllib.request.urlretrieve(url, p)
        return p
    rng = range(sq_from, sq_to + 1)
    with ThreadPoolExecutor(16) as ex:
        vs = list(ex.map(lambda i: get("v", i), rng)); as_ = list(ex.map(lambda i: get("a", i), rng))
    vcat, acat, ref = (os.path.join(work, n) for n in ("v.ts", "a.aac", "ref.mp4"))
    with open(vcat, "wb") as o:
        for p in vs: o.write(open(p, "rb").read())
    with open(acat, "wb") as o:
        for p in as_: o.write(open(p, "rb").read())
    v0 = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=start_time", "-of", "csv=p=0", vs[0]],
                              capture_output=True, text=True).stdout.split()[0])
    a0 = _id3_ts(as_[0]) or v0
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", vcat, "-itsoffset", f"{a0 - v0:.4f}", "-f", "aac", "-i", acat,
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", ref], check=True)
    return ref


def repair(src, early, dst):
    """Voice `early` seconds ahead of the picture -> drop that much picture from the start so both line up."""
    if early >= 0:
        args = ["-ss", f"{early:.3f}", "-i", src, "-i", src]
    else:
        args = ["-i", src, "-ss", f"{-early:.3f}", "-i", src]
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args, "-map", "0:v", "-map", "1:a", "-shortest",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-c:a", "aac", "-b:a", "192k", dst], check=True)


def audit(src, edge, capture_start, work):
    """src = clip source cut from capture second `capture_start`. Returns (src_to_use, trim_seconds, report)."""
    sq_mid = edge["sq"] + int(capture_start // 2)
    ref = reference(edge, sq_mid - 45, sq_mid + 75, work)
    m = sync_ref.measure(src, ref)
    report = {"measured_early": m["audio_early"], "match": min(m["video_match"], m["audio_match"])}
    if report["match"] < 0.8:
        report["status"] = "unverified"  # couldn't find the clip in YouTube's copy
        return src, 0.0, report
    if abs(m["audio_early"]) <= TOLERANCE:
        report["status"] = "in sync"
        return src, 0.0, report
    fixed = src.replace(".mp4", "_synced.mp4")
    repair(src, m["audio_early"], fixed)
    m2 = sync_ref.measure(fixed, ref)
    report.update(after=m2["audio_early"], status="repaired" if abs(m2["audio_early"]) <= TOLERANCE else "still off")
    return fixed, max(0.0, m["audio_early"]), report

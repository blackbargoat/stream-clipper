"""Shared helpers: Supabase REST/storage with the service key, timing, logging."""
import json, os, subprocess, time, urllib.request

SB = "https://etdbjahwegsmuelcawgf.supabase.co"
KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
H = {"apikey": KEY, "Authorization": f"Bearer {KEY}"}


def log(*a):
    print(time.strftime("%H:%M:%S"), "›", *a, flush=True)


def req(method, url, body=None, headers=None, raw=None, timeout=120):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    hd = {**H, **(headers or {})}
    if body is not None and raw is None:
        hd.setdefault("content-type", "application/json")
    r = urllib.request.Request(url, data=data, method=method, headers=hd)
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        t = resp.read()
        return json.loads(t) if t and resp.headers.get("content-type", "").startswith("application/json") else t


def rest(method, path, body=None, prefer="return=representation"):
    return req(method, f"{SB}/rest/v1/{path}", body, {"Prefer": prefer})


def upload(path, key, ctype):
    with open(path, "rb") as f:
        req("POST", f"{SB}/storage/v1/object/clips/{key}", raw=f.read(), headers={"content-type": ctype, "x-upsert": "true"}, timeout=300)
    return f"{SB}/storage/v1/object/public/clips/{key}"


def title_for(transcript, creator, stream_title):
    """Short hook for the title box, written by Gemini through the 'clips' edge function. Falls back to a plain one."""
    try:
        r = req("POST", f"{SB}/functions/v1/clips/title", {"transcript": transcript[:2000], "creator": creator, "stream": stream_title or ""},
                headers={"x-worker-key": os.environ.get("CLIP_WORKER_KEY", "")}, timeout=40)
        t = (r or {}).get("title", "").strip()
        if t:
            return t
    except Exception as e:  # noqa: BLE001
        log("title fallback:", e)
    return f"{creator.upper()} 😭"


def ytdlp_json(url):
    out = subprocess.run(["yt-dlp", "--no-warnings", "-J", url], capture_output=True, text=True, timeout=120)
    return json.loads(out.stdout) if out.returncode == 0 and out.stdout.strip() else None

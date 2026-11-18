#!/bin/bash
# Clip one live stream on this Mac until it ends. Restarts the recorder if the connection drops.
# usage: SUPABASE_SERVICE_KEY=... CLIP_WORKER_KEY=... ./run-local.sh <clip_streams id>
cd "$(dirname "$0")/clipper"
SID="$1"; export CLIP_WORK="/tmp/clipora/$SID"
while true; do
  mkdir -p "$CLIP_WORK"
  caffeinate -i python3 live.py --stream "$SID" --max-minutes 330
  ST=$(curl -s "https://etdbjahwegsmuelcawgf.supabase.co/rest/v1/clip_streams?id=eq.$SID&select=status" -H "apikey: $SUPABASE_SERVICE_KEY" -H "Authorization: Bearer $SUPABASE_SERVICE_KEY")
  echo "$(date +%H:%M) stream status: $ST"
  case "$ST" in *done*) break;; esac
  sleep 20
done

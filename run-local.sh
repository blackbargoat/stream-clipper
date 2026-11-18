#!/bin/bash
# Clip one live stream on this Mac until it ends. Restarts the recorder if the connection drops.
# usage: SUPABASE_SERVICE_KEY=... CLIP_WORKER_KEY=... ./run-local.sh <clip_streams id>
cd "$(dirname "$0")/clipper"
SID="$1"; export CLIP_WORK="/tmp/clipora/$SID"
# stopping this script stops everything it started (recorder, chat reader) instead of leaving orphans
trap 'pkill -TERM -P $$; pkill -f "copyts.*$CLIP_WORK"; exit 0' INT TERM
while true; do
  rm -rf "$CLIP_WORK/pieces" "$CLIP_WORK/audio.pcm"; mkdir -p "$CLIP_WORK"  # fresh recording each start
  caffeinate -i python3 live.py --stream "$SID" --max-minutes 330
  ST=$(curl -s "https://etdbjahwegsmuelcawgf.supabase.co/rest/v1/clip_streams?id=eq.$SID&select=status" -H "apikey: $SUPABASE_SERVICE_KEY" -H "Authorization: Bearer $SUPABASE_SERVICE_KEY")
  echo "$(date +%H:%M) stream status: $ST"
  case "$ST" in *done*) break;; esac
  sleep 20
done

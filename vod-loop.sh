#!/bin/bash
# Every 30 min: look for creators' finished GTA 6 streams and clip them (YouTube needs this home connection).
cd "$(dirname "$0")/clipper"
while true; do
  caffeinate -i python3 vod_scan.py --per-creator 2 --clips 8
  sleep 1800
done

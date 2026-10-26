#!/usr/bin/env bash
# Independent backstop: deletes NAME after SECS no matter what the launcher does.
# brev delete has no --yes flag; names are positional. Retry because the API rate-limits.
NAME="$1"; SECS="$2"
sleep "$SECS"
for _ in 1 2 3 4 5 6; do
  ~/.local/bin/brev delete "$NAME" </dev/null >>"$3" 2>&1 && { echo "backstop deleted $NAME" >>"$3"; exit 0; }
  sleep 60
done
echo "BACKSTOP FAILED for $NAME -- delete by hand" >>"$3"

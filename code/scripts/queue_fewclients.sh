#!/usr/bin/env bash
# Wait for the fresh-review orchestrator to finish, then launch fewclients.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
while pgrep -f "run_fresh_review_fixes" >/dev/null; do
    sleep 300
done
echo "[$(date '+%F %T')] fresh_review chain done; starting fewclients" >> runs/_remaining/orchestrator.log
exec bash scripts/run_fewclients.sh

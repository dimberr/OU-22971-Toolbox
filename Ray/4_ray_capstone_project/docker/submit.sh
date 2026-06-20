#!/usr/bin/env bash
# Submit a replay run to the Dockerized Ray cluster via `ray job submit`.
#
# Usage (from the host):
#   docker/submit.sh <mode> [extra run.py args...]
#   docker/submit.sh blocking --max-ticks 20 --slow-zone-sleep-s 1.0
#   docker/submit.sh async    --max-ticks 20 --slow-zone-sleep-s 1.0
#   docker/submit.sh stress   --max-ticks 20
#
# The job runs on the cluster; --output-dir is absolute (/app/runs/<mode>) so
# artifacts land in the bind-mounted project folder on the host.
set -euo pipefail

MODE="${1:?usage: [RUN_LABEL=name] submit.sh <blocking|async|stress> [extra args]}"
shift

# Output dir defaults to the mode; override with RUN_LABEL to keep comparison
# runs (e.g. blocking under harsh skew) in separate folders.
LABEL="${RUN_LABEL:-$MODE}"

COMPOSE="docker compose -f $(dirname "$0")/docker-compose.yml"

# The head's job-submission API (port 8265) takes a few seconds after the
# cluster starts. Wait for it so the first submit doesn't hit "connection
# refused".
echo "waiting for the Ray job server on the head..."
$COMPOSE exec -T ray-head bash -c \
  'until ray job list --address http://localhost:8265 >/dev/null 2>&1; do sleep 2; done'

# --working-dir ships the code to the cluster (entrypoint runs in a temp dir),
# so --prepared-dir / --output-dir use absolute /app paths (the bind mount) to
# read prepared assets and persist artifacts back to the host.
$COMPOSE exec -T ray-head \
  ray job submit --address http://localhost:8265 --working-dir /app -- \
  python main.py run \
    --prepared-dir /app/prepared \
    --output-dir "/app/runs/${LABEL}" \
    --mode "${MODE}" "$@"

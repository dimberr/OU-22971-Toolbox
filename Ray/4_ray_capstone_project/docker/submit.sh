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

MODE="${1:?usage: submit.sh <blocking|async|stress> [extra args]}"
shift

COMPOSE="docker compose -f $(dirname "$0")/docker-compose.yml"

# --working-dir ships the code to the cluster (entrypoint runs in a temp dir),
# so --prepared-dir / --output-dir use absolute /app paths (the bind mount) to
# read prepared assets and persist artifacts back to the host.
$COMPOSE exec -T ray-head \
  ray job submit --address http://localhost:8265 --working-dir /app -- \
  python main.py run \
    --prepared-dir /app/prepared \
    --output-dir "/app/runs/${MODE}" \
    --mode "${MODE}" "$@"

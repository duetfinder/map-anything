#!/usr/bin/env bash

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <scene_bundle.pt> [public_host] [web_port] [grpc_port] [bind_host]"
    exit 1
fi

BUNDLE_PATH="$1"
PUBLIC_HOST="${2:-127.0.0.1}"
WEB_PORT="${3:-9011}"
GRPC_PORT="${4:-9876}"
BIND_HOST="${5:-0.0.0.0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RRD_PATH="${BUNDLE_PATH%.pt}.rrd"
ENCODED_PROXY_URL="rerun%2Bhttp%3A%2F%2F${PUBLIC_HOST}%3A${GRPC_PORT}%2Fproxy"
FULL_BROWSER_URL="http://${PUBLIC_HOST}:${WEB_PORT}/?url=${ENCODED_PROXY_URL}"

cd "${REPO_ROOT}"

python scripts/visualize_rs_guided_scene.py "${BUNDLE_PATH}" --save "${RRD_PATH}"

echo
echo "Starting Rerun web viewer..."
echo "RRD: ${RRD_PATH}"
echo "Web port: ${WEB_PORT}"
echo "gRPC port: ${GRPC_PORT}"
echo "Bind host: ${BIND_HOST}"
echo
echo "Open in browser:"
echo "  ${FULL_BROWSER_URL}"
echo
echo "If you use SSH forwarding, forward both ports:"
echo "  ssh -L ${WEB_PORT}:127.0.0.1:${WEB_PORT} -L ${GRPC_PORT}:127.0.0.1:${GRPC_PORT} <user>@<server>"
echo

exec rerun "${RRD_PATH}" \
  --web-viewer \
  --bind "${BIND_HOST}" \
  --web-viewer-port "${WEB_PORT}" \
  --port "${GRPC_PORT}"

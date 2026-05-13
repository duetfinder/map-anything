#!/usr/bin/env bash

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <scene_bundle.pt|scene_dir> [public_host] [web_port] [grpc_port] [bind_host]"
    exit 1
fi

BUNDLE_PATH="$1"
PUBLIC_HOST="${2:-127.0.0.1}"
WEB_PORT="${3:-9047}"
GRPC_PORT="${4:-9877}"
BIND_HOST="${5:-0.0.0.0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -d "${BUNDLE_PATH}" ]; then
    RRD_PATH="${BUNDLE_PATH%/}/rs_guided_collection.rrd"
else
    RRD_PATH="${BUNDLE_PATH%.pt}.rrd"
fi
ENCODED_PROXY_URL="rerun%2Bhttp%3A%2F%2F${PUBLIC_HOST}%3A${GRPC_PORT}%2Fproxy"
FULL_BROWSER_URL="http://${PUBLIC_HOST}:${WEB_PORT}/?url=${ENCODED_PROXY_URL}"

find_listeners() {
    local port="$1"
    lsof -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null | sort -u || true
}

print_kill_hint() {
    local pid="$1"
    local pgid
    pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]' || true)"
    if [ -n "${pgid}" ]; then
        echo "  Stop process group: kill -TERM -${pgid}"
        echo "  Force stop if needed: kill -KILL -${pgid}"
    else
        echo "  Stop process: kill ${pid}"
    fi
}

check_port_available() {
    local port="$1"
    local label="$2"
    local pids
    pids="$(find_listeners "${port}")"
    if [ -z "${pids}" ]; then
        return 0
    fi

    echo "Port ${port} (${label}) is already in use."
    while IFS= read -r pid; do
        [ -z "${pid}" ] && continue
        ps -fp "${pid}" || true
        print_kill_hint "${pid}"
    done <<< "${pids}"
    echo
    echo "Use different ports, or stop the old viewer process first."
    exit 1
}

cd "${REPO_ROOT}"

check_port_available "${WEB_PORT}" "web viewer"
check_port_available "${GRPC_PORT}" "gRPC proxy"

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

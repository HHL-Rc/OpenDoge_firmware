#!/usr/bin/env bash
# Set the capabilities opendoge_deploy needs when run as a normal user:
#   cap_sys_nice  -> SCHED_FIFO / CPU affinity (--realtime --cpu N)
#   cap_net_raw   -> open AF_CAN raw sockets for can0-can3
#
# Caps are stored on the file inode and are LOST every time colcon relinks
# the binary. Re-run this after every build.
#
# Usage:
#   bash scripts/setup_deploy_caps.sh
#   bash scripts/setup_deploy_caps.sh install_aarch64/opendoge_deploy/bin/opendoge_deploy
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${1:-}"

if [ -z "$BIN" ]; then
  for cand in \
    "${INSTALL_BASE:-}/opendoge_deploy/bin/opendoge_deploy" \
    "$ROOT/install/opendoge_deploy/bin/opendoge_deploy" \
    "$ROOT/install_aarch64/opendoge_deploy/bin/opendoge_deploy"; do
    if [ -n "$cand" ] && [ -x "$cand" ]; then
      BIN="$cand"
      break
    fi
  done
fi

if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
  echo "opendoge_deploy binary not found." >&2
  echo "Build first (x86: install/, Orange Pi: install_aarch64/), or pass the path:" >&2
  echo "  bash $0 install_aarch64/opendoge_deploy/bin/opendoge_deploy" >&2
  exit 1
fi

if ! command -v setcap >/dev/null 2>&1; then
  echo "setcap not found; install libcap2-bin: sudo apt install -y libcap2-bin" >&2
  exit 1
fi

BIN="$(readlink -f "$BIN")"
sudo setcap cap_sys_nice,cap_net_raw+ep "$BIN"
echo "capabilities set on: $BIN"
if command -v getcap >/dev/null 2>&1; then
  getcap "$BIN"
fi
echo "note: re-run this after every colcon build (relink drops file caps)."

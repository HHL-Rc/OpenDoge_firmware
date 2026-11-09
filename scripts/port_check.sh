#!/usr/bin/env bash
# Read-only Orange Pi / porting checklist.
# Does not enable motors, write config, or change services.
#
# Usage: bash scripts/port_check.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAIL=0
WARN=0

section() { echo; echo "=== $*"; }
ok()      { echo "  [OK]   $*"; }
warn()    { echo "  [WARN] $*"; WARN=$((WARN + 1)); }
fail()    { echo "  [FAIL] $*"; FAIL=$((FAIL + 1)); }

section "System"
echo "  arch:    $(uname -m)"
if [ "$(uname -m)" = "aarch64" ]; then ok "aarch64"; else warn "not aarch64 (is this the Orange Pi?)"; fi
if command -v lsb_release >/dev/null 2>&1; then echo "  os:      $(lsb_release -ds 2>/dev/null)"; fi
echo "  repo:    $ROOT"
echo "  user:    $(id -un)  groups=$(id -Gn)"

section "Required commands"
for cmd in ip udevadm loginctl python3 setcap getcap lsusb rsync; do
  if command -v "$cmd" >/dev/null 2>&1; then ok "$cmd"; else warn "missing $cmd"; fi
done

section "deploy binary"
DEPLOY_BIN="${DEPLOY_BIN:-}"
if [ -z "$DEPLOY_BIN" ]; then
  for cand in "$ROOT/install/opendoge_deploy/bin/opendoge_deploy" \
              "$ROOT/install_aarch64/opendoge_deploy/bin/opendoge_deploy"; do
    [ -x "$cand" ] && DEPLOY_BIN="$cand" && break
  done
fi
if [ -n "$DEPLOY_BIN" ] && [ -x "$DEPLOY_BIN" ]; then
  ok "found $DEPLOY_BIN"
  file "$DEPLOY_BIN" 2>/dev/null | sed 's/^/         /'
  if command -v getcap >/dev/null 2>&1; then
    caps="$(getcap "$DEPLOY_BIN" 2>/dev/null || true)"
    echo "  caps:   ${caps:-none}"
    case "$caps" in
      *cap_sys_nice*) ok "cap_sys_nice";; *) warn "cap_sys_nice missing (run scripts/setup_deploy_caps.sh)";;
    esac
    case "$caps" in
      *cap_net_raw*) ok "cap_net_raw";; *) warn "cap_net_raw missing (run scripts/setup_deploy_caps.sh)";;
    esac
  fi
  if command -v ldd >/dev/null 2>&1; then
    if ldd "$DEPLOY_BIN" 2>/dev/null | grep -q onnxruntime; then ok "onnxruntime linked"; else warn "onnxruntime not linked"; fi
  fi
else
  warn "opendoge_deploy not built (install/ or install_aarch64/)"
fi

section "USB devices"
if command -v lsusb >/dev/null 2>&1; then
  lsusb 2>/dev/null | grep -Ei '1d50:606f|6877:4d55|3537:1041|3537:2106' || warn "no CHIWU/IMU/gamepad USB IDs found"
else
  warn "lsusb not available (install usbutils)"
fi

section "CAN"
if command -v ip >/dev/null 2>&1; then
  ip -br link show 2>/dev/null | grep -E 'can[0-3]' || warn "can0-3 not found"
fi
if [ -x "$ROOT/scripts/prepare_can_interfaces.sh" ]; then
  echo "  prepare_can_interfaces --dry-run:"
  CAN_RENAME_TIMEOUT=3 bash "$ROOT/scripts/prepare_can_interfaces.sh" --dry-run 2>&1 | sed 's/^/    /' || true
fi
if [ -f "$ROOT/scripts/verify_can_mapping.py" ]; then
  echo "  verify_can_mapping --interfaces-only:"
  python3 "$ROOT/scripts/verify_can_mapping.py" --interfaces-only 2>&1 | sed 's/^/    /' || true
fi

section "IMU / gamepad permissions"
for dev in /dev/ttyACM* /dev/ttyUSB*; do
  [ -e "$dev" ] && ls -l "$dev"
done
for dev in /dev/input/js*; do
  [ -e "$dev" ] && ls -l "$dev"
done
if ! ls -l /dev/input/js* >/dev/null 2>&1; then warn "no /dev/input/js* (receiver mode?)"; fi
if ! ls -l /dev/ttyACM* /dev/ttyUSB* >/dev/null 2>&1; then warn "no IMU tty (/dev/ttyACM* or /dev/ttyUSB*)"; fi

section "systemd services"
for svc in opendoge-can opendoge-xboxdrv; do
  if systemctl is-active --quiet "$svc" 2>/dev/null; then ok "$svc active"; else warn "$svc not active"; fi
done
for svc in opendoge-imu opendoge-joystick; do
  if systemctl --user is-active --quiet "$svc" 2>/dev/null; then ok "$svc active (user)"; else warn "$svc not active (user)"; fi
done
if command -v loginctl >/dev/null 2>&1; then
  linger="$(loginctl show-user "$(id -un)" 2>/dev/null | grep -i '^Linger=' || true)"
  echo "  $linger"
  [ "$linger" = "Linger=yes" ] || warn "lingering off: sudo loginctl enable-linger $(id -un)"
fi

section "CPU / realtime hint"
echo "  nproc=$(nproc 2>/dev/null || echo '?')"
echo "  start_robot.sh defaults: REALTIME_ARGS='--realtime --cpu 0'"
echo "  RK3588: CPU0-3=A55, CPU4-7=A76; if max_late_us/missed_ctrl is bad, try --cpu 4."

echo
if [ "$FAIL" -eq 0 ]; then
  echo "port_check: no hard failures; $WARN warning(s)."
  exit 0
fi
echo "port_check: $FAIL hard failure(s), $WARN warning(s)."
exit 1

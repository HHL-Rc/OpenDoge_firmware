#!/usr/bin/env python3
"""monitor_status.py — 只读监控 deploy 的 JSON 状态，检测关节异常。

数据来源是 opendoge_deploy 每秒写的 /tmp/opendoge_status.json：
  - 不占用 CAN、不需要 root；
  - 可以和 deploy 同时运行。

检查项：
  - q 超出配置的 [lower, upper]（OUT，只在主动控制状态检查）
  - |q - target| 过大（TRACK，只在主动控制状态检查）
  - |dq| 过大（DQ，默认 >15 rad/s）
  - 关节侧力矩 |tau * reduction| 接近 max_torque（TAU）
  - 电机 status fault 位非 0（FAULT，并打印欠压/过温/过流等位名）
  - 反馈过旧（STALE，默认 >80 ms；JSON 需含 age_ms）
  - 该电机没有反馈（NOFDBK）
  - state / can_rx / fault_reason 变化

注意：OUT/TRACK 只在 entering_position/active_pc/active_rl/standing_up/
low_gain_test 这些“真的在跟踪目标”的状态下检查。damping_fault/ready 时
JSON 里的 target 只是默认姿态占位值，拿它和实际 q 比会误报。

用法:
    python3 hardware/motor/monitor_status.py
    python3 hardware/motor/monitor_status.py --rate-hz 2 --csv /tmp/opendoge_monitor.csv
    python3 hardware/motor/monitor_status.py --once

注意：JSON 是 1 Hz；快速瞬态（比如几十 ms 的过流）可能只留下最后一次快照。
需要更高频记录时，告诉我，我再给 deploy 加 20–50 Hz 的 debug CSV。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPT_DIR.parent.parent
DEFAULT_STATUS = Path("/tmp/opendoge_status.json")
DEFAULT_CONFIG = REPO_ROOT / "deploy" / "configs" / "opendoge_deploy.conf"

JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

# EL05 *status frame* fault field (CAN-ID bits 21:16, `kStatusFaultMask=0x3F`).
# This layout is NOT the same as the 0x3022 faultSta register used by
# el05_motor_menu.py::FAULT_NAMES. Manual §3.4:
#   bit5 未标定, bit4 堵转过载, bit3 磁编码故障,
#   bit2 过温, bit1 三相电流故障, bit0 欠压.
# So `fault=1` means UNDERVOLTAGE, not overcurrent (that would be 2).
STATUS_FAULT_NAMES = {
    0: "欠压",
    1: "三相电流故障",
    2: "过温",
    3: "磁编码故障",
    4: "堵转过载",
    5: "未标定",
}


# 0x3022 faultSta register layout (manual §3.3.7), different from the status
# frame field above. Keep in sync with el05_motor_menu.py::FAULT_NAMES.
FAULT_STA_NAMES = {
    0: "过温",
    1: "驱动芯片",
    2: "欠压",
    3: "过压",
    4: "B相过流",
    5: "C相过流",
    7: "编码器未标定",
    8: "硬件识别",
    9: "位置初始化",
    14: "堵转过载",
    16: "A相过流",
}


def fault_description(fault: int, fault_sta: int = 0) -> str:
    parts = []
    if fault:
        parts.append("status:" + ",".join(
            STATUS_FAULT_NAMES[b] for b in range(6) if fault & (1 << b)))
    if fault_sta:
        parts.append("faultSta:" + ",".join(
            FAULT_STA_NAMES.get(b, f"bit{b}")
            for b in range(32) if fault_sta & (1 << b)))
    return " | ".join(parts) if parts else "无"


def load_limits(path: Path) -> Dict[str, Dict[str, float]]:
    limits = {
        name: {"lower": -12.57, "upper": 12.57, "max_torque": 4.0, "reduction": 1.0}
        for name in JOINT_NAMES
    }
    if not path.exists():
        print(f"⚠️  配置不存在，使用默认限位: {path}")
        return limits
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parts = key.strip().split(".")
        if len(parts) != 3 or parts[0] != "joint":
            continue
        name, field = parts[1], parts[2]
        if name in limits and field in limits[name]:
            try:
                limits[name][field] = float(value.strip())
            except ValueError:
                pass
    return limits


def load_status(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# Only in these states is q actively driven toward `target`. In
# damping_fault/ready/wait_feedback the status `target` is just bookkeeping
# (the default pose), while the motors are actually damping at their measured
# position — comparing q to it would produce meaningless OUT/TRACK alarms.
ACTIVE_STATES = {
    "entering_position", "active_pc", "active_rl",
    "standing_up", "low_gain_test",
}


def check_joint(j: dict, lim: Dict[str, float], dq_warn: float,
                track_warn: float, state: str, stale_ms: float) -> list[str]:
    flags: list[str] = []
    q = float(j.get("q", 0.0))
    target = j.get("target")
    dq = float(j.get("dq", 0.0))
    tau_j = float(j.get("tau", 0.0)) * lim["reduction"]
    fault = int(j.get("fault", 0))
    fault_sta = int(j.get("fault_sta", 0))
    recv = bool(j.get("recv", True))
    age_ms = j.get("age_ms")
    if state in ACTIVE_STATES:
        if q < lim["lower"] - 0.05 or q > lim["upper"] + 0.05:
            flags.append("OUT")
        if target is not None and abs(q - float(target)) > track_warn:
            flags.append("TRACK")
    if abs(dq) > dq_warn:
        flags.append("DQ")
    if abs(tau_j) > 0.8 * max(lim["max_torque"], 1e-6):
        flags.append("TAU")
    if fault or fault_sta:
        flags.append("FAULT")
    if age_ms is not None and int(age_ms) >= 0 and int(age_ms) > stale_ms:
        flags.append("STALE")
    if not recv:
        flags.append("NOFDBK")
    return flags


def fmt(value, width: int, precision: int = 4) -> str:
    if value is None:
        return f"{'n/a':>{width}s}"
    return f"{float(value):+{width}.{precision}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rate-hz", type=float, default=2.0)
    parser.add_argument("--dq-warn", type=float, default=15.0, help="|dq| warning threshold (rad/s)")
    parser.add_argument("--tracking-warn", type=float, default=0.5, help="|q-target| warning threshold (rad)")
    parser.add_argument("--stale-ms", type=float, default=80.0, help="feedback age warning threshold (ms)")
    parser.add_argument("--max-age", type=float, default=3.0,
                        help="status file age before warning that deploy is not running (s)")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    limits = load_limits(args.config)
    period = 1.0 / max(args.rate_hz, 0.2)
    start_s = time.monotonic()
    last_state = None
    last_fault = None
    stale_file_warned = False
    csv_file = None
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = args.csv.open("a", encoding="utf-8", newline="")
        writer = csv.writer(csv_file)
        if csv_file.tell() == 0:
            writer.writerow(["t", "state", "joint", "q", "target", "dq", "tau",
                             "tau_j", "temp", "fault", "fault_sta", "age_ms",
                             "recv", "flags"])
    else:
        writer = None

    try:
        while True:
            data = load_status(args.status)
            if data is None:
                print(f"\r⚠️  读不到 {args.status} ...", end="", flush=True)
                if args.once:
                    return 1
                time.sleep(period)
                continue

            try:
                file_age_s = time.time() - args.status.stat().st_mtime
            except OSError:
                file_age_s = float("inf")
            file_is_stale = file_age_s > args.max_age
            if file_is_stale and not stale_file_warned:
                print(f"\n⏸  状态文件 {file_age_s:.1f}s 未更新：deploy 可能没在跑。"
                      f"下面是最后一次快照，不能代表电机当前在线状态。")
                stale_file_warned = True
            elif not file_is_stale:
                stale_file_warned = False

            state = data.get("state", "?")
            fault = data.get("fault_reason", "")
            imu = data.get("imu", {})
            if state != last_state or fault != last_fault:
                print(f"\n[state] {state}  can_rx={data.get('can_rx')}  "
                      f"can_tx={data.get('can_tx')}  err={data.get('can_err')}  "
                      f"late_us={data.get('max_late_us')}  fault={fault!r}  "
                      f"file_age={file_age_s:.1f}s")
                last_state, last_fault = state, fault

            print(f"\n t={time.monotonic() - start_s:7.2f}s  state={state}  "
                  f"active={data.get('active_cmd')}  rl={data.get('rl_inference')}  "
                  f"can_rx={data.get('can_rx')}  file_age={file_age_s:.1f}s  "
                  f"imu(wz={imu.get('wz', 0.0):+.3f}, "
                  f"gz={imu.get('gz', 0.0):+.3f})")
            print("  {:<16s} {:>9s} {:>9s} {:>9s} {:>9s} {:>9s} {:>6s} {:>10s} {:>8s} {:>5s} {}".format(
                "joint", "q", "target", "dq", "tau", "tau_j", "temp",
                "fault/fSta", "age_ms", "recv", "flags"))
            warnings = []
            for j in data.get("joints", []):
                name = j.get("n", "?")
                lim = limits.get(name, limits[JOINT_NAMES[0]])
                flags = check_joint(j, lim, args.dq_warn, args.tracking_warn,
                                    state, args.stale_ms)
                tau_j = float(j.get("tau", 0.0)) * lim["reduction"]
                joint_fault = int(j.get("fault", 0))
                joint_fault_sta = int(j.get("fault_sta", 0))
                fault_cell = f"{joint_fault}"
                if joint_fault_sta:
                    fault_cell += f"/{joint_fault_sta:X}"
                age = j.get("age_ms")
                age_cell = "n/a" if age is None else str(int(age))
                print("  {:<16s} {:>9s} {:>9s} {:>9s} {:>9s} {:>9s} {:>6s} {:>10s} {:>8s} {:>5s} {}".format(
                    name,
                    fmt(j.get("q"), 9),
                    fmt(j.get("target"), 9),
                    fmt(j.get("dq"), 9, 3),
                    fmt(j.get("tau"), 9, 3),
                    fmt(tau_j, 9, 3),
                    f"{float(j.get('temp', 0.0)):6.1f}",
                    fault_cell,
                    age_cell,
                    "yes" if j.get("recv") else "NO",
                    ",".join(flags),
                ))
                if flags:
                    detail = (f" ({fault_description(joint_fault, joint_fault_sta)})"
                              if (joint_fault or joint_fault_sta) else "")
                    warnings.append(f"{name}: {'/'.join(flags)}{detail}")
                if writer is not None:
                    writer.writerow([
                        f"{time.monotonic() - start_s:.3f}", state, name,
                        j.get("q"), j.get("target"), j.get("dq"), j.get("tau"),
                        f"{tau_j:.6f}", j.get("temp"), joint_fault,
                        joint_fault_sta, age, j.get("recv"), "|".join(flags),
                    ])
            if warnings:
                marker = "旧快照 " if file_is_stale else ""
                print(f"  ⚠️  {marker}[{state}] " + " | ".join(warnings))
            if writer is not None:
                csv_file.flush()
            if state == "damping_fault":
                print(f"  ❌ damping_fault: {fault}")
                if args.once:
                    return 2
            if args.once:
                return 0
            time.sleep(period)
    except KeyboardInterrupt:
        print("\n停止中...")
    finally:
        if csv_file is not None:
            csv_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Summarize OpenDoge high-rate recorder CSV logs.

The deploy status JSON is only 1 Hz and may miss a 50-120 ms CAN feedback
dropout. `hardware/motor/record_highrate.py` logs 100-200 Hz q/target/dq/tau,
status ages and fault bits. This script turns those large CSVs into a short
report: state transitions, fault reasons, flag counts, max torque/velocity,
max feedback/target ages and the worst events.

Usage:
  python3 test/analyze_highrate.py
  python3 test/analyze_highrate.py logs/highrate/highrate_20261107_134107.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from collections import Counter

JOINTS = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]


def num(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def parse_flags(text: str) -> list[str]:
    return [item.strip() for item in (text or "").split("|") if item.strip()]


def summarize(path: str) -> None:
    with open(path, newline="") as file:
        rows = list(csv.DictReader(file))
    print("=" * 100)
    print(f"{os.path.basename(path)}  rows={len(rows)}  columns={len(rows[0]) if rows else 0}")
    if not rows:
        return

    states = Counter(row.get("state", "?") for row in rows)
    print("states:", dict(states))

    faults: dict[str, list[float]] = {}
    for row in rows:
        reason = (row.get("fault_reason") or "").strip()
        if not reason:
            continue
        faults.setdefault(reason, []).append(num(row, "t"))
    if faults:
        print("fault_reason events:")
        for reason, times in sorted(faults.items(), key=lambda item: item[1][0]):
            print(f"  t={times[0]:8.3f}  n={len(times):5d}  {reason}")
    else:
        print("fault_reason events: none")

    flag_counts: Counter[str] = Counter()
    for row in rows:
        for flag in parse_flags(row.get("flags", "")):
            flag_counts[flag.split(":", 1)[0]] += 1
    print("flag counts:", dict(flag_counts))

    print("per-joint extremes:")
    print(f"  {'joint':<16} {'max|tau_j|':>10} {'max|dq|':>9} {'max err':>9} "
          f"{'max status age':>15} {'max target age':>15}")
    for joint in JOINTS:
        tau = max(abs(num(row, f"{joint}_tau_j")) for row in rows)
        dq = max(abs(num(row, f"{joint}_dq")) for row in rows)
        err = max(
            abs(num(row, f"{joint}_q") - num(row, f"{joint}_target"))
            for row in rows if row.get(f"{joint}_target") not in (None, "")
        )
        status_age = max(num(row, f"{joint}_status_age_ms", -1.0) for row in rows)
        target_age = max(num(row, f"{joint}_target_age_ms", -1.0) for row in rows)
        print(f"  {joint:<16} {tau:10.3f} {dq:9.2f} {err:9.3f} "
              f"{status_age:15.1f} {target_age:15.1f}")

    # Worst events: any flag or the largest torque.
    scored = []
    for row in rows:
        flags = parse_flags(row.get("flags", ""))
        tau = max(abs(num(row, f"{joint}_tau_j")) for joint in JOINTS)
        score = len(flags) * 1000.0 + tau
        if score > 0:
            scored.append((score, row, flags, tau))
    scored.sort(key=lambda item: item[0], reverse=True)
    print("worst events:")
    for _, row, flags, tau in scored[:10]:
        unique_flags = []
        for flag in flags:
            name = flag.split(":", 1)[0]
            if name not in unique_flags:
                unique_flags.append(name)
        print(f"  t={num(row,'t'):8.3f} state={row.get('state','?'):<14} "
              f"max|tau_j|={tau:6.3f} flags={','.join(unique_flags) or '-'}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths", nargs="*",
        help="CSV paths (default: all logs/highrate/*.csv)",
    )
    args = parser.parse_args()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = args.paths or sorted(glob.glob(os.path.join(repo, "logs", "highrate", "*.csv")))
    if not paths:
        print("no high-rate logs found", file=sys.stderr)
        return 1
    for path in paths:
        summarize(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

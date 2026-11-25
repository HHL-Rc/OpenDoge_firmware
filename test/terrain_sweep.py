#!/usr/bin/env python3
"""Run the deploy sim2sim over several terrains and summarize the logs.

Usage:
  python3 test/terrain_sweep.py --policy policy/opendoge_supermimic_rough_20000.onnx \
      --terrains flat,rough,stairs,boxes --duration 10 --cmd 0.4 0 0
"""

import argparse
import csv
import glob
import os
import subprocess
import sys
import tempfile


def summarize_csv(path: str) -> dict:
    with open(path, newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return {"rows": 0}
    z = [float(row["z"]) for row in rows]
    x = [float(row["base_x"]) for row in rows]
    y = [float(row["base_y"]) for row in rows]
    vx = [float(row["true_vx"]) for row in rows]
    return {
        "rows": len(rows),
        "z_min": min(z),
        "z_final": z[-1],
        "x_final": x[-1],
        "y_final": y[-1],
        "distance": (x[-1] ** 2 + y[-1] ** 2) ** 0.5,
        "mean_vx": sum(vx) / len(vx),
        "fault": max(int(row["fault"]) for row in rows),
    }


def main() -> int:
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument(
        "--terrains", default="flat,rough,stairs,boxes",
        help="comma separated: flat,rough,stairs,boxes,slope",
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--cmd", nargs=3, type=float, default=[0.4, 0.0, 0.0])
    parser.add_argument("--training-root", default=os.path.join(repo, "..", "OpenDoge_train"))
    parser.add_argument("--terrain-level", type=int, default=1)
    parser.add_argument("--terrain-seed", type=int, default=0)
    args = parser.parse_args()

    sim = os.path.join(repo, "sim2sim", "deploy_mujoco.py")
    header = f"{'terrain':<8} {'rows':>5} {'z_min':>7} {'z_final':>8} {'distance':>9} {'mean_vx':>8} {'fault':>6}"
    print(header)
    print("-" * len(header))
    for terrain in [item.strip() for item in args.terrains.split(",") if item.strip()]:
        log_dir = tempfile.mkdtemp(prefix=f"opendoge_sweep_{terrain}_")
        command = [
            sys.executable, sim,
            "--training-root", args.training_root,
            "--terrain", terrain,
            "--terrain-level", str(args.terrain_level),
            "--terrain-seed", str(args.terrain_seed),
            "--start-pose", "default",
            "--mode", "rl",
            "--policy", args.policy,
            "--cmd", *[str(value) for value in args.cmd],
            "--duration", str(args.duration),
            "--no-render",
            "--log-dir", log_dir,
        ]
        completed = subprocess.run(
            command, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True,
        )
        logs = sorted(glob.glob(os.path.join(log_dir, "*.csv")))
        if completed.returncode != 0 or not logs:
            print(f"{terrain:<8} FAILED rc={completed.returncode}")
            print(completed.stdout[-800:])
            continue
        metrics = summarize_csv(logs[-1])
        print(
            f"{terrain:<8} {metrics['rows']:>5} {metrics['z_min']:>7.3f} "
            f"{metrics['z_final']:>8.3f} {metrics['distance']:>9.3f} "
            f"{metrics['mean_vx']:>8.3f} {metrics['fault']:>6}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

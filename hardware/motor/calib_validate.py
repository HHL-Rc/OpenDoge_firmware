#!/usr/bin/env python3
"""
OpenDoge 标定方向校验 — 用 ONNX 策略验证 motor direction 是否正确。

原理:
  正确的 direction → 观测中关节角符合物理 → 策略输出合理动作
  错误的 direction → 观测中关节角方向反了 → 策略输出极端动作试图"纠错"

用法:
  python3 hardware/motor/calib_validate.py                 # 验证当前 captures.json
  python3 hardware/motor/calib_validate.py --auto-flip     # 自动检测并翻转错误方向
"""

import argparse
import json
import struct
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# ── 常量 (与训练一致) ──
NUM_JOINTS = 12
OBS_DIM = 49
DEFAULT_POS = np.array([
    0.0, 0.5, -1.3,   # FL
    0.0, 0.5, -1.3,   # FR
    0.0, 0.7, -1.3,   # RL
    0.0, 0.7, -1.3,   # RR
])
ACTION_SCALE = 0.25

JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

# MuJoCo 仿真趴伏角 (物理参考)
MUJOCO_PRONE = np.array([
    0.0600, 0.7615, -2.2623,   # FL
    -0.0600, 0.7615, -2.2624,   # FR
    0.2610, 1.1358, -2.6275,   # RL
    -0.2610, 1.1358, -2.6275,   # RR
])

# Motor-to-joint reduction ratios (calf joints have 1.5:1 gear reduction)
REDUCTIONS = np.array([
    1.0, 1.0, 1.5,   # FL: hip, thigh, calf
    1.0, 1.0, 1.5,   # FR: hip, thigh, calf
    1.0, 1.0, 1.5,   # RL: hip, thigh, calf
    1.0, 1.0, 1.5,   # RR: hip, thigh, calf
])

LOG_DIR = Path(__file__).resolve().parent / "calibration_logs"
# Legacy: this tool still targets the old 49-dim opendoge_r5.onnx. It is NOT
# compatible with the current 47x5 SuperMimic policy; see
# docs/opendoge_supermimic_migration.md.
POLICY_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "policy" / "legacy_48dim" / "opendoge_r5.onnx"
)


def build_observation(
    motor_positions: np.ndarray,    # 12, raw mechPos
    directions: np.ndarray,          # 12, ±1
    offsets: np.ndarray,             # 12, rad
    gyro: np.ndarray = None,
    gravity: np.ndarray = None,
    commands: np.ndarray = None,
) -> np.ndarray:
    """Build 49-dim observation matching C++ buildObservation()."""
    if gyro is None:
        gyro = np.zeros(3)
    if gravity is None:
        gravity = np.array([0.0, 0.0, -1.0])  # upright
    if commands is None:
        commands = np.zeros(3)

    # logical position from motor mechPos (with reduction ratio)
    logical_pos = directions * (motor_positions / REDUCTIONS - offsets)
    dof_pos_diff = logical_pos - DEFAULT_POS
    dof_vel = np.zeros(NUM_JOINTS)  # assume stationary
    last_action = np.zeros(NUM_JOINTS)
    feet_phase = np.zeros(4)

    obs = np.concatenate([
        gyro,                    # 0:3
        gravity,                 # 3:6
        dof_pos_diff,            # 6:18
        dof_vel,                 # 18:30
        last_action,             # 30:42
        commands,                # 42:45
        feet_phase,              # 45:49
    ])
    return obs.astype(np.float32)


def run_policy(obs: np.ndarray) -> np.ndarray:
    """Run ONNX policy, return 12-dim action."""
    import onnxruntime as ort
    session = ort.InferenceSession(str(POLICY_PATH), providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    action = session.run(None, {input_name: obs.reshape(1, OBS_DIM)})[0]
    return action.flatten()


def load_captures() -> Optional[list]:
    state_path = LOG_DIR / "captures.json"
    if not state_path.exists():
        return None
    return json.loads(state_path.read_text())


def compute_offsets(captures: list, directions: np.ndarray) -> np.ndarray:
    """offset = mechpos_at_standing / reduction - direction * default_pos"""
    offsets = np.zeros(NUM_JOINTS)
    for i in range(NUM_JOINTS):
        cap = captures[i]
        if cap is not None:
            offsets[i] = cap / REDUCTIONS[i] - directions[i] * DEFAULT_POS[i]
        else:
            # Fallback: MuJoCo prone method
            offsets[i] = -MUJOCO_PRONE[i]
    return offsets


def evaluate(directions: np.ndarray, captures: list) -> dict:
    """Evaluate a direction configuration at both prone and standing poses."""
    offsets = compute_offsets(captures, directions)

    # Build observations at prone (mechPos=0) and standing (mechPos=captured)
    motor_prone = np.zeros(NUM_JOINTS)
    motor_standing = np.array([c if c is not None else 0.0 for c in captures])

    # At prone: robot is on ground, gravity = body-frame gravity
    # The body is horizontal, so gravity in body frame is ~[-gx, -gy, -gz] depending on orientation
    # Simplification: use upright gravity for both (policy trained for upright)
    # For prone validation, the joint angles are far from standing, so actions should be LARGE
    # For standing validation, actions should be SMALL if calibration is correct

    obs_prone = build_observation(motor_prone, directions, offsets)
    obs_stand = build_observation(motor_standing, directions, offsets)

    try:
        act_prone = run_policy(obs_prone)
        act_stand = run_policy(obs_stand)
    except Exception as e:
        return {"error": str(e), "rms_prone": 999, "rms_stand": 999}

    rms_prone = float(np.sqrt(np.mean(act_prone ** 2)))
    rms_stand = float(np.sqrt(np.mean(act_stand ** 2)))
    max_abs_prone = float(np.max(np.abs(act_prone)))
    max_abs_stand = float(np.max(np.abs(act_stand)))

    # Check prone logical vs MuJoCo
    logical_prone = directions * (motor_prone - offsets)
    prone_error = float(np.sqrt(np.mean((logical_prone - MUJOCO_PRONE) ** 2)))

    return {
        "rms_prone": rms_prone,
        "rms_stand": rms_stand,
        "max_abs_prone": max_abs_prone,
        "max_abs_stand": max_abs_stand,
        "prone_error": prone_error,
        "action_prone": act_prone.tolist(),
        "action_stand": act_stand.tolist(),
        "logical_prone": logical_prone.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="OpenDoge 标定方向校验")
    parser.add_argument("--auto-flip", action="store_true",
                        help="自动检测并建议方向翻转")
    parser.add_argument("--auto-discover", action="store_true",
                        help="全自动发现: 从 MuJoCo 出发, ONNX 迭代优化 direction")
    args = parser.parse_args()

    captures = load_captures()
    if captures is None:
        print("❌ 没有 captures.json，请先运行 record_standing.py 标定")
        return 1

    n_captured = sum(1 for c in captures if c is not None)
    print(f"已锁定: {n_captured}/12 关节")
    print()

    # Current directions (all +1 from record_standing.py)
    directions = np.ones(NUM_JOINTS, dtype=int)

    print("=" * 90)
    print(f"{'关节':<18s} {'direction':>9s} {'offset':>10s} "
          f"{'prone_logical':>14s} {'mujoco_prone':>14s} {'匹配':>6s}")
    print("-" * 90)

    offsets = compute_offsets(captures, directions)
    for i in range(NUM_JOINTS):
        cap = captures[i]
        logical_prone = directions[i] * (0.0 - offsets[i])
        match = "✅" if abs(logical_prone - MUJOCO_PRONE[i]) < 0.3 else "❌"
        src = "实测" if cap is not None else "MuJoCo"
        print(f"  {JOINT_NAMES[i]:<18s} {directions[i]:>+9d} {offsets[i]:>10.4f} "
              f"{logical_prone:>14.4f} {MUJOCO_PRONE[i]:>14.4f} {match:>6s}  [{src}]")

    print()
    print("prone_logical = 趴伏时 URDF 关节角 (direction * (0 - offset))")
    print("mujoco_prone  = MuJoCo 仿真趴伏角")
    print("匹配: |prone_logical - mujoco_prone| < 0.3 rad (17°)")
    print()

    # ── ONNX evaluation ──
    if not POLICY_PATH.exists():
        print(f"⚠️  策略文件不存在: {POLICY_PATH}")
        print("跳过 ONNX 校验。请根据上方 prone_logical 匹配判断方向。")
        return 0

    print("=" * 90)
    print("ONNX 策略校验")
    print("=" * 90)

    # Evaluate current directions
    result = evaluate(directions, captures)
    if "error" in result:
        print(f"❌ ONNX 错误: {result['error']}")
        return 1

    print(f"\n当前方向配置:")
    print(f"  趴伏姿态策略 RMS: {result['rms_prone']:.4f} (max={result['max_abs_prone']:.4f})")
    print(f"  站立姿态策略 RMS: {result['rms_stand']:.4f} (max={result['max_abs_stand']:.4f})")
    print(f"  MuJoCo prone 误差: {result['prone_error']:.4f} rad")

    print(f"\n站立姿态各关节策略输出 (应接近 0):")
    for i in range(NUM_JOINTS):
        a = result['action_stand'][i]
        bar = '█' * min(40, int(abs(a) * 20))
        ok = "✅" if abs(a) < 0.3 else ("⚠️" if abs(a) < 0.6 else "❌")
        print(f"  {ok} {JOINT_NAMES[i]:<18s} action={a:+.4f}  {bar}")

    # ── Auto-flip test ──
    if args.auto_flip:
        print(f"\n{'='*90}")
        print("自动方向检测 — 逐个关节翻转测试")
        print(f"{'='*90}")

        best_directions = directions.copy()
        best_score = result['rms_stand']

        for i in range(NUM_JOINTS):
            if captures[i] is None:
                continue  # Skip non-captured joints

            # Try flipping this joint
            test_dir = directions.copy()
            test_dir[i] *= -1
            test_result = evaluate(test_dir, captures)

            if "error" in test_result:
                continue

            current_act = abs(result['action_stand'][i])
            flipped_act = abs(test_result['action_stand'][i])
            current_prone_err = abs(directions[i] * (0 - offsets[i]) - MUJOCO_PRONE[i])
            flipped_offsets = compute_offsets(captures, test_dir)
            flipped_prone_err = abs(test_dir[i] * (0 - flipped_offsets[i]) - MUJOCO_PRONE[i])

            improved = flipped_act < current_act * 0.5  # 50% improvement
            prone_better = flipped_prone_err < current_prone_err

            marker = ""
            if improved and prone_better:
                marker = " ← 建议翻转!"
                best_directions[i] *= -1
                best_score = min(best_score, test_result['rms_stand'])

            print(f"  {JOINT_NAMES[i]:<18s} "
                  f"dir=+1: act={current_act:+.4f} prone_err={current_prone_err:.3f} | "
                  f"dir=-1: act={flipped_act:+.4f} prone_err={flipped_prone_err:.3f}"
                  f"{marker}")

        # Show recommended changes
        changes = [(i, JOINT_NAMES[i]) for i in range(NUM_JOINTS)
                   if best_directions[i] != directions[i]]
        if changes:
            print(f"\n建议修改 direction (共 {len(changes)} 个关节):")
            for i, name in changes:
                print(f"  joint.{name}.direction = -1  (当前 +1)")
        else:
            print(f"\n✅ 所有已锁定关节 direction 正确, 无需修改.")

    # ── Auto-discover: fully automatic direction discovery ──
    if args.auto_discover:
        print(f"\n{'='*90}")
        print("全自动方向发现 — MuJoCo prone + ONNX 迭代优化")
        print(f"{'='*90}")
        print()
        print("方法: 从 MuJoCo 趴伏角出发, ONNX 策略评估, 逐个翻转方向优化")
        print("     无需任何手动掰腿!")
        print()

        all_mujoco_captures = [None] * NUM_JOINTS
        best_dir = np.ones(NUM_JOINTS, dtype=int)
        best_offsets = compute_offsets(all_mujoco_captures, best_dir)
        motor_prone = np.zeros(NUM_JOINTS)

        # Initial evaluation
        print("初始: 全部 direction=+1, MuJoCo offsets")
        best_action = run_policy(build_observation(motor_prone, best_dir, best_offsets))
        best_rms = float(np.sqrt(np.mean(best_action ** 2)))
        print(f"  策略 RMS = {best_rms:.4f}")

        # Iterative optimization (2 passes usually enough)
        for iteration in range(3):
            improved = False
            for i in range(NUM_JOINTS):
                test_dir = best_dir.copy()
                test_dir[i] *= -1
                test_offsets = compute_offsets(all_mujoco_captures, test_dir)
                test_obs = build_observation(motor_prone, test_dir, test_offsets)
                test_action = run_policy(test_obs)

                best_act = abs(best_action[i])
                test_act = abs(test_action[i])

                if test_act < best_act * 0.7:  # 30%+ improvement
                    best_dir[i] = test_dir[i]
                    best_action = test_action
                    best_rms = float(np.sqrt(np.mean(test_action ** 2)))
                    improved = True
                    print(f"  iter{iteration}: 翻转 {JOINT_NAMES[i]:<18s} "
                          f"|act| {best_act:.3f} → {test_act:.3f}  RMS={best_rms:.4f}")

            if not improved:
                print(f"  iter{iteration}: 收敛")
                break

        # Final output
        final_offsets = compute_offsets(all_mujoco_captures, best_dir)
        final_action = run_policy(build_observation(motor_prone, best_dir, final_offsets))
        final_rms = float(np.sqrt(np.mean(final_action ** 2)))

        print(f"\n{'='*90}")
        print("最终标定 (全自动, 零手动)")
        print(f"{'='*90}")
        print(f"策略趴伏 RMS: {final_rms:.4f}")
        print()
        print(f"{'关节':<18s} {'dir':>4s} {'offset':>10s} {'prone_log':>10s}")
        print("-" * 48)
        for i in range(NUM_JOINTS):
            lp = best_dir[i] * (0 - final_offsets[i])
            print(f"  {JOINT_NAMES[i]:<18s} {best_dir[i]:>+4d} {final_offsets[i]:>10.4f} {lp:>10.4f}")

        print(f"\n# Deploy config:")
        for i in range(NUM_JOINTS):
            print(f"joint.{JOINT_NAMES[i]}.direction={best_dir[i]}")
            print(f"joint.{JOINT_NAMES[i]}.offset={final_offsets[i]:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

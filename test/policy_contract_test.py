#!/usr/bin/env python3
"""Policy contract test for the OpenDoge_train SuperMimic ONNX artifact.

Checks the input/output shape and the metadata that the C++ deploy runtime
relies on, then runs one deterministic standing inference and compares it with
a golden action. No MuJoCo and no hardware are required.

Usage:
  python3 test/policy_contract_test.py [--policy PATH]
"""

import argparse
import json
import os
import sys

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("onnxruntime is required: pip install onnxruntime", file=sys.stderr)
    sys.exit(1)

OBS_DIM = 47
HISTORY = 5
POLICY_INPUT = OBS_DIM * HISTORY
CONTROL_DT = 0.02
JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]
ACTION_SCALES = [0.25, 0.4, 0.4] * 4
DEFAULT_TRAIN = [
    0.0, 0.8, -1.6, 0.0, -0.8, 1.6,
    0.0, 0.8, -1.6, 0.0, -0.8, 1.6,
]

# Deterministic standing history: zero ang_vel, gravity [0,0,-1], zero joints,
# zero last_action, zero command, zero phase; repeated five times.
GOLDEN_ACTION = np.array([
    0.01321755, 0.007009119, 0.2153393, 0.07400297,
    -0.002482623, -0.1669809, 0.1289425, -0.01519401,
    0.3218557, -0.01875342, -0.02323685, -0.2923968,
])

passed = 0
failed = 0


def check(label: str, condition: bool):
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {label}")


def main() -> int:
    global passed, failed
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "policy",
            "opendoge_supermimic_rough_20000.onnx",
        ),
    )
    args = parser.parse_args()

    session = ort.InferenceSession(args.policy, providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    print(f"policy: {args.policy}")

    check("one input", len(inputs) == 1)
    check("one output", len(outputs) == 1)
    input_shape = inputs[0].shape
    output_shape = outputs[0].shape
    check("input shape [batch,235]", len(input_shape) == 2 and input_shape[1] == POLICY_INPUT)
    check("output shape [batch,12]", len(output_shape) == 2 and output_shape[1] == 12)
    check("input name", inputs[0].name == "obs_history")
    check("output name", outputs[0].name == "actions")

    raw = session.get_modelmeta().custom_metadata_map
    meta = {key: json.loads(value) for key, value in raw.items()}
    print(f"metadata: obs={meta.get('observation_dim')} "
          f"history={meta.get('history_length')} dt={meta.get('control_dt_seconds')}")

    check("observation_dim 47", meta.get("observation_dim") == OBS_DIM)
    check("history_length 5", meta.get("history_length") == HISTORY)
    check("control_dt 0.02", abs(meta.get("control_dt_seconds", 0.0) - CONTROL_DT) < 1e-12)
    check("history_order", meta.get("history_order") == "frame_major_oldest_first")
    check("history_reset", meta.get("history_reset") == "repeat_initial_frame")
    check("joint_names", list(meta.get("joint_names", [])) == JOINT_NAMES)
    check("action_scales", np.allclose(meta.get("action_scales", []), ACTION_SCALES))
    check("default_joint_positions",
          np.allclose(meta.get("default_joint_positions", []), DEFAULT_TRAIN))
    check("action_semantics",
          meta.get("action_semantics") == "unscaled_joint_position_offsets")
    check("gait_phase observed", meta.get("gait_phase", {}).get("observed") is True)
    check("gait period", abs(meta.get("gait_phase", {}).get("period_seconds", 0.0) - 0.5) < 1e-12)
    check("command frame", meta.get("command_frame") == "body_xy_world_yaw")

    frame = np.zeros(OBS_DIM, dtype=np.float32)
    frame[5] = -1.0  # projected_gravity z
    history = np.tile(frame, HISTORY).reshape(1, POLICY_INPUT)
    action = session.run([outputs[0].name], {inputs[0].name: history})[0].flatten()
    check("finite action", np.all(np.isfinite(action)))
    check("action dimension", len(action) == 12)
    if np.allclose(action, GOLDEN_ACTION, atol=1e-4):
        passed += 1
    else:
        failed += 1
        print("  FAIL: golden action mismatch")
        print(f"    expected: {GOLDEN_ACTION}")
        print(f"    actual:   {action}")

    print(f"\n{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

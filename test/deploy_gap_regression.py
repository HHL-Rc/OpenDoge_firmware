#!/usr/bin/env python3
"""
deploy_gap_regression.py — C++/Python control-loop alignment regression test.

Covers the OpenDoge_train SuperMimic contract (47-dim frame, 5-frame history,
50 Hz, per-joint action scales, gait phase). Pure functions only: no MuJoCo
window, no ONNX session, no hardware.

Usage: python3 test/deploy_gap_regression.py
Exit code: 0 = all tests passed, 1 = failures detected
"""

import math
import os
import sys

import numpy as np

# Allow `from deploy_mujoco import ...` when run from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim2sim"))

from deploy_mujoco import (  # noqa: E402
    ACTION_SCALE,
    CONTROL_DT,
    DEFAULT_POS,
    GAIT_PERIOD,
    GAIT_STEPS,
    GAIT_THRESHOLD,
    HISTORY,
    JOINT_NAMES,
    JOINT_SIGN,
    NUM_JOINTS,
    OBS_DIM,
    POLICY_INPUT,
    DeployConfig,
    HistoryBuffer,
    OperatorCommand,
    build_observation_frame,
    gait_phase_features,
    rate_limit,
)

passed = 0
failed = 0


def check(label: str, condition: bool):
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {label}")


def test_contract_constants():
    check("obs dim 47", OBS_DIM == 47)
    check("history 5", HISTORY == 5)
    check("policy input 235", POLICY_INPUT == 235)
    check("control dt 0.02", abs(CONTROL_DT - 0.02) < 1e-12)
    check("gait period 0.5", abs(GAIT_PERIOD - 0.5) < 1e-12)
    check("gait steps 25", GAIT_STEPS == 25)
    check("gait threshold 0.1", abs(GAIT_THRESHOLD - 0.1) < 1e-12)
    check("action scales", np.allclose(ACTION_SCALE, [0.25, 0.4, 0.4] * 4))


def test_rate_limit():
    check("rate_limit forward step", rate_limit(5.0, 0.0, 1.0) == 1.0)
    check("rate_limit reverse step", rate_limit(-5.0, 0.0, 1.0) == -1.0)
    check("rate_limit within range", rate_limit(0.5, 0.0, 1.0) == 0.5)
    check("rate_limit zero step", rate_limit(1.0, 1.0, 0.5) == 1.0)


def test_gait_phase():
    # 50 Hz, 0.5 s period -> 25 steps; phase is quantized.
    p0 = gait_phase_features(0.0, OperatorCommand(vx=0.4))
    check("phase 0 sin", abs(p0[0]) < 1e-12)
    check("phase 0 cos", abs(p0[1] - 1.0) < 1e-12)

    p1 = gait_phase_features(0.02, OperatorCommand(vx=0.4))
    expected = 2.0 * math.pi / 25.0
    check("phase 1 sin", abs(p1[0] - math.sin(expected)) < 1e-12)
    check("phase 1 cos", abs(p1[1] - math.cos(expected)) < 1e-12)

    p24 = gait_phase_features(0.48, OperatorCommand(vx=0.4))
    expected24 = 2.0 * math.pi * 24.0 / 25.0
    check("phase 24 sin", abs(p24[0] - math.sin(expected24)) < 1e-12)

    p25 = gait_phase_features(0.50, OperatorCommand(vx=0.4))
    check("phase wraps at 25 steps", np.allclose(p25, p0))

    # Standing gate: l2(vx,vy) + abs(yaw) <= 0.1 -> zeros.
    check("standing zero", np.allclose(
        gait_phase_features(0.2, OperatorCommand(vx=0.0, vy=0.0, yaw_rate=0.0)),
        np.zeros(2)))
    check("yaw-only moving", not np.allclose(
        gait_phase_features(0.2, OperatorCommand(vx=0.0, vy=0.0, yaw_rate=0.2)),
        np.zeros(2)))
    check("threshold uses l2 + abs", np.allclose(
        gait_phase_features(
            0.2, OperatorCommand(vx=0.05, vy=0.05, yaw_rate=0.0)),
        np.zeros(2)))
    check("l2 above threshold moves", not np.allclose(
        gait_phase_features(
            0.2, OperatorCommand(vx=0.08, vy=0.08, yaw_rate=0.0)),
        np.zeros(2)))


def test_observation_frame():
    q = DEFAULT_POS.copy()
    dq = np.full(NUM_JOINTS, 1.5)
    last_action = np.linspace(-3.8, 3.3, NUM_JOINTS)
    command = OperatorCommand(vx=0.5, vy=-0.2, yaw_rate=0.3)
    imu = type("Imu", (), {})()
    imu.angular_velocity = np.array([0.1, -0.2, 0.3])
    imu.projected_gravity = np.array([0.0, 0.0, -1.0])
    phase = np.array([0.25, 0.75])

    obs = build_observation_frame(q, dq, DEFAULT_POS, last_action, command, imu, phase)
    check("frame dimension", len(obs) == OBS_DIM)
    check("ang_vel first", np.allclose(obs[0:3], imu.angular_velocity))
    check("gravity 3:6", np.allclose(obs[3:6], imu.projected_gravity))
    check("no base_lin_vel", not np.allclose(obs[0:3], np.zeros(3)))
    check("joint pos zero at default", np.allclose(obs[6:18], np.zeros(NUM_JOINTS)))
    check("joint vel sign mapped", np.allclose(obs[18:30], JOINT_SIGN * 1.5))
    check("last_action raw", np.allclose(obs[30:42], last_action))
    check("command 42:45", np.allclose(obs[42:45], [0.5, -0.2, 0.3]))
    check("phase 45:47", np.allclose(obs[45:47], phase))

    q2 = DEFAULT_POS + np.array([0.1] * NUM_JOINTS)
    obs2 = build_observation_frame(
        q2, dq, DEFAULT_POS, np.zeros(NUM_JOINTS), command, imu, phase)
    check("joint pos sign mapped", np.allclose(obs2[6:18], JOINT_SIGN * 0.1))


def test_history_buffer():
    buf = HistoryBuffer()
    frame_a = np.arange(OBS_DIM, dtype=float)
    frame_b = frame_a + 1000.0
    buf.reset(frame_a)
    check("reset ready", buf.ready)
    check("reset repeats initial", np.allclose(
        buf.flatten(), np.tile(frame_a, HISTORY)))

    buf.append(frame_b)
    flat = buf.flatten()
    check("append oldest-first", np.allclose(flat[:OBS_DIM], frame_a))
    check("append newest-last", np.allclose(flat[-OBS_DIM:], frame_b))
    check("flatten length", len(flat) == POLICY_INPUT)

    buf.clear()
    check("clear not ready", not buf.ready)


def test_target_scales():
    # target_fw = default_fw + sign * action * action_scale[i]
    action = np.ones(NUM_JOINTS)
    target = DEFAULT_POS + JOINT_SIGN * action * ACTION_SCALE
    check("hip scale 0.25", np.allclose(target[[0, 3, 6, 9]],
                                        DEFAULT_POS[[0, 3, 6, 9]]
                                        + JOINT_SIGN[[0, 3, 6, 9]] * 0.25))
    check("thigh scale 0.4", np.allclose(target[[1, 4, 7, 10]],
                                         DEFAULT_POS[[1, 4, 7, 10]]
                                         + JOINT_SIGN[[1, 4, 7, 10]] * 0.4))
    check("calf scale 0.4", np.allclose(target[[2, 5, 8, 11]],
                                        DEFAULT_POS[[2, 5, 8, 11]]
                                        + JOINT_SIGN[[2, 5, 8, 11]] * 0.4))
    # FR thigh sign flip: action +1 moves the firmware target down.
    check("FR thigh sign flip", target[4] < DEFAULT_POS[4])


def test_config():
    cfg = DeployConfig()
    check("config inference_hz 50", cfg.inference_hz == 50.0)
    check("config action scale array", np.allclose(cfg.action_scale, ACTION_SCALE))
    check("config gait period", cfg.gait_period_s == GAIT_PERIOD)
    check("config gait threshold", cfg.gait_command_threshold == GAIT_THRESHOLD)


def test_joint_names():
    check("12 joints", len(JOINT_NAMES) == NUM_JOINTS)
    check("first joint", JOINT_NAMES[0] == "FL_hip_joint")
    check("last joint", JOINT_NAMES[-1] == "RR_calf_joint")


def main():
    test_contract_constants()
    test_rate_limit()
    test_gait_phase()
    test_observation_frame()
    test_history_buffer()
    test_target_scales()
    test_config()
    test_joint_names()
    print(f"\n{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

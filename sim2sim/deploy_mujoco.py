#!/usr/bin/env python3
"""
deploy_mujoco.py — OpenDoge MuJoCo 仿真测试

在 MuJoCo 物理引擎中复现 opendoge_deploy 的完整控制回路，验证：
  - 状态机：WaitFeedback → Ready → EnteringPosition → ActivePC / ActiveRL
  - 位置控制预启动斜坡（kp/kd 从阻尼值平滑过渡到满 PD）
  - RL 推理直跳路径（跳过 EnteringPosition）
  - 模式切换和故障降级

与 C++ 固件 (main.cpp) 保持一致：
  - 12 关节顺序：FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
                  RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
  - 默认姿态：每腿 {0, 0.8, -1.6} (固件 URDF 约定)
  - PD 增益：kp=12, kd=0.5, safe_kd=2.0 (OpenDoge_train BuiltinPositionActuator)
  - 动作缩放：hip 0.25 / thigh 0.4 / calf 0.4
  - 观测：47 维 × 5 帧 = 235（OpenDoge_train SuperMimic，无 base_lin_vel）
  - 控制：策略 50 Hz，target/PD 200 Hz
  - 斜坡时长：pc_startup_ramp_s=2.0

用法:
  cd Lain_Code/sim2sim
  python3 deploy_mujoco.py \
    --training-root ../../OpenDoge_train \
    --start-pose default --mode rl \
    --policy ../policy/opendoge_supermimic_rough_20000.onnx \
    --cmd 0.4 0 0 --duration 10
  或者
    python3 deploy_mujoco.py \
    --training-root ../../OpenDoge_train \
    --start-pose default --mode rl \
    --policy ../policy/opendoge_supermimic_rough_20000.onnx \
    --joystick
  joystick需要显式指定不然不填写不会使用手柄为默认操作
  --mode 可以不填
  
  --training-root 默认自动查找同级 ../OpenDoge_train；
  --joint-convention 默认 training（训练 XML，镜像轴、无执行器）；
  --terrain flat|rough|stairs|boxes|slope 选择地形；
  --joystick 使用 /dev/input/js* 手柄。
  旧 48 维模型不兼容当前脚本，归档在 policy/legacy_48dim/。

完整手动流程 (键盘):
  idle (Ready 趴伏) → A/空格 → EnteringPosition → ActivePC (站立)
  ActivePC → X → ActiveRL (RL 推理, 方向键控制运动)
  ActiveRL → Y → ActivePC (切回站立)
  ActivePC/ActiveRL → B → Ready (趴伏)

键盘控制 (渲染窗口):
  A / 空格   — 激活位置控制 (EnteringPosition → ActivePC)
  B          — 停用 (回 Ready / 阻尼)
  X          — 切换到 RL 推理模式
  Y          — 切换回位置控制模式
  BACKSPACE  — 急停 (estop)
  ESC / Q    — 退出

依赖: pip install mujoco numpy onnxruntime
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np

# ─── 尝试导入 ONNX Runtime ──────────────────────────────────────────
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    ort = None
    print("[WARN] onnxruntime 未安装。运行: pip install onnxruntime", file=sys.stderr)

# ─── 尝试导入 MuJoCo ────────────────────────────────────────────────
try:
    import mujoco
    from mujoco import viewer as mujoco_viewer
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False
    mujoco_viewer = None
    print("[WARN] mujoco 未安装。运行: pip install mujoco", file=sys.stderr)

# ─── 尝试导入实机手柄映射 (与 hardware/gamepad 共用同一实现) ────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GAMEPAD_DIR = os.path.join(_REPO_ROOT, "hardware", "gamepad")
try:
    if _GAMEPAD_DIR not in sys.path:
        sys.path.insert(0, _GAMEPAD_DIR)
    from opendoge_joystick import (  # noqa: E402
        JS_EVENT_AXIS,
        JS_EVENT_BUTTON,
        DeviceLostError,
        LinuxJoystick,
        XboxCommandMapper,
        first_joystick_device,
    )
    HAS_JOYSTICK = True
except Exception as _joy_exc:  # pragma: no cover - 仅缺少文件时触发
    HAS_JOYSTICK = False
    _JOYSTICK_IMPORT_ERROR = _joy_exc


# ══════════════════════════════════════════════════════════════════════
# 常量 (与 types.hpp 保持一致)
# ══════════════════════════════════════════════════════════════════════

NUM_JOINTS = 12
OBS_DIM = 47
HISTORY = 5
POLICY_INPUT = OBS_DIM * HISTORY  # 235
POLICY_HZ = 50.0
CONTROL_DT = 1.0 / POLICY_HZ
GAIT_PERIOD = 0.5
GAIT_STEPS = 25  # round(GAIT_PERIOD / CONTROL_DT)
GAIT_THRESHOLD = 0.1

# OpenDoge_train action scales, per joint type (hip/thigh/calf × 4 legs).
ACTION_SCALE = np.array([0.25, 0.4, 0.4] * 4, dtype=float)

# Training actuator effort limits (joint side, N·m).
MAX_JOINT_TORQUE = np.array([6.0, 6.0, 9.0] * 4, dtype=float)

JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

# 训练 URDF ↔ 固件 URDF 符号约定 (与 types.hpp kJointSign 一致)。
# LainLab 模型中 FR/RR 腿镜像 (thigh/calf axis 0 -1 0)，RL/RR hip 镜像。
#   q_train = JOINT_SIGN * q_firmware
# 观测/动作在训练空间计算，目标位置再映射回固件空间。
JOINT_SIGN = np.array([
    +1, +1, +1,   # FL
    +1, -1, -1,   # FR
    -1, +1, +1,   # RL
    -1, -1, -1,   # RR
], dtype=float)

# 站立默认姿态 (rad, 固件 URDF 约定) — LainLab OPENDOGE_INIT_STATE
# 经 JOINT_SIGN 映射后四条腿一致。
DEFAULT_POS = np.array([
    0.0, 0.8, -1.6,   # FL
    0.0, 0.8, -1.6,   # FR
    0.0, 0.8, -1.6,   # RL
    0.0, 0.8, -1.6,   # RR
])

# 自然趴伏姿态 (rad, 固件 URDF 约定) — 关节失能、纯阻尼稳态在平地上
# 用于 Ready 态的初始位置, 不参与观测/目标计算
REST_POSE = np.array([
    0.230,  1.079, -2.681,   # FL
   -0.230,  1.079, -2.681,   # FR
    0.231,  1.090, -2.681,   # RL
   -0.230,  1.084, -2.681,   # RR
])

# 固件 URDF 关节限位 (rad, 固件约定)。
# 2026-11-06 (修改 54): 已对齐 LainLab 训练限位和实机手掰可用范围，
# 与 deploy/configs/opendoge_deploy.conf 保持一致；旧的
# hip 0.26/0.785、thigh 1.134 比训练窄，会让策略目标在 clamp 处饱和。
FIRMWARE_JOINT_LOWER = np.array([
    -1.0000, -1.0468, -2.68,   # FL
    -0.5218, -1.0468, -2.68,   # FR
    -0.9000, -0.8500, -2.68,   # RL
    -0.5218, -0.8500, -2.68,   # RR
])

FIRMWARE_JOINT_UPPER = np.array([
    0.5218, 1.3958, -1.04,     # FL
    1.0000, 1.3958, -1.04,     # FR
    0.5218, 1.3958, -1.04,     # RL
    0.9300, 1.3958, -1.04,     # RR
])

# LainLab URDF 关节限位映射到固件约定 (JOINT_SIGN * training_limits)。
# 仅在使用 LainLab 模型做策略消融时使用；实机部署仍用固件机械限位。
TRAINING_JOINT_LOWER = np.array([
    -1.0468, -1.0468, -2.68,   # FL
    -0.5218, -1.0468, -2.68,   # FR
    -1.0468, -1.0468, -2.68,   # RL
    -0.5218, -1.0468, -2.68,   # RR
])

TRAINING_JOINT_UPPER = np.array([
    0.5218, 1.3958, -1.04,     # FL
    1.0468, 1.3958, -1.04,     # FR
    0.5218, 1.3958, -1.04,     # RL
    1.0468, 1.3958, -1.04,     # RR
])

# 默认使用固件机械限位；--joint-convention training 时切换
JOINT_LOWER = FIRMWARE_JOINT_LOWER.copy()
JOINT_UPPER = FIRMWARE_JOINT_UPPER.copy()

MAX_POSITION_STEP = 0.015   # rad/target_tick @ 200Hz = 3 rad/s
# LowGainTest stand-up slew (matches DeployConfig::low_gain_max_position_step).
# The target is snapshotted from the measured pose on entry, so this is a ramp.
LOW_GAIN_MAX_POSITION_STEP = 0.002   # @200Hz = 0.4 rad/s
# Stand-up from a prone/far pose (matches DeployConfig::standup_max_position_step).
STANDUP_MAX_POSITION_STEP = 0.002    # @200Hz = 0.4 rad/s


# ══════════════════════════════════════════════════════════════════════
# 配置与数据结构
# ══════════════════════════════════════════════════════════════════════

class RuntimeState(Enum):
    WaitFeedback = 0
    Ready = 1
    EnteringPosition = 2
    ActivePC = 3
    ActiveRL = 4
    LowGainTest = 5
    DampingFault = 6
    StandingUp = 7


@dataclass
class DeployConfig:
    inference_hz: float = POLICY_HZ
    target_hz: float = 200.0
    control_hz: float = 1000.0
    kp: float = 12.0  # OpenDoge_train BuiltinPositionActuatorCfg.stiffness
    kd: float = 0.5   # OpenDoge_train BuiltinPositionActuatorCfg.damping
    safe_kd: float = 2.0
    action_scale: np.ndarray = field(default_factory=lambda: ACTION_SCALE.copy())
    gait_period_s: float = GAIT_PERIOD
    gait_command_threshold: float = GAIT_THRESHOLD
    # Keep the ActiveRL target this far inside the joint range (mirrors the
    # C++ rl_joint_limit_margin). 0 = training-exact target.
    rl_joint_limit_margin: float = 0.03
    pc_startup_ramp_s: float = 2.0
    pc_startup_max_deviation: float = 2.0  # MuJoCo 从 REST_POSE (~2 rad from stand) 启动，需大容差。硬件侧为 0.25 (C++ config)
    standup_duration_s: float = 4.0       # 趴伏/远离站姿起身时长
    standup_settle_s: float = 1.0         # 斜坡到位后保持默认姿态的稳定时间
    standup_handover_s: float = 1.0       # 交接后策略目标从默认姿态淡入的时长
    # Investigation-only sensor delay/loss models. They are applied to the
    # observation path only; the plant/PD still uses current state.
    imu_delay_ms: float = 0.0
    joint_delay_ms: float = 0.0
    packet_loss: float = 0.0
    loss_burst_ms: float = 0.0
    loss_target: str = "joint"  # joint|bus|all
    cmd_loss: float = 0.0


@dataclass
class OperatorCommand:
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0
    active: bool = False
    estop: bool = False
    position_control: bool = False
    rl_inference: bool = False
    low_gain_mode: bool = False


@dataclass
class ImuSample:
    linear_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    projected_gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0]))
    valid: bool = False


# ══════════════════════════════════════════════════════════════════════
# 工具函数 (与 main.cpp 一致)
# ══════════════════════════════════════════════════════════════════════

def rate_limit(desired: float, previous: float, max_step: float) -> float:
    """速率限制器 — 与 main.cpp rateLimit() 一致"""
    return previous + np.clip(desired - previous, -max_step, max_step)


def gait_phase_features(
    elapsed_s: float,
    command: OperatorCommand,
    control_dt: float = CONTROL_DT,
    period_s: float = GAIT_PERIOD,
    threshold: float = GAIT_THRESHOLD,
) -> np.ndarray:
    """OpenDoge_train gait phase observation (matches observer.cpp).

    The clock is episode-elapsed-time based and is NOT reset by the standing
    gate; it is reset when the deployment enters ActiveRL.
    """
    dt = max(control_dt, 1.0e-6)
    period = max(period_s, 1.0e-6)
    period_steps = max(1, int(round(period / dt)))
    steps = int(round(max(0.0, elapsed_s) / dt))
    phase = (steps % period_steps) / period_steps
    moving = math.hypot(command.vx, command.vy) + abs(command.yaw_rate) > threshold
    if not moving:
        return np.zeros(2)
    angle = 2.0 * math.pi * phase
    return np.array([math.sin(angle), math.cos(angle)])


def build_observation_frame(
    joint_positions: np.ndarray,
    joint_velocities: np.ndarray,
    default_pos: np.ndarray,
    last_action: np.ndarray,
    command: OperatorCommand,
    imu: ImuSample,
    phase_features: np.ndarray,
) -> np.ndarray:
    """Build one 47-dim OpenDoge_train SuperMimic actor frame.

    joint_positions/joint_velocities/default_pos are in firmware URDF
    convention; joint terms are mapped to the training convention with
    JOINT_SIGN. There is no base_lin_vel: the policy is blind.
    """
    obs = np.zeros(OBS_DIM)
    offset = 0

    # 1. base_ang_vel (3)
    obs[offset:offset+3] = imu.angular_velocity
    offset += 3

    # 2. projected_gravity (3)
    obs[offset:offset+3] = imu.projected_gravity
    offset += 3

    # 3. joint_pos_rel (12) — training convention
    obs[offset:offset+NUM_JOINTS] = JOINT_SIGN * (joint_positions - default_pos)
    offset += NUM_JOINTS

    # 4. joint_vel_rel (12) — raw velocity, training convention
    obs[offset:offset+NUM_JOINTS] = JOINT_SIGN * joint_velocities
    offset += NUM_JOINTS

    # 5. last_action (12) — raw policy output, no clipping
    obs[offset:offset+NUM_JOINTS] = last_action
    offset += NUM_JOINTS

    # 6. command (3)
    obs[offset:offset+3] = [command.vx, command.vy, command.yaw_rate]
    offset += 3

    # 7. gait phase sin/cos (2)
    obs[offset:offset+2] = phase_features

    return obs


class HistoryBuffer:
    """Five-frame, oldest-first history (ONNX history_order)."""

    def __init__(self):
        self.frames = np.zeros((HISTORY, OBS_DIM))
        self.ready = False

    def reset(self, frame: np.ndarray) -> None:
        self.frames[:] = frame
        self.ready = True

    def append(self, frame: np.ndarray) -> None:
        self.frames[:-1] = self.frames[1:]
        self.frames[-1] = frame
        self.ready = True

    def flatten(self) -> np.ndarray:
        return self.frames.reshape(-1)

    def clear(self) -> None:
        self.ready = False


# ══════════════════════════════════════════════════════════════════════
# MuJoCo 仿真器封装
# ══════════════════════════════════════════════════════════════════════

class OpenDogeSimulator:
    """OpenDoge MuJoCo 仿真器"""

    def __init__(
        self,
        model_path: str,
        render: bool = True,
        key_callback=None,
        joint_convention: str = "firmware",
    ):
        if not HAS_MUJOCO:
            raise RuntimeError("MuJoCo 未安装: pip install mujoco")

        # firmware: docs/URDF/xml (all joints +axis, 固件约定)
        # training: LainLab mjlab XML (FR/RR mirrored, 训练约定)
        if joint_convention not in ("firmware", "training"):
            raise ValueError("joint_convention must be 'firmware' or 'training'")
        self.joint_convention = joint_convention
        # model joint angle → firmware logical angle
        self._model_sign = (
            np.ones(NUM_JOINTS) if joint_convention == "firmware"
            else JOINT_SIGN.copy()
        )

        self.render_enabled = render
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # 构建关节映射：逻辑索引 → (joint_id, qpos_adr, dof_adr, actuator_id)
        self._qpos_adr = np.zeros(NUM_JOINTS, dtype=int)
        self._dof_adr = np.zeros(NUM_JOINTS, dtype=int)
        self._actuator_ids = np.zeros(NUM_JOINTS, dtype=int)

        for i, name in enumerate(JOINT_NAMES):
            try:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            except Exception:
                raise RuntimeError(f"MuJoCo 模型中找不到关节: {name}")
            if jid < 0:
                raise RuntimeError(f"MuJoCo 模型中找不到关节: {name}")
            self._qpos_adr[i] = self.model.jnt_qposadr[jid]
            self._dof_adr[i] = self.model.jnt_dofadr[jid]
            # 查找对应的 actuator (LainLab XML 无 actuator，需用 qfrc_applied)
            aid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
            if aid < 0 and i < self.model.nu:
                aid = i  # fallback: 顺序匹配
            self._actuator_ids[i] = aid
        self._has_actuators = bool(
            self.model.nu >= NUM_JOINTS
            and np.all(self._actuator_ids >= 0)
        )

        # LainLab 的 armature 由 mjlab BuiltinPositionActuatorCfg 在运行时注入
        # (armature=0.005)，XML 本身没有。缺少它会因 calf 惯量极小而导致
        # 显式积分发散 (ω·dt > 2)。
        if joint_convention == "training" and not self._has_actuators:
            self.model.dof_armature[self._dof_adr] = 0.005
        if joint_convention == "training":
            self._apply_training_physics()

        # 传感器地址：必须用 sensor_adr（sensordata 偏移），不能用 sensor id。
        # 多分量传感器会让 id 与 adr 不同。
        def sensor_adr(name: str) -> Optional[int]:
            sid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, name
            )
            return None if sid < 0 else int(self.model.sensor_adr[sid])

        gyro_names = ("angular-velocity", "imu_ang_vel")
        self._gyro_adr = next(
            (a for a in (sensor_adr(n) for n in gyro_names) if a is not None), None
        )
        self._quat_adr = sensor_adr("orientation")
        self._has_sensors = self._gyro_adr is not None

        # 线速度传感器: 固件 XML = "linear-velocity", LainLab XML = "imu_lin_vel"
        self._lin_vel_adr = next(
            (a for a in (
                sensor_adr("linear-velocity"), sensor_adr("imu_lin_vel")
            ) if a is not None),
            None,
        )

        # 渲染
        if self.render_enabled:
            self.viewer = mujoco_viewer.launch_passive(
                self.model, self.data, key_callback=key_callback,
            )
        else:
            self.viewer = None

        self.step_count = 0

    def _apply_training_physics(self) -> None:
        """Align the bare MuJoCo model with mjlab's runtime physics.

        The training XML intentionally has no actuators and no runtime contact
        overrides; mjlab applies timestep=0.005, armature=0.005, foot
        condim/friction/solref and effort limits at load time. Without these
        the policy is evaluated on a different plant than it was trained on.
        """
        self.model.opt.timestep = 0.005
        self.model.opt.iterations = 10
        self.model.opt.ls_iterations = 20
        self.model.opt.ccd_iterations = 50
        self.model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        self.model.opt.impratio = 1.0

        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
            ) or ""
            if "foot_collision" in name:
                self.model.geom_condim[geom_id] = 3
                self.model.geom_priority[geom_id] = 1
                self.model.geom_friction[geom_id] = [1.0, 0.005, 0.0005]
                self.model.geom_solref[geom_id] = [0.01, 1.0]
            elif "collision" in name:
                self.model.geom_condim[geom_id] = 1

    def get_joint_positions(self) -> np.ndarray:
        """读取 12 个关节位置 (rad), 返回固件约定 — 使用 jnt_qposadr"""
        q = np.zeros(NUM_JOINTS)
        for i in range(NUM_JOINTS):
            q[i] = self._model_sign[i] * self.data.qpos[self._qpos_adr[i]]
        return q

    def get_joint_velocities(self) -> np.ndarray:
        """读取 12 个关节速度 (rad/s), 返回固件约定 — 使用 jnt_dofadr"""
        dq = np.zeros(NUM_JOINTS)
        for i in range(NUM_JOINTS):
            dq[i] = self._model_sign[i] * self.data.qvel[self._dof_adr[i]]
        return dq

    def apply_pd_control(
        self,
        targets: np.ndarray,
        kp_gains: np.ndarray,
        kd_gains: np.ndarray,
    ):
        """施加 PD 控制：力矩 = kp*(target - q) - kd*dq, 写入 data.ctrl。

        targets 为固件约定；q/dq 读取时已映射到固件约定，力矩再映射回模型约定。
        """
        q = self.get_joint_positions()
        dq = self.get_joint_velocities()
        torque = kp_gains * (targets - q) - kd_gains * dq
        if self.joint_convention == "training":
            torque = np.clip(torque, -MAX_JOINT_TORQUE, MAX_JOINT_TORQUE)
        for i in range(NUM_JOINTS):
            tau_model = self._model_sign[i] * torque[i]
            if self._has_actuators:
                self.data.ctrl[self._actuator_ids[i]] = tau_model
            else:
                self.data.qfrc_applied[self._dof_adr[i]] = tau_model

    def reset_to_rest_pose(self):
        """将机器人重置到自然趴伏姿态 (Ready 态初始位)"""
        mujoco.mj_resetData(self.model, self.data)
        for i in range(NUM_JOINTS):
            self.data.qpos[self._qpos_adr[i]] = self._model_sign[i] * REST_POSE[i]
        self.data.qpos[2] = 0.05  # 趴伏 base 贴近地面
        mujoco.mj_forward(self.model, self.data)

    def reset_to_default_pose(self, base_height: float = 0.158):
        """重置到站立默认姿态 (用于无键盘消融实验)。"""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = [0.0, 0.0, base_height]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        for i in range(NUM_JOINTS):
            self.data.qpos[self._qpos_adr[i]] = self._model_sign[i] * DEFAULT_POS[i]
        mujoco.mj_forward(self.model, self.data)

    @staticmethod
    def _rotate_vector_by_quat(v: np.ndarray, q: np.ndarray) -> np.ndarray:
        """用四元数 q=[w,x,y,z] 旋转向量 v: q * v * q^-1  (即 R(q)·v)"""
        w, x, y, z = q
        t = 2.0 * (x * v[0] + y * v[1] + z * v[2])
        t2 = w * w - (x * x + y * y + z * z)
        return np.array([
            t * x + t2 * v[0] + 2.0 * w * (y * v[2] - z * v[1]),
            t * y + t2 * v[1] + 2.0 * w * (z * v[0] - x * v[2]),
            t * z + t2 * v[2] + 2.0 * w * (x * v[1] - y * v[0]),
        ])

    @staticmethod
    def _rotate_vector_by_quat_inverse(v: np.ndarray, q: np.ndarray) -> np.ndarray:
        """q^-1 * v * q (即 R(q)^T·v)，与 mjlab quat_apply_inverse 一致。"""
        w, x, y, z = q
        t = 2.0 * np.array([
            y * v[2] - z * v[1],
            z * v[0] - x * v[2],
            x * v[1] - y * v[0],
        ])
        return v - w * t + np.array([
            y * t[2] - z * t[1],
            z * t[0] - x * t[2],
            x * t[1] - y * t[0],
        ])

    def get_imu(self) -> ImuSample:
        """读取 IMU 传感器数据 — 用 framequat 计算 projected_gravity 与训练一致"""
        imu = ImuSample()
        if self._has_sensors:
            imu.angular_velocity = self.data.sensordata[
                self._gyro_adr:self._gyro_adr + 3
            ].copy()
            # 用姿态四元数将世界系重力 [0,0,-1] 旋转到机体坐标系。
            # 有 framequat 时用传感器，否则用 floating base 的 qpos 四元数。
            if self._quat_adr is not None:
                quat = self.data.sensordata[
                    self._quat_adr:self._quat_adr + 4
                ].copy()  # [w, x, y, z]
            else:
                quat = self.data.qpos[3:7].copy()  # free joint [w, x, y, z]
            # 训练侧 mjlab: projected_gravity_b = R^T · (0,0,-1)
            # （quat_apply_inverse(root_link_quat_w, gravity_vec_w)）。
            # framequat / qpos 四元数是 body->world，所以这里必须用逆旋转；
            # 旧代码用正向 R·g，会把 gx/gy 的符号翻反（低头时 gx 变负）。
            imu.projected_gravity = self._rotate_vector_by_quat_inverse(
                np.array([0.0, 0.0, -1.0]), quat
            )
            imu.valid = True
        # base_lin_vel (body/site frame). Present in both XMLs; the controller
        # may override it depending on --lin-vel-mode.
        if self._lin_vel_adr is not None:
            imu.linear_velocity = self.data.sensordata[
                self._lin_vel_adr:self._lin_vel_adr + 3
            ].copy()
        return imu

    def step(self):
        """执行一步物理仿真"""
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

    def render(self):
        """渲染当前帧"""
        if self.viewer is not None:
            try:
                if self.viewer.is_running():
                    self.viewer.sync()
            except Exception:
                pass  # 窗口已关闭

    def close(self):
        """关闭仿真 — 不显式调用 viewer.close() 以避免 GLFW segfault"""
        self.viewer = None  # 放弃引用, 让进程退出时 OS 清理 GLFW

    def is_running(self) -> bool:
        """检查渲染窗口是否仍在运行"""
        if self.viewer is not None:
            return self.viewer.is_running()
        return True


# ══════════════════════════════════════════════════════════════════════
# 模拟策略 (替代真实的 ONNX 推理)
# ══════════════════════════════════════════════════════════════════════

class MockPolicy:
    """
    模拟 RL 策略 — 生成简单的正弦波动作用于测试。
    实际部署时替换为 ONNX 推理。
    """

    def __init__(self, failure_rate: float = 0.0):
        self.failure_rate = failure_rate
        self.call_count = 0

    def infer(self, obs: np.ndarray) -> Tuple[bool, np.ndarray, str]:
        """
        模拟推理。
        返回 (success, action, error_msg)
        """
        self.call_count += 1

        if self.failure_rate > 0 and np.random.random() < self.failure_rate:
            return False, np.zeros(NUM_JOINTS), "mock: simulated inference failure"

        t = self.call_count * CONTROL_DT  # 50 Hz
        action = np.zeros(NUM_JOINTS)
        for leg_idx in range(4):
            hip_idx = leg_idx * 3
            thigh_idx = hip_idx + 1
            action[hip_idx] = 0.05 * math.sin(t * 3.0 + leg_idx * math.pi / 2)
            action[thigh_idx] = 0.03 * math.sin(t * 2.5 + leg_idx * math.pi / 2 + 0.3)
        return True, np.clip(action, -1.0, 1.0), ""


class OnnxPolicy:
    """通过 ONNX Runtime 加载真实 RL 策略进行推理。"""

    def __init__(self, model_path: str):
        if not HAS_ONNX:
            raise RuntimeError("onnxruntime 未安装: pip install onnxruntime")
        self._session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self._validate_contract(model_path)
        self.call_count = 0

    def _validate_contract(self, model_path: str) -> None:
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                f"{model_path}: expected one input and one output, "
                f"got {len(inputs)} and {len(outputs)}"
            )
        input_shape = inputs[0].shape
        output_shape = outputs[0].shape
        if len(input_shape) != 2 or input_shape[1] != POLICY_INPUT:
            raise ValueError(
                f"{model_path}: ONNX input shape must be [batch,{POLICY_INPUT}], "
                f"got {input_shape}"
            )
        if len(output_shape) != 2 or output_shape[1] != NUM_JOINTS:
            raise ValueError(
                f"{model_path}: ONNX output shape must be [batch,{NUM_JOINTS}], "
                f"got {output_shape}"
            )

        raw_meta = self._session.get_modelmeta().custom_metadata_map
        try:
            meta = {key: json.loads(value) for key, value in raw_meta.items()}
        except Exception as exc:
            raise ValueError(f"{model_path}: cannot parse ONNX metadata: {exc}") from exc

        required = (
            "observation_dim", "history_length", "control_dt_seconds",
            "history_order", "action_scales", "default_joint_positions",
            "joint_names",
        )
        missing = [key for key in required if key not in meta]
        if missing:
            raise ValueError(
                f"{model_path}: ONNX metadata missing {missing}; "
                "this is not an OpenDoge_train SuperMimic policy"
            )
        if meta["observation_dim"] != OBS_DIM:
            raise ValueError(
                f"{model_path}: observation_dim={meta['observation_dim']} != {OBS_DIM}"
            )
        if meta["history_length"] != HISTORY:
            raise ValueError(
                f"{model_path}: history_length={meta['history_length']} != {HISTORY}"
            )
        if abs(meta["control_dt_seconds"] - CONTROL_DT) > 1e-9:
            raise ValueError(
                f"{model_path}: control_dt_seconds={meta['control_dt_seconds']} "
                f"!= {CONTROL_DT}"
            )
        if meta["history_order"] != "frame_major_oldest_first":
            raise ValueError(
                f"{model_path}: history_order={meta['history_order']!r} "
                "must be 'frame_major_oldest_first'"
            )
        scales = np.asarray(meta["action_scales"], dtype=float)
        if scales.shape != (NUM_JOINTS,) or not np.allclose(scales, ACTION_SCALE):
            raise ValueError(
                f"{model_path}: action_scales mismatch: {scales.tolist()} "
                f"!= {ACTION_SCALE.tolist()}"
            )
        default_train = np.array([
            0.0, 0.8, -1.6,  0.0, -0.8, 1.6,
            0.0, 0.8, -1.6,  0.0, -0.8, 1.6,
        ])
        defaults = np.asarray(meta["default_joint_positions"], dtype=float)
        if defaults.shape != (NUM_JOINTS,) or not np.allclose(defaults, default_train):
            raise ValueError(
                f"{model_path}: default_joint_positions mismatch: "
                f"{defaults.tolist()} != {default_train.tolist()}"
            )
        if list(meta["joint_names"]) != JOINT_NAMES:
            raise ValueError(
                f"{model_path}: joint_names mismatch: {meta['joint_names']}"
            )
        self.metadata = meta
        print(
            f"[POLICY] {model_path}: obs={meta['observation_dim']} "
            f"history={meta['history_length']} dt={meta['control_dt_seconds']} "
            f"action_scale={scales.tolist()}"
        )

    def infer(self, history: np.ndarray) -> Tuple[bool, np.ndarray, str]:
        """ONNX 推理。返回 (success, action, error_msg)"""
        self.call_count += 1
        try:
            input_feed = {
                self._input_name: history.astype(np.float32).reshape(1, -1)
            }
            action = self._session.run([self._output_name], input_feed)[0]
            return True, action.flatten(), ""
        except Exception as e:
            return False, np.zeros(NUM_JOINTS), f"ONNX inference error: {e}"


# ══════════════════════════════════════════════════════════════════════
# 键盘输入 (模拟手柄)
# ══════════════════════════════════════════════════════════════════════

class KeyboardHandler:
    """通过 MuJoCo viewer 的键盘回调模拟手柄输入

    GLFW 键码参考:
      ESC=256, Q=81, Backspace=259, Space=32
      A=65, B=66, X=88, Y=89
      上=265, 下=264, 左=263, 右=262  (GLFW: UP=265, DOWN=264)
    """

    def __init__(self, quit_only: bool = False):
        # quit_only=True: 只响应 Esc/Q，其余按键交给手柄输入源
        self.quit_only = quit_only
        self.active_requested = False
        self.estop = False
        self.position_control = False
        self.rl_inference = False
        self.low_gain_mode = False
        self.should_quit = False
        self._pending_keys = []
        # 方向键速度命令
        self._arrow_vx = 0.0
        self._arrow_vy = 0.0
        self._arrow_yaw = 0.0

    def feed_key(self, keycode: int):
        """从 viewer callback 接收按键 (GLFW 线程调用)"""
        self._pending_keys.append(keycode)

    def process_keys(self):
        """处理累积的按键 (主线程调用)"""
        for keycode in self._pending_keys:
            self._handle_key(keycode)
        self._pending_keys.clear()

    def _handle_key(self, keycode: int):
        if self.quit_only and keycode not in (256, 257, 81):
            return
        if keycode in (256, 257, 81):  # ESC, GLFW_ESCAPE, Q
            self.should_quit = True
            print(f"[KEY] 退出 (key={keycode})")

        elif keycode in (259, 8):  # Backspace (GLFW=259, ASCII=8)
            self.estop = True
            self.active_requested = False
            self.position_control = False
            self.rl_inference = False
            self.low_gain_mode = False
            self._arrow_vx = 0.0
            self._arrow_vy = 0.0
            self._arrow_yaw = 0.0
            print(f"[KEY] 急停 estop (key={keycode})")

        elif keycode in (65, 32):  # A, Space
            self.estop = False
            self.active_requested = True
            self.position_control = True
            self.rl_inference = False
            print(f"[KEY] 位置控制 → EnteringPosition (key={keycode})")

        elif keycode in (66,):  # B
            self.active_requested = False
            self.position_control = False
            self.rl_inference = False
            self.low_gain_mode = False
            self._arrow_vx = 0.0
            self._arrow_vy = 0.0
            self._arrow_yaw = 0.0
            print(f"[KEY] 停用 → Ready (key={keycode})")

        elif keycode in (88,):  # X
            if self.active_requested and not self.estop:
                self.rl_inference = True
                self.position_control = False
                print(f"[KEY] RL 推理 → ActiveRL (key={keycode})")

        elif keycode in (89,):  # Y
            if self.active_requested and not self.estop:
                self.rl_inference = False
                self.position_control = True
                print(f"[KEY] 位置控制 → ActivePC (key={keycode})")

        elif keycode in (76,):  # L
            self.low_gain_mode = not self.low_gain_mode
            print(f"[KEY] Low gain mode {'ON' if self.low_gain_mode else 'OFF'} (key={keycode})")

        # ── 方向键速度命令 ──────────────────────────────────────────
        elif keycode == 265:  # ↑ 前进
            self._arrow_vx = 0.6
            print(f"[KEY] ↑ vx=+{self._arrow_vx}")
        elif keycode == 264:  # ↓ 后退 (GLFW_KEY_DOWN = 264, not 266)
            self._arrow_vx = -0.4
            print(f"[KEY] ↓ vx={self._arrow_vx}")
        elif keycode == 263:  # ← 左移
            self._arrow_vy = 0.3
            print(f"[KEY] ← vy=+{self._arrow_vy}")
        elif keycode == 262:  # → 右移
            self._arrow_vy = -0.3
            print(f"[KEY] → vy={self._arrow_vy}")
        elif keycode == 32:  # Space (已在上面处理, 这里重置方向)
            pass
        elif keycode == 48:  # 0 键: 停止移动
            self._arrow_vx = 0.0
            self._arrow_vy = 0.0
            self._arrow_yaw = 0.0
            print(f"[KEY] 0 停止移动")

    def get_command(
        self, vx: float = 0.0, vy: float = 0.0, yaw_rate: float = 0.0
    ) -> OperatorCommand:
        """生成 OperatorCommand — 方向键优先于静态命令"""
        cmd = OperatorCommand()
        cmd.vx = self._arrow_vx if self._arrow_vx != 0.0 else vx
        cmd.vy = self._arrow_vy if self._arrow_vy != 0.0 else vy
        cmd.yaw_rate = self._arrow_yaw if self._arrow_yaw != 0.0 else yaw_rate
        cmd.estop = self.estop

        active = self.active_requested and not self.estop
        cmd.active = active
        cmd.position_control = self.position_control if active else False
        cmd.rl_inference = self.rl_inference if active else False
        cmd.low_gain_mode = self.low_gain_mode

        return cmd


class JoystickHandler:
    """把 /dev/input/js* 手柄接入仿真，与实机 bridge 使用同一套映射。

    接口与 KeyboardHandler 对齐 (process_keys / get_command / should_quit)，
    DeployController 无需区分输入源。
    """

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        axis_vx: int = 1,
        axis_vy: int = 0,
        axis_yaw: int = 2,
        # BEITONG BTP-KP70C (20bc:5174) native js0 defaults:
        #   A=0 B=1 X=3 Y=4; left stick axes 1/0, right stick X axis 2.
        # Back/Start are optional: M6=btn13 (low_gain), M5=btn14 (enable).
        # They are left unassigned by default; pass --joy-btn-back/--joy-btn-start
        # if you want the physical M5/M6 buttons.
        btn_a: int = 0,
        btn_b: int = 1,
        btn_x: int = 3,
        btn_y: int = 4,
        btn_back: Optional[int] = None,
        btn_start: Optional[int] = None,
        debug: bool = False,
    ):
        if not HAS_JOYSTICK:
            raise RuntimeError(
                f"无法导入 hardware/gamepad/opendoge_joystick.py: {_JOYSTICK_IMPORT_ERROR}"
            )
        self.device = device or first_joystick_device()
        if not self.device:
            raise FileNotFoundError(
                "未找到 /dev/input/js* 手柄设备。插入接收器后确认 ls /dev/input/js*"
            )
        self.joystick = LinuxJoystick(self.device)
        self.joystick.open()
        # 默认与 hardware/gamepad/xbox_command_bridge.py 一致；
        # 不同手柄可用 --joy-* 参数重映射。
        self.mapper = XboxCommandMapper(
            max_vx=0.6, max_vy=0.4, max_yaw_rate=1.0, deadzone=0.08,
            axis_vy=axis_vy, axis_vx=axis_vx, axis_yaw=axis_yaw,
            btn_a=btn_a, btn_b=btn_b, btn_x=btn_x, btn_y=btn_y,
            btn_back=btn_back, btn_start=btn_start,
        )
        self.should_quit = False
        self._lost = False
        self.debug = debug
        print(f"[JOY] {self.device} ({self.joystick.name()})")

    def process_keys(self):
        """非阻塞读取手柄事件 (命名与 KeyboardHandler 保持一致)。"""
        try:
            events = self.joystick.read_events()
            if self.debug:
                for e in events:
                    if e.base_type == JS_EVENT_BUTTON:
                        kind = "BTN "
                    elif e.base_type == JS_EVENT_AXIS:
                        kind = "AXIS"
                    else:
                        kind = "EV  "
                    print(f"[JOY] {kind} {e.number:2d} = {e.value:6d}")
            self.mapper.update(events)
            self._lost = False
        except DeviceLostError:
            if not self._lost:
                print("\n[JOY] 手柄设备丢失，强制 active=false")
                self.mapper.active_requested = False
                self.mapper.position_control = False
                self.mapper.rl_inference = False
                self._lost = True

    def get_command(
        self, vx: float = 0.0, vy: float = 0.0, yaw_rate: float = 0.0
    ) -> OperatorCommand:
        c = self.mapper.command()
        return OperatorCommand(
            vx=c.vx,
            vy=c.vy,
            yaw_rate=c.yaw_rate,
            active=c.active,
            estop=c.estop,
            position_control=c.position_control,
            rl_inference=c.rl_inference,
            low_gain_mode=c.low_gain_mode,
        )

    def close(self):
        self.joystick.close()


# ══════════════════════════════════════════════════════════════════════
# 日志记录
# ══════════════════════════════════════════════════════════════════════

class DeployLogger:
    """CSV 日志记录器 — 每个控制周期记录完整状态, 方便 debug。"""

    COLUMNS = [
        "t", "state", "fault",
        "cmd_vx", "cmd_vy", "cmd_yaw", "cmd_active", "cmd_pos", "cmd_rl",
        "z", "base_x", "base_y", "base_yaw",
        "true_vx", "true_vy", "true_wz", "phase", "rl_fb",
        # 12 joints × {q, dq, target, action, kp, kd, torque}
        *[f"{j}_{s}" for j in JOINT_NAMES for s in (
            "q", "dq", "target", "action", "kp", "kd", "torque",
        )],
    ]

    def __init__(self, log_path: str, decimation: int = 1):
        import csv
        self._file = open(log_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.COLUMNS)
        self._decimation = decimation
        self._step = 0
        self._last_torque = np.zeros(NUM_JOINTS)

    def record(
        self,
        t: float,
        state: "RuntimeState",
        fault_latched: bool,
        command: OperatorCommand,
        q: np.ndarray,
        dq: np.ndarray,
        target: np.ndarray,
        action: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        z: float,
        base_x: float,
        base_y: float,
        base_yaw: float,
        true_lin_vel: np.ndarray,
        true_wz: float,
        phase: float,
        rl_fallback: bool,
        torque: np.ndarray,
    ):
        self._step += 1
        if self._step % self._decimation != 0:
            return
        row = [
            f"{t:.6f}", state.name, int(fault_latched),
            f"{command.vx:.4f}", f"{command.vy:.4f}", f"{command.yaw_rate:.4f}",
            int(command.active), int(command.position_control), int(command.rl_inference),
            f"{z:.6f}", f"{base_x:.6f}", f"{base_y:.6f}", f"{base_yaw:.6f}",
            f"{true_lin_vel[0]:.6f}", f"{true_lin_vel[1]:.6f}", f"{true_wz:.6f}",
            f"{phase:.4f}", int(rl_fallback),
        ]
        for i in range(NUM_JOINTS):
            row += [
                f"{q[i]:.6f}", f"{dq[i]:.6f}", f"{target[i]:.6f}",
                f"{action[i]:.6f}", f"{kp[i]:.4f}", f"{kd[i]:.4f}",
                f"{torque[i]:.6f}",
            ]
        self._writer.writerow(row)

    def close(self):
        self._file.close()


# ══════════════════════════════════════════════════════════════════════
# 主控制回路 (复现 main.cpp 的完整逻辑)
# ══════════════════════════════════════════════════════════════════════

class DeployController:
    """复现 C++ opendoge_deploy 的完整控制回路"""

    def __init__(
        self,
        sim: OpenDogeSimulator,
        config: DeployConfig,
        keyboard: KeyboardHandler,
        mock_policy: Optional[MockPolicy] = None,
        static_vx: float = 0.0,
        static_vy: float = 0.0,
        static_yaw: float = 0.0,
        logger: Optional["DeployLogger"] = None,
        seed: int = 0,
    ):
        self.sim = sim
        self.config = config
        self.keyboard = keyboard
        self.policy = mock_policy
        self.static_vx = static_vx
        self.static_vy = static_vy
        self.static_yaw = static_yaw
        self.logger = logger
        self._rng = np.random.default_rng(seed)
        # Sensor history / loss state for --imu-delay-ms/--joint-delay-ms/
        # --packet-loss/--cmd-loss. The history is observation-only.
        self._sensor_history = []
        self._loss_rng = np.random.default_rng(seed + 0x5EED)
        self._loss_remaining = np.zeros(NUM_JOINTS)
        self._loss_remaining_bus = np.zeros(4)
        self._loss_hold_q = None
        self._loss_hold_dq = None
        self._cmd_loss_remaining = 0.0
        self._applied_target = DEFAULT_POS.copy()

        # 状态机变量
        self.runtime_state = RuntimeState.WaitFeedback
        self.fault_latched = False
        self.fault_reason = ""
        self.rl_fallback_active = False

        # 控制变量
        self.phase = 0.0
        self.phase_features = np.zeros(2)
        self.history = HistoryBuffer()
        self.history_ready = False
        self.rl_start_s = -1.0
        self.action = np.zeros(NUM_JOINTS)
        self.last_action = np.zeros(NUM_JOINTS)
        self.logical_target = DEFAULT_POS.copy()
        self.limited_target = DEFAULT_POS.copy()
        self.pc_startup_start_s = 0.0
        self.standup_start_s = 0.0
        self.standup_handover_start_s = -1.0

        # 快照时间戳
        self._last_infer_tick = -1
        self._last_target_tick = -1

    def _get_command(self) -> OperatorCommand:
        cmd = self.keyboard.get_command(
            vx=self.static_vx, vy=self.static_vy, yaw_rate=self.static_yaw
        )
        if cmd.rl_inference and cmd.position_control:
            cmd.position_control = False
        return cmd

    def reset_after_fall(self, t: float) -> None:
        """Reset the plant and start a fresh ActiveRL episode (used by --auto-reset)."""
        self.sim.reset_to_default_pose()
        # Keep simulation time continuous so --duration semantics stay intact.
        self.sim.data.time = t
        mujoco.mj_forward(self.sim.model, self.sim.data)
        command = self._get_command()
        self.fault_latched = False
        self.fault_reason = ""
        self.rl_fallback_active = False
        self.action.fill(0.0)
        self.last_action.fill(0.0)
        self.logical_target = DEFAULT_POS.copy()
        self.limited_target = DEFAULT_POS.copy()
        self.phase = 0.0
        self.phase_features = np.zeros(2)
        self.history.clear()
        self.history_ready = False
        self.rl_start_s = -1.0
        self.standup_handover_start_s = -1.0
        self._last_infer_tick = -1
        self._last_target_tick = -1
        self._sensor_history.clear()
        if command.active and command.rl_inference:
            self.runtime_state = RuntimeState.ActiveRL
            self.rl_start_s = t
            self.standup_handover_start_s = t
            frame0 = build_observation_frame(
                self.sim.get_joint_positions(),
                self.sim.get_joint_velocities(),
                DEFAULT_POS,
                self.last_action,
                command,
                self.sim.get_imu(),
                np.zeros(2),
            )
            self.history.reset(frame0)
            self.history_ready = True
        else:
            self.runtime_state = RuntimeState.Ready

    def _delayed_snapshot(self, t: float, delay_s: float):
        """Return the newest history entry at or before t-delay_s.

        Observation-only: the PD/plant must keep using the current state.
        """
        if delay_s <= 0.0 or not self._sensor_history:
            return None
        target = t - delay_s
        for entry in reversed(self._sensor_history):
            if entry[0] <= target:
                return entry
        return self._sensor_history[0]

    def _update_loss_state(self, config: DeployConfig, dt: float) -> None:
        """Update per-joint/bus/all packet-loss hold timers (target rate)."""
        max_burst = max(config.loss_burst_ms / 1000.0, dt)
        if config.loss_target == "all":
            if self._loss_remaining[0] <= 0.0 and self._loss_rng.random() < config.packet_loss:
                self._loss_remaining[:] = max_burst
            else:
                self._loss_remaining[:] = np.maximum(0.0, self._loss_remaining[:] - dt)
            return
        if config.loss_target == "bus":
            for b in range(4):
                if (self._loss_remaining_bus[b] <= 0.0
                        and self._loss_rng.random() < config.packet_loss):
                    self._loss_remaining_bus[b] = max_burst
                else:
                    self._loss_remaining_bus[b] = max(
                        0.0, self._loss_remaining_bus[b] - dt)
            for i in range(NUM_JOINTS):
                self._loss_remaining[i] = self._loss_remaining_bus[i // 3]
            return
        for i in range(NUM_JOINTS):
            if self._loss_remaining[i] <= 0.0 and self._loss_rng.random() < config.packet_loss:
                self._loss_remaining[i] = max_burst
            else:
                self._loss_remaining[i] = max(0.0, self._loss_remaining[i] - dt)

    def _check_startup_deviation(self) -> bool:
        """检查 ramp 期间关节偏离 DEFAULT_POS (防失控，与 C++ 一致)"""
        q = self.sim.get_joint_positions()
        for i in range(NUM_JOINTS):
            if abs(q[i] - DEFAULT_POS[i]) > self.config.pc_startup_max_deviation:
                return False
        return True

    def step(self) -> bool:
        """执行一个控制周期。返回 True 继续，False 退出。"""
        t = self.sim.data.time
        config = self.config
        prev_state = self.runtime_state
        # Observation-only sensor history (used by the delay model). Filled
        # every controller step so the delay resolution is the sim timestep.
        self._sensor_history.append((
            t,
            self.sim.get_imu(),
            self.sim.get_joint_positions().copy(),
            self.sim.get_joint_velocities().copy(),
        ))
        if len(self._sensor_history) > 2000:
            del self._sensor_history[:len(self._sensor_history) - 2000]
        self.keyboard.process_keys()
        command = self._get_command()

        # --- 急停 ---
        if command.estop and not self.fault_latched:
            self.fault_latched = True
            self.fault_reason = "operator estop"
            self.runtime_state = RuntimeState.DampingFault
            print(f"\n[{t:.3f}s] 故障锁存: {self.fault_reason}")

        # --- 状态转换 (与 C++ controller.cpp updateStateMachine 一致) ---
        if not self.fault_latched:

            # ── Mode priority: RL beats position control ──
            if command.rl_inference and command.position_control:
                command.position_control = False

            # ── WaitFeedback → Ready (MuJoCo always has "feedback" immediately) ──
            if self.runtime_state == RuntimeState.WaitFeedback:
                self.runtime_state = RuntimeState.Ready

            # ── Ready ↔ LowGainTest ──
            if self.runtime_state == RuntimeState.Ready and command.low_gain_mode:
                self.runtime_state = RuntimeState.LowGainTest
                # Start the stand-up from the measured pose, not from DEFAULT_POS.
                self.limited_target = self.sim.get_joint_positions().copy()
                print(f"\n[{t:.3f}s] Ready → LowGainTest")
            if self.runtime_state == RuntimeState.LowGainTest and not command.low_gain_mode:
                self.runtime_state = RuntimeState.Ready
                print(f"\n[{t:.3f}s] LowGainTest → Ready")

            # ── Ready → StandingUp / EnteringPosition / ActiveRL ──
            if self.runtime_state == RuntimeState.Ready and command.active:
                if not self._check_startup_deviation():
                    # Far from the default pose (e.g. prone): stand up first,
                    # no policy; the target ramps from the measured pose.
                    self.runtime_state = RuntimeState.StandingUp
                    self.standup_start_s = t
                    self.standup_handover_start_s = -1.0
                    self.limited_target = self.sim.get_joint_positions().copy()
                    print(f"\n[{t:.3f}s] Ready → StandingUp (远离站姿)")
                elif command.rl_inference:
                    self.runtime_state = RuntimeState.ActiveRL
                    print(f"\n[{t:.3f}s] Ready → ActiveRL (RL 直跳)")
                else:
                    self.runtime_state = RuntimeState.EnteringPosition
                    self.pc_startup_start_s = t
                    print(
                        f"\n[{t:.3f}s] Ready → EnteringPosition "
                        f"(斜坡 {config.pc_startup_ramp_s}s)"
                    )

            # ── EnteringPosition transitions ──
            elif self.runtime_state == RuntimeState.EnteringPosition:
                if not command.active:
                    self.runtime_state = RuntimeState.Ready
                    print(f"\n[{t:.3f}s] EnteringPosition → Ready (取消)")
                elif command.rl_inference:
                    self.runtime_state = RuntimeState.ActiveRL
                    print(f"\n[{t:.3f}s] EnteringPosition → ActiveRL")
                else:
                    if not self._check_startup_deviation():
                        self.fault_latched = True
                        self.fault_reason = (
                            "position control startup: joint deviation exceeds limit"
                        )
                        self.runtime_state = RuntimeState.DampingFault
                        print(f"\n[{t:.3f}s] 故障: {self.fault_reason}")
                    elif t - self.pc_startup_start_s >= config.pc_startup_ramp_s:
                        self.runtime_state = RuntimeState.ActivePC
                        print(f"\n[{t:.3f}s] EnteringPosition → ActivePC (斜坡完成)")

            # ── StandingUp: ramp to the default pose, no policy ──
            elif self.runtime_state == RuntimeState.StandingUp:
                if not command.active:
                    self.runtime_state = RuntimeState.Ready
                    print(f"\n[{t:.3f}s] StandingUp → Ready (取消)")
                elif t - self.standup_start_s >= (
                    config.standup_duration_s + config.standup_settle_s
                ):
                    self.runtime_state = (
                        RuntimeState.ActiveRL if command.rl_inference
                        else RuntimeState.ActivePC
                    )
                    self.standup_handover_start_s = t
                    print(f"\n[{t:.3f}s] StandingUp → {self.runtime_state.name}")

            # ── ActivePC / ActiveRL → Ready ──
            elif (
                self.runtime_state in (RuntimeState.ActivePC, RuntimeState.ActiveRL)
                and not command.active
            ):
                print(f"\n[{t:.3f}s] {self.runtime_state.name} → Ready")
                self.runtime_state = RuntimeState.Ready
                self.rl_fallback_active = False

            # ── ActivePC → ActiveRL ──
            elif self.runtime_state == RuntimeState.ActivePC and command.rl_inference:
                self.runtime_state = RuntimeState.ActiveRL
                self.rl_fallback_active = False
                print(f"\n[{t:.3f}s] ActivePC → ActiveRL")

            # ── ActiveRL → ActivePC ──
            elif (
                self.runtime_state == RuntimeState.ActiveRL
                and not command.rl_inference
            ):
                self.runtime_state = RuntimeState.ActivePC
                self.rl_fallback_active = False
                print(f"\n[{t:.3f}s] ActiveRL → ActivePC")

        # ── ActiveRL episode boundary (mirrors main.cpp) ──
        entered_rl = (
            prev_state != RuntimeState.ActiveRL
            and self.runtime_state == RuntimeState.ActiveRL
        )
        if entered_rl:
            self.rl_start_s = t
            self.standup_handover_start_s = t
            self.action.fill(0.0)
            self.last_action.fill(0.0)
            frame0 = build_observation_frame(
                self.sim.get_joint_positions(),
                self.sim.get_joint_velocities(),
                DEFAULT_POS,
                self.last_action,
                command,
                self.sim.get_imu(),
                np.zeros(2),
            )
            self.history.reset(frame0)
            self.history_ready = True
            self.phase = 0.0
            print(f"\n[{t:.3f}s] ActiveRL episode reset (5-frame history, phase=0)")
        left_rl = (
            prev_state == RuntimeState.ActiveRL
            and self.runtime_state != RuntimeState.ActiveRL
        )
        if left_rl:
            self.rl_start_s = -1.0
            self.history_ready = False
            self.history.clear()
            self.action.fill(0.0)
            self.last_action.fill(0.0)
            self.standup_handover_start_s = -1.0

        # --- 推理块 (50 Hz, ActiveRL only) ---
        infer_tick = int(t * config.inference_hz + 1.0e-9)
        if infer_tick != self._last_infer_tick:
            self._last_infer_tick = infer_tick
            if (
                self.runtime_state == RuntimeState.ActiveRL
                and command.active
                and self.policy is not None
            ):
                imu = self.sim.get_imu()
                q_obs = self.sim.get_joint_positions()
                dq_obs = self.sim.get_joint_velocities()
                # Observation-only delays: never feed these into the plant/PD.
                if config.imu_delay_ms > 0.0:
                    snap = self._delayed_snapshot(t, config.imu_delay_ms / 1000.0)
                    if snap is not None:
                        src = snap[1]
                        imu = ImuSample(
                            linear_velocity=src.linear_velocity.copy(),
                            angular_velocity=src.angular_velocity.copy(),
                            projected_gravity=src.projected_gravity.copy(),
                            valid=src.valid,
                        )
                if config.joint_delay_ms > 0.0:
                    snap = self._delayed_snapshot(t, config.joint_delay_ms / 1000.0)
                    if snap is not None:
                        q_obs = snap[2].copy()
                        dq_obs = snap[3].copy()
                if config.packet_loss > 0.0:
                    if self._loss_hold_q is None:
                        self._loss_hold_q = q_obs.copy()
                        self._loss_hold_dq = dq_obs.copy()
                    for i in range(NUM_JOINTS):
                        if self._loss_remaining[i] > 0.0:
                            q_obs[i] = self._loss_hold_q[i]
                            dq_obs[i] = self._loss_hold_dq[i]
                        else:
                            self._loss_hold_q[i] = q_obs[i]
                            self._loss_hold_dq[i] = dq_obs[i]

                elapsed = max(0.0, t - self.rl_start_s)
                steps = int(round(elapsed / max(1.0 / config.inference_hz, 1e-6)))
                period_steps = max(1, int(round(config.gait_period_s * config.inference_hz)))
                self.phase = (steps % period_steps) / period_steps
                phase_features = gait_phase_features(
                    elapsed,
                    command,
                    1.0 / config.inference_hz,
                    config.gait_period_s,
                    config.gait_command_threshold,
                )
                if not self.history_ready:
                    # Rebuild after a skipped tick; do not infer on a
                    # non-time-regular history.
                    frame0 = build_observation_frame(
                        q_obs, dq_obs, DEFAULT_POS, self.last_action,
                        command, imu, np.zeros(2),
                    )
                    self.history.reset(frame0)
                    self.history_ready = True
                    self.rl_start_s = t
                    self.action.fill(0.0)
                else:
                    frame = build_observation_frame(
                        q_obs, dq_obs, DEFAULT_POS, self.last_action,
                        command, imu, phase_features,
                    )
                    self.history.append(frame)
                    success, action, error = self.policy.infer(self.history.flatten())
                    if not success:
                        self.runtime_state = RuntimeState.ActivePC
                        self.rl_fallback_active = True
                        self.history_ready = False
                        self.history.clear()
                        self.action.fill(0.0)
                        self.last_action.fill(0.0)
                        print(f"\n[{t:.3f}s] RL 推理失败 → ActivePC (降级): {error}")
                    else:
                        self.action = action
            else:
                self.action.fill(0.0)

        # --- 目标计算块 (200 Hz, 与 C++ controller.cpp updateTargets 一致) ---
        target_tick = int(t * config.target_hz)
        if target_tick != self._last_target_tick:
            self._last_target_tick = target_tick
            # 进入 ActiveRL 的交接淡入：第一帧策略动作可能很大，先从默认姿态淡入，
            # 避免 calf 目标瞬跳（与 controller.cpp updateTargets 一致）。
            handover = (
                self.runtime_state == RuntimeState.ActiveRL
                and self.standup_handover_start_s >= 0.0
                and t - self.standup_handover_start_s < config.standup_handover_s
            )
            alpha = 1.0
            if handover:
                alpha = min(
                    (t - self.standup_handover_start_s)
                    / max(config.standup_handover_s, 1e-6),
                    1.0,
                )
            for i in range(NUM_JOINTS):
                if self.runtime_state == RuntimeState.ActiveRL:
                    # 训练无裁剪；last_action 观测原始策略输出
                    self.last_action[i] = self.action[i]
                    # 动作在训练约定，映射回固件约定；逐关节 action scale
                    tgt = (DEFAULT_POS[i]
                           + JOINT_SIGN[i] * self.last_action[i] * config.action_scale[i])
                else:
                    self.last_action[i] = 0.0
                    tgt = DEFAULT_POS[i]
                lower = JOINT_LOWER[i]
                upper = JOINT_UPPER[i]
                if (self.runtime_state == RuntimeState.ActiveRL
                        and config.rl_joint_limit_margin > 0.0):
                    margin = min(
                        config.rl_joint_limit_margin, 0.5 * (upper - lower))
                    lower += margin
                    upper -= margin
                tgt = np.clip(tgt, lower, upper)
                self.logical_target[i] = tgt
                if handover:
                    blended = DEFAULT_POS[i] + alpha * (tgt - DEFAULT_POS[i])
                    self.limited_target[i] = rate_limit(
                        blended, self.limited_target[i], MAX_POSITION_STEP
                    )
                # RL 模式跳过 rate_limit，策略训练时目标瞬时切换，无平滑延迟
                elif self.runtime_state == RuntimeState.ActiveRL:
                    self.limited_target[i] = tgt
                else:
                    step = MAX_POSITION_STEP
                    if self.runtime_state == RuntimeState.LowGainTest:
                        step = min(step, LOW_GAIN_MAX_POSITION_STEP)
                    elif self.runtime_state == RuntimeState.StandingUp:
                        step = min(step, STANDUP_MAX_POSITION_STEP)
                    self.limited_target[i] = rate_limit(
                        self.logical_target[i], self.limited_target[i], step
                    )

            # Loss models are driven at the 200 Hz target rate. A dropped
            # command holds the last applied target (the motor keeps its last
            # CAN command); a feedback drop holds the last observed q/dq.
            tick = 1.0 / max(config.target_hz, 1.0)
            if config.cmd_loss > 0.0:
                if (self._cmd_loss_remaining <= 0.0
                        and self._loss_rng.random() < config.cmd_loss):
                    self._cmd_loss_remaining = max(
                        config.loss_burst_ms / 1000.0, tick)
                else:
                    self._cmd_loss_remaining = max(
                        0.0, self._cmd_loss_remaining - tick)
            if config.packet_loss > 0.0:
                self._update_loss_state(config, tick)

        # --- 控制块 (1000 Hz, 与 C++ controller.cpp computeMotorCommands 一致) ---
        is_active = self.runtime_state in (RuntimeState.ActivePC, RuntimeState.ActiveRL)
        is_ramping = self.runtime_state == RuntimeState.EnteringPosition
        is_low_gain = self.runtime_state == RuntimeState.LowGainTest
        is_standing_up = self.runtime_state == RuntimeState.StandingUp

        if not is_active and not is_ramping and not is_low_gain and not is_standing_up:
            # Ready / WaitFeedback / DampingFault: 关节全失能, 仅阻尼
            kp_gains = np.zeros(NUM_JOINTS)
            kd_gains = np.full(NUM_JOINTS, config.safe_kd)
            target_positions = self.sim.get_joint_positions()
        elif is_low_gain:
            # LowGainTest: 30% gains, hold DEFAULT_POS
            kp_gains = np.full(NUM_JOINTS, config.kp * 0.3)
            kd_gains = np.full(NUM_JOINTS, config.kd * 0.3)
            target_positions = DEFAULT_POS.copy()
        elif is_standing_up:
            # StandingUp: 30% -> full stiffness, target ramps to DEFAULT_POS
            frac = min(
                (t - self.standup_start_s) / max(config.standup_duration_s, 1e-6), 1.0
            )
            kp_gains = np.full(NUM_JOINTS, config.kp * (0.3 + 0.7 * frac))
            kd_gains = np.full(
                NUM_JOINTS,
                config.safe_kd + frac * (config.kd - config.safe_kd),
            )
            target_positions = self.limited_target
        elif is_ramping:
            ramp_elapsed = t - self.pc_startup_start_s
            ramp_frac = min(ramp_elapsed / config.pc_startup_ramp_s, 1.0)
            kp_gains = np.full(NUM_JOINTS, ramp_frac * config.kp)
            kd_gains = np.full(
                NUM_JOINTS,
                config.safe_kd + ramp_frac * (config.kd - config.safe_kd),
            )
            target_positions = self.limited_target
        else:
            kp_gains = np.full(NUM_JOINTS, config.kp)
            kd_gains = np.full(NUM_JOINTS, config.kd)
            target_positions = self.limited_target

        if config.cmd_loss > 0.0:
            if self._cmd_loss_remaining > 0.0:
                target_positions = self._applied_target.copy()
            else:
                self._applied_target = target_positions.copy()

        self.sim.apply_pd_control(target_positions, kp_gains, kd_gains)

        # ── 日志: 记录控制输出 ──────────────────────────────────────────
        if self.logger is not None:
            q = self.sim.get_joint_positions()
            dq = self.sim.get_joint_velocities()
            qw, qx, qy, qz = self.sim.data.qpos[3:7]
            base_yaw = math.atan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy * qy + qz * qz),
            )
            log_imu = self.sim.get_imu()
            if self.sim._has_actuators:
                torque = np.array([self.sim.data.ctrl[self.sim._actuator_ids[i]]
                                   for i in range(NUM_JOINTS)])
            else:
                torque = np.array([self.sim.data.qfrc_applied[self.sim._dof_adr[i]]
                                   for i in range(NUM_JOINTS)])
            self.logger.record(
                t=t,
                state=self.runtime_state,
                fault_latched=self.fault_latched,
                command=command,
                q=q, dq=dq,
                target=target_positions,
                action=self.last_action,
                kp=kp_gains, kd=kd_gains,
                z=self.sim.data.qpos[2],
                base_x=self.sim.data.qpos[0],
                base_y=self.sim.data.qpos[1],
                base_yaw=base_yaw,
                true_lin_vel=log_imu.linear_velocity,
                true_wz=log_imu.angular_velocity[2],
                phase=self.phase,
                rl_fallback=self.rl_fallback_active,
                torque=torque,
            )

        # 物理步进
        self.sim.step()

        if self.runtime_state == RuntimeState.DampingFault:
            self.fault_latched = True

        return not self.keyboard.should_quit

    def status_line(self) -> str:
        """生成状态行 (与 main.cpp 1Hz 输出一致)"""
        ramp_pct = 100
        if self.runtime_state == RuntimeState.EnteringPosition:
            elapsed = self.sim.data.time - self.pc_startup_start_s
            ramp_pct = min(int(100.0 * elapsed / self.config.pc_startup_ramp_s), 100)

        cmd = self._get_command()
        q = self.sim.get_joint_positions()
        z = self.sim.data.qpos[2]

        line = (
            f"state={self.runtime_state.name:<18} "
            f"active_cmd={1 if cmd.active else 0} "
            f"pos_ctrl={1 if cmd.position_control else 0} "
            f"rl_infer={1 if cmd.rl_inference else 0} "
            f"ramp_pct={ramp_pct:>3} "
            f"rl_fb={1 if self.rl_fallback_active else 0} "
            f"phase={self.phase:.2f} hist={1 if self.history_ready else 0} "
            f"z={z:.4f} "
            f"q0={q[0]:+.3f} q1={q[1]:+.3f} q2={q[2]:+.3f}"
        )
        if self.fault_latched:
            line += f' fault="{self.fault_reason}"'
        return line


# ══════════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════════

def find_training_root(explicit: str = "") -> str:
    """Locate the OpenDoge_train checkout that owns the training XML/STLs."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        explicit,
        os.path.join(repo_root, "..", "OpenDoge_train"),
        os.path.join(repo_root, "..", "..", "OpenDoge_train"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        candidate = os.path.abspath(candidate)
        xml = os.path.join(
            candidate, "src", "assets", "robots", "opendoge", "xmls", "opendoge.xml"
        )
        if os.path.isfile(xml):
            return candidate
    return ""


def training_scene_path(training_root: str) -> str:
    return os.path.join(
        training_root, "src", "assets", "robots", "opendoge", "xmls",
        "scene_opendoge.xml",
    )


def find_model_path() -> str:
    """查找固件 MuJoCo XML 模型路径 (legacy/firmware convention)."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(repo_root, "docs", "URDF", "xml", "scene.xml"),
        os.path.join(repo_root, "docs", "URDF", "xml", "Opendoge.xml"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "找不到 MuJoCo XML 模型文件。\n"
        f"尝试过的路径: {candidates}\n"
        "请使用 --model 参数指定路径。"
    )

"""
脚本填写参数声明区，不懂的怎么填写的时候记得来看看，或是要增加参数时
"""

def parse_args():
    p = argparse.ArgumentParser(
        description="OpenDoge MuJoCo 仿真测试 — 复现 deploy 控制回路",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
键盘控制:
  A / Space  激活位置控制
  B          停用 (阻尼)
  X          切换到 RL 推理
  Y          切换回位置控制
  Backspace  急停
  Esc / Q    退出
""",
    )
    p.add_argument(
        "--model", type=str, default="",
        help="MuJoCo XML 模型文件路径 (默认自动查找)"
    )
    p.add_argument(
        "--mode", choices=["pc", "rl", "idle"], default="idle",
        help="初始模式"
    )
    p.add_argument(
        "--duration", type=float, default=0.0,
        help="运行时长 (秒), 0=直到手动退出"
    )
    p.add_argument(
        "--cmd", nargs=3, type=float, default=[0.0, 0.0, 0.0],
        metavar=("VX", "VY", "YAW"), help="静态速度命令"
    )
    p.add_argument("--no-render", action="store_true", help="无渲染模式")
    p.add_argument(
        "--joystick", nargs="?", const="auto", default=None,
        metavar="DEVICE",
        help="使用 /dev/input/js* 手柄控制 (与实机同一映射)。"
             "不带值自动选择第一个设备，也可指定 /dev/input/js0"
    )
    # 手柄映射覆盖 (默认值与实机 bridge 一致)
    p.add_argument("--joy-axis-vx", type=int, default=1, help="vx 轴号 (默认 1)")
    p.add_argument("--joy-axis-vy", type=int, default=0, help="vy 轴号 (默认 0)")
    p.add_argument("--joy-axis-yaw", type=int, default=2, help="yaw 轴号 (默认 2)")
    p.add_argument("--joy-btn-a", type=int, default=0, help="A 键号 (默认 0)")
    p.add_argument("--joy-btn-b", type=int, default=1, help="B 键号 (默认 1)")
    p.add_argument("--joy-btn-x", type=int, default=3, help="X 键号 (默认 3)")
    p.add_argument("--joy-btn-y", type=int, default=4, help="Y 键号 (默认 4)")
    p.add_argument("--joy-btn-back", type=int, default=None,
                   help="BACK/low_gain 键号 (BEITONG 可选 M6=13；默认不指定)")
    p.add_argument("--joy-btn-start", type=int, default=None,
                   help="START/使能 键号 (BEITONG 可选 M5=14；默认不指定)")
    p.add_argument("--joy-debug", action="store_true",
                   help="打印原始手柄事件 (确认轴/按键编号时使用)")
    p.add_argument(
        "--ramp", type=float, default=2.0,
        help="位置控制启动斜坡时长 (秒)"
    )
    p.add_argument(
        "--max-deviation", type=float, default=2.0,
        help="启动时关节最大允许偏差 (rad)"
    )
    p.add_argument(
        "--rl-failure-rate", type=float, default=0.0,
        help="模拟 RL 推理失败率 (0-1, 用于测试降级)"
    )
    p.add_argument(
        "--imu-delay-ms", type=float, default=0.0,
        help="观测路径的 IMU 延迟 (ms)；PD/plant 仍用当前状态"
    )
    p.add_argument(
        "--joint-delay-ms", type=float, default=0.0,
        help="观测路径的关节 q/dq 延迟 (ms)；PD/plant 仍用当前状态"
    )
    p.add_argument(
        "--packet-loss", type=float, default=0.0,
        help="每个 target tick 的关节反馈丢包概率 (0-1)"
    )
    p.add_argument(
        "--loss-burst-ms", type=float, default=0.0,
        help="丢包 burst 持续时间 (ms)；0 = 一个 target tick"
    )
    p.add_argument(
        "--loss-target", choices=["joint", "bus", "all"], default="joint",
        help="丢包作用对象: joint=单关节, bus=整条腿/总线, all=全部"
    )
    p.add_argument(
        "--cmd-loss", type=float, default=0.0,
        help="每个 target tick 的命令丢包概率 (保持上一目标)"
    )
    p.add_argument(
        "--policy", type=str, default="",
        help="ONNX 策略模型路径 (使用真实 RL 推理替代 mock 策略)"
    )
    p.add_argument(
        "--log-dir", type=str, default="logs",
        help="日志输出目录 (默认 logs/)"
    )
    p.add_argument(
        "--training-root", type=str, default="",
        help="OpenDoge_train 仓库路径 (默认自动查找同级 ../OpenDoge_train)"
    )
    p.add_argument(
        "--terrain", choices=["flat", "rough", "stairs", "boxes", "slope"],
        default="flat",
        help="仿真地形 (flat/rough/stairs/boxes/slope)"
    )
    p.add_argument(
        "--terrain-level", type=int, default=1,
        help="地形难度 0-3 (rough/stairs/boxes)"
    )
    p.add_argument(
        "--terrain-seed", type=int, default=0,
        help="地形随机种子"
    )
    p.add_argument(
        "--fall-angle-deg", type=float, default=70.0,
        help="摔倒判定角 (度, 从竖直方向)"
    )
    p.add_argument(
        "--auto-reset", action="store_true",
        help="摔倒后自动重置到默认姿态并继续 (地形批量测试用)"
    )
    p.add_argument(
        "--joint-convention", choices=["firmware", "training"],
        default="training",
        help="模型关节符号约定: training=OpenDoge_train mjlab XML (默认), firmware=docs/URDF/xml"
    )
    p.add_argument(
        "--start-pose", choices=["rest", "default"], default="rest",
        help="初始姿态: rest=趴伏 (Ready), default=站立默认位 (RL/地形测试)"
    )
    p.add_argument(
        "--base-height", type=float, default=0.158,
        help="--start-pose default 时的 base 高度 (m)"
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="随机种子"
    )
    return p.parse_args()


def key_callback_wrapper(keyboard: KeyboardHandler):
    """返回一个适合 mujoco viewer 的 key callback"""
    def callback(keycode: int):
        keyboard.feed_key(keycode)
    return callback


def main():
    global JOINT_LOWER, JOINT_UPPER
    args = parse_args()

    if not HAS_MUJOCO:
        print("错误: MuJoCo 未安装。运行: pip install mujoco", file=sys.stderr)
        sys.exit(1)

    training_root = find_training_root(args.training_root)
    if args.model:
        model_path = args.model
    elif training_root:
        model_path = training_scene_path(training_root)
    else:
        model_path = find_model_path()
    if args.terrain != "flat":
        if not training_root:
            print(
                "错误: --terrain 需要 OpenDoge_train 训练资产 "
                "(--training-root 或同级 ../OpenDoge_train)",
                file=sys.stderr,
            )
            sys.exit(1)
        from terrain import build_scene
        model_path = build_scene(
            training_root, args.terrain, args.terrain_level, args.terrain_seed
        )
    print(f"训练仓库: {training_root or '(未找到, 使用固件模型)'}")
    print(f"模型: {model_path}")

    if args.joint_convention == "training":
        JOINT_LOWER = TRAINING_JOINT_LOWER.copy()
        JOINT_UPPER = TRAINING_JOINT_UPPER.copy()

    config = DeployConfig(
        pc_startup_ramp_s=args.ramp,
        pc_startup_max_deviation=args.max_deviation,
        imu_delay_ms=args.imu_delay_ms,
        joint_delay_ms=args.joint_delay_ms,
        packet_loss=args.packet_loss,
        loss_burst_ms=args.loss_burst_ms,
        loss_target=args.loss_target,
        cmd_loss=args.cmd_loss,
    )
    keyboard = KeyboardHandler(quit_only=args.joystick is not None)
    command_source = keyboard
    joystick: Optional[JoystickHandler] = None
    if args.joystick is not None:
        try:
            joystick = JoystickHandler(
                None if args.joystick in ("", "auto") else args.joystick,
                axis_vx=args.joy_axis_vx,
                axis_vy=args.joy_axis_vy,
                axis_yaw=args.joy_axis_yaw,
                btn_a=args.joy_btn_a,
                btn_b=args.joy_btn_b,
                btn_x=args.joy_btn_x,
                btn_y=args.joy_btn_y,
                btn_back=args.joy_btn_back,
                btn_start=args.joy_btn_start,
                debug=args.joy_debug,
            )
        except (FileNotFoundError, RuntimeError, OSError) as exc:
            print(f"错误: {exc}", file=sys.stderr)
            sys.exit(1)
        command_source = joystick

    if args.policy:
        print(f"ONNX 策略: {args.policy}")
        mock_policy = OnnxPolicy(args.policy)
    else:
        mock_policy = MockPolicy(failure_rate=args.rl_failure_rate)

    if args.mode == "pc":
        keyboard.active_requested = True
        keyboard.position_control = True
        if joystick is not None:
            joystick.mapper.active_requested = True
            joystick.mapper.position_control = True
    elif args.mode == "rl":
        keyboard.active_requested = True
        keyboard.rl_inference = True
        if joystick is not None:
            joystick.mapper.active_requested = True
            joystick.mapper.rl_inference = True

    sim = OpenDogeSimulator(
        model_path, render=not args.no_render,
        key_callback=key_callback_wrapper(keyboard) if not args.no_render else None,
        joint_convention=args.joint_convention,
    )
    if args.start_pose == "default":
        sim.reset_to_default_pose(base_height=args.base_height)
    else:
        sim.reset_to_rest_pose()

    print(f"站立默认位 (DEFAULT_POS, 策略训练用): {dict(zip(JOINT_NAMES, DEFAULT_POS))}")
    print(f"趴伏初始位 (REST_POSE, Ready 态):       {dict(zip(JOINT_NAMES, REST_POSE))}")
    print()
    print(
        f"配置: kp={config.kp}, kd={config.kd}, safe_kd={config.safe_kd}, "
        f"action_scale={config.action_scale.tolist()}"
    )
    print(f"斜坡: {config.pc_startup_ramp_s}s, 最大偏差: {config.pc_startup_max_deviation}rad")
    print(f"初始模式: {args.mode}")
    print(f"速度命令: vx={args.cmd[0]}, vy={args.cmd[1]}, yaw={args.cmd[2]}")
    print(f"RL 失败率: {args.rl_failure_rate}")
    print()
    print("═" * 80)
    if joystick is not None:
        print(f"手柄控制 ({joystick.device}):")
        print("  左摇杆     → vx / vy     右摇杆 X → yaw_rate")
        print("  A          → 激活 + 位置控制      B → 停用")
        print("  X          → RL 推理              Y → 位置控制")
        print("  START(LB)  → 切换使能             BACK(RB) → 切换 low_gain")
        print("  Esc / Q    → 退出 (终端 Ctrl+C 同样有效)")
    else:
        print("键盘控制:")
        print("  A / Space  → 位置控制    B → 停用")
        print("  X          → RL 推理     Y → 位置控制")
        print("  ↑ / ↓      → 前进/后退   ← / → → 左移/右移")
        print("  0          → 停止移动")
        print("  Backspace  → 急停        Esc / Q → 退出")
    print("═" * 80)
    print()

    # ── 日志 ──────────────────────────────────────────────────────────
    logger = None
    if args.log_dir:
        # 相对于脚本仓库根目录
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.isabs(args.log_dir):
            log_dir = os.path.join(repo_root, args.log_dir)
        else:
            log_dir = args.log_dir
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        mode_tag = args.mode
        policy_tag = "onnx" if args.policy else "mock"
        log_path = os.path.join(
            log_dir, f"deploy_{mode_tag}_{policy_tag}_{timestamp}.csv"
        )
        logger = DeployLogger(log_path)
        print(f"日志: {log_path}")

    controller = DeployController(
        sim=sim, config=config, keyboard=command_source,
        mock_policy=mock_policy,
        static_vx=args.cmd[0], static_vy=args.cmd[1], static_yaw=args.cmd[2],
        logger=logger,
        seed=args.seed,
    )

    start_time = time.time()
    last_status_time = start_time
    step_count = 0
    fall_count = 0
    fall_latched = False
    paused = False

    try:
        while sim.is_running() and not keyboard.should_quit:
            # 检查 duration
            if args.duration > 0 and sim.data.time >= args.duration:
                print(f"\n达到设定运行时长 {args.duration}s，退出。")
                break

            # 键盘输入 (通过 GLFW 回调)
            if sim.viewer is not None:
                # MuJoCo viewer 通过 with 上下文获取 key events
                # 我们在这里轮询
                pass

            if not paused:
                # 手柄模式下键盘只负责 Esc/Q 退出
                if joystick is not None:
                    keyboard.process_keys()
                if not controller.step():
                    break
                step_count += 1
                if controller.runtime_state == RuntimeState.ActiveRL:
                    gz = sim.get_imu().projected_gravity[2]
                    fallen = gz > -math.cos(math.radians(args.fall_angle_deg))
                    if fallen and not fall_latched:
                        fall_count += 1
                        fall_latched = True
                        print(
                            f"\n[{sim.data.time:.3f}s] FALL detected "
                            f"(gravity.z={gz:.3f}, angle>{args.fall_angle_deg:.0f} deg)"
                        )
                        if args.auto_reset:
                            controller.reset_after_fall(sim.data.time)
                            fall_latched = False
                            print(f"[{sim.data.time:.3f}s] auto-reset -> ActiveRL")
                    elif not fallen:
                        fall_latched = False
            else:
                # 暂停时仍推进渲染
                sim.step_count += 1

            # 每秒打印状态
            now = time.time()
            if now - last_status_time >= 1.0:
                print(f"\r{controller.status_line()}", end="", flush=True)
                last_status_time = now

            # 渲染
            sim.render()

    except KeyboardInterrupt:
        print("\n用户中断 (Ctrl+C)")
    finally:
        sim_time = sim.data.time
        if joystick is not None:
            joystick.close()
        sim.close()
        if logger is not None:
            logger.close()
            size_kb = os.path.getsize(log_path) / 1024
            print(f"日志已保存: {log_path} ({size_kb:.1f} KB)")
        elapsed = time.time() - start_time
        print(
            f"\n仿真结束。运行 {elapsed:.1f}s, "
            f"物理时间 {sim_time:.2f}s, "
            f"{step_count} 步, "
            f"fall={fall_count}, "
            f"平均 {step_count / max(elapsed, 0.001):.0f} Hz",
            flush=True,
        )


if __name__ == "__main__":
    main()
    # GLFW viewer 在 Python 析构阶段会 segfault, 直接干净退出
    os._exit(0)

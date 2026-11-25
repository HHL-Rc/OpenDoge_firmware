# OpenDoge SuperMimic 部署策略

本目录保存 OpenDoge_train 训练并导出的部署模型。

## 当前策略

- `opendoge_supermimic_rough_20000.onnx`
  - 来源：`/home/hhl/RC/OpenDoge_train/policy/opendoge_supermimic_rough_20000.onnx`
  - 训练任务：`Opendoge-SuperMimic-Rough`
  - 训练轮次：20,000
  - 算法：SuperMimic `moe_history`（MoE 历史编码器 + 盲本体观测）
  - 导出精度：FP32（源网络 BF16）
- `task_config.json`
  - 训练任务快照（来源 `OpenDoge_train/policy/task_config.json`），用于追溯；运行时**不读取**。

## 部署契约（不可变）

| 项目 | 值 |
|------|-----|
| ONNX 输入 | `obs_history` `[batch, 235]` |
| ONNX 输出 | `actions` `[batch, 12]` |
| 单帧观测 | 47 维 |
| 历史 | 5 帧，旧→新，重置时重复首帧 |
| 控制周期 | 0.02 s（50 Hz） |
| 动作语义 | 原始关节位置偏移，无裁剪 |
| 动作缩放 | hip 0.25 / thigh 0.4 / calf 0.4 |
| 默认姿态（训练系） | `[0,0.8,-1.6, 0,-0.8,1.6, 0,0.8,-1.6, 0,-0.8,1.6]` |
| 命令系 | body xy + world yaw |
| 本体线速度 | 不进入观测（策略内部估计） |

47 维单帧布局：

```
base_ang_vel(3) + projected_gravity(3)
+ joint_pos_rel(12) + joint_vel_rel(12) + last_action(12)
+ command(3) + gait_phase_sin_cos(2) = 47
```

步态相位：

```
steps = round(episode_elapsed_time / 0.02)
p = (steps mod 25) / 25
moving = hypot(vx, vy) + abs(yaw_rate) > 0.1
features = moving ? [sin(2πp), cos(2πp)] : [0, 0]
```

## 运行

### C++ 单次推理校验

```bash
./install/opendoge_deploy/bin/opendoge_deploy \
  --policy-backend onnx \
  --policy-path policy/opendoge_supermimic_rough_20000.onnx \
  --verify-policy
```

### C++ dry-run

```bash
./install/opendoge_deploy/bin/opendoge_deploy \
  --policy-backend onnx \
  --policy-path policy/opendoge_supermimic_rough_20000.onnx \
  --start-rl --allow-missing-imu --duration-sec 3
```

### sim2sim

```bash
cd sim2sim
python3 deploy_mujoco.py \
  --training-root ../../OpenDoge_train \
  --start-pose default --mode rl \
  --policy ../policy/opendoge_supermimic_rough_20000.onnx \
  --cmd 0.4 0 0 --duration 10 --no-render
```

### 真机

```bash
POLICY_PATH=policy/opendoge_supermimic_rough_20000.onnx \
  bash scripts/start_robot.sh policy
```

操作：A（PC 站立）→ X（ActiveRL）→ 摇杆；B 回阻尼。

## 旧模型

`policy/legacy_48dim/` 保存旧的 48 维 LainLab/UniLab 模型：

- `opendoge_flat.onnx`
- `opendoge_r5.onnx`（49 维，更旧）

它们**不兼容**当前运行时（输入 235 维），仅作历史参考。

## 相关文档

- `docs/opendoge_supermimic_migration.md`：完整迁移记录、验证结果、风险与回滚。
- `sim2sim/README.md`：仿真与地形用法。

# OpenDoge_train SuperMimic 策略部署迁移文档

> 日期: 2026-11-17
> 范围: `Lain_Code` 的 C++ 部署运行时、MuJoCo sim2sim、配置、测试、地形测试与文档
> 训练侧: `/home/hhl/RC/OpenDoge_train`（mjlab + SuperMimic MoE 历史策略）
> 策略: `policy/opendoge_supermimic_rough_20000.onnx`（20,000 轮，Rough）

---

## 1. 概要

本次修改把部署侧从 LainLab/mjlab 的 48 维速度跟踪策略迁移到 OpenDoge_train 的 SuperMimic 策略。新策略是**盲本体策略**：不使用 `base_lin_vel`，而是把 5 帧历史（235 维）送入 ONNX，由内部 MoE 历史编码器估计机身速度。

**最重要的不兼容变更**：

| 项目 | 旧（LainLab 48 维） | 新（OpenDoge_train SuperMimic） |
|------|--------------------|---------------------------------|
| 单帧观测 | 48 维，含 `base_lin_vel` | **47 维，无 `base_lin_vel`** |
| 历史 | 无 | **5 帧，旧→新，235 维扁平输入** |
| 输入名 | `obs` | `obs_history` |
| 控制频率 | 100 Hz | **50 Hz**（`control_dt=0.02`） |
| 动作缩放 | 单一 0.25 | **hip 0.25 / thigh 0.4 / calf 0.4** |
| 步态相位 | 无（端到端） | **有：0.5 s 周期、25 步、移动门控** |
| 策略运行状态 | ActivePC/EnteringPosition/ActiveRL | **仅 ActiveRL**；PC 保持默认站姿 |
| 旧模型 | `opendoge_flat.onnx` | **不再兼容**，归档为 legacy |

> 旧 48 维 ONNX 在新运行时下会因输入维度不匹配加载失败。旧模型不能与新模型共存。

---

## 2. 变更文件清单

| 文件 | 变更类型 | 主要内容 |
|------|----------|----------|
| `deploy/include/opendoge_deploy/types.hpp` | 契约 | `kObsDim=47`、`kObsHistoryLength=5`、`kObsHistoryDim=235`、逐关节 `action_scale`、gait 参数、删除 `lin_vel_source` |
| `deploy/include/opendoge_deploy/observer.hpp` | 契约 | `buildObservationFrame`、`gaitPhaseFeatures`、`HistoryBuffer` |
| `deploy/src/observer.cpp` | 契约 | 47 维 frame、历史扁平化、训练相位公式 |
| `deploy/include/opendoge_deploy/policy.hpp` | 契约 | `PolicyInfo`、`infer(const ObsHistory&)` |
| `deploy/src/policy.cpp` | 契约 | `NonePolicy` 提供默认契约；删除 `linear_csv` |
| `deploy/src/onnx_policy.cpp` | 契约 | 235 维输入、动态输入名、metadata 校验、有限性检查 |
| `deploy/src/controller.cpp` | 控制 | 仅 ActiveRL 用动作；逐关节 scale；PC 保持默认 |
| `deploy/src/main.cpp` | 控制 | 历史/相位状态、RL 入口重置、仅 ActiveRL 推理、交接淡入、`--start-rl`、`--verify-policy` |
| `deploy/src/runtime_io.cpp` | 配置 | 解析 `action_scale_hip/thigh/calf`、gait 参数；删除 `lin_vel_source` |
| `deploy/src/cli.cpp` / `cli.hpp` | CLI | 新增 `--start-rl`、`--verify-policy`；删除 `--lin-vel-source` |
| `deploy/configs/opendoge_deploy.conf` | 配置 | `inference_hz=50`、`kd=0.5`、逐关节 scale、gait 参数 |
| `sim2sim/deploy_mujoco.py` | 仿真 | 47×5 契约、逐关节 scale、训练物理、地形 |
| `sim2sim/terrain.py` | 仿真 | 生成 flat/rough/stairs/boxes/slope 场景（不复制训练资产） |
| `test/deploy_gap_regression.py` | 测试 | 47×5 契约回归（49 项） |
| `test/policy_contract_test.py` | 测试 | ONNX metadata + golden action（21 项） |
| `test/terrain_sweep.py` | 测试 | 多地形批量测试 |
| `docs/opendoge_supermimic_migration.md` | 文档 | 本文档 |
| `scripts/start_robot.sh` | 脚本 | 默认策略路径改为新模型 |
| `policy/README.md` | 文档 | 新策略契约与运行方式 |

---

## 3. 详细变更

### 3.1 观测契约：48 维 → 47×5

新单帧 47 维：

```
base_ang_vel(3) + projected_gravity(3)
+ joint_pos_rel(12) + joint_vel_rel(12) + last_action(12)
+ command(3) + gait_phase_sin_cos(2) = 47
```

与旧 48 维的差异：

- **删除 `base_lin_vel`**：策略内部 MoE 历史编码器从 5 帧历史估计机身速度；真机不需要状态估计器。
- 第一个字段从 `base_lin_vel` 变为 `base_ang_vel`，所有索引前移 3。
- 末尾追加 2 维步态相位。
- `last_action` 仍为原始策略输出，不裁剪、不翻符号。
- 关节位置/速度仍经 `kJointSign` 从固件 URDF 约定映射到训练约定。

实现位置：`deploy/src/observer.cpp`、`sim2sim/deploy_mujoco.py::build_observation_frame()`。

### 3.2 5 帧历史

ONNX metadata：

```
history_length = 5
history_order  = frame_major_oldest_first
history_reset  = repeat_initial_frame
input_shape    = [batch, 235]
```

部署侧：

- 进入 `ActiveRL` 时，用当前状态构建首帧（相位 0、`last_action=0`、当前 command），重复 5 次。
- 每个 50 Hz 推理 tick 追加当前帧，左移最旧帧。
- 扁平化时 `frames[0]` 最旧、`frames[4]` 最新。
- 离开 `ActiveRL`、推理失败、反馈冻结时使历史失效；下一次成功 tick 用当前帧重新初始化。

实现位置：`HistoryBuffer`（C++/Python）。

### 3.3 步态相位

训练公式（与 `Go2ReferenceTaskModel.observe` 一致）：

```
steps = round(episode_elapsed_time / 0.02)
p = (steps mod 25) / 25
moving = hypot(vx, vy) + abs(yaw_rate) > 0.1
features = moving ? [sin(2πp), cos(2πp)] : [0, 0]
```

要点：

- 相位时钟是**回合时间**，进入 `ActiveRL` 时归零。
- 站立门控只清零特征，不重置时钟。
- 门控是 `l2(vx,vy) + abs(yaw)`，不是三者平方和。
- 与旧 48 维策略的自适应频率相位完全不同；旧 `advancePhase()` 已删除/替换。

实现位置：`deploy/src/observer.cpp::gaitPhaseFeatures()`、`sim2sim/deploy_mujoco.py::gait_phase_features()`。

### 3.4 命令

ONNX metadata：`command_frame = body_xy_world_yaw`。

- `vx, vy`：机体系平面速度。
- `yaw_rate`：世界/heading 系偏航角速度。
- 手柄桥现有映射直接可用（左摇杆 vx/vy，右摇杆 yaw），无需改硬件。
- 命令平滑（`command_smoothing_alpha`）保留。

### 3.5 动作与目标

ONNX metadata：

```
action_semantics = unscaled_joint_position_offsets
joint_target_formula = default_joint_positions + actions * action_scales
action_scales = [0.25, 0.4, 0.4] × 4
```

部署公式：

```
target_firmware[i] = default_firmware[i]
                     + kJointSign[i] * action[i] * action_scale[i]
```

- `action_scale` 从单一 0.25 改为逐关节：hip 0.25、thigh 0.4、calf 0.4。
- 动作不裁剪；目标 clamp 到关节限位。
- `last_action` 观测原始输出。
- ONNX metadata 的 `default_joint_positions` 是训练系；启动时与 `kJointSign * default_firmware` 逐项校验。

### 3.6 控制频率与 PD

| 项目 | 值 |
|------|-----|
| 策略推理 | **50 Hz**（`inference_hz=50`，`control_dt=0.02`） |
| 目标更新 | 200 Hz |
| PD/CAN | 200 Hz |
| kp / kd | 12.0 / 0.5（训练值；实机可再调） |
| 动作保持 | 每个策略动作保持 4 个 200 Hz target tick |

训练 `decimation=4`、`sim.timestep=0.005`，等效 50 Hz 策略、200 Hz 执行器。sim2sim 已按 0.005 s / 4 倍 decimation 对齐。

### 3.7 状态机：仅 ActiveRL 跑策略

旧实现中 `ActivePC`/`EnteringPosition` 也运行策略。新策略是“从站立开始的回合策略”，不适合当位置控制器，因此改为：

- `ActiveRL`：运行策略；目标 = 默认 + 动作偏移；交接淡入 1 s。
- `ActivePC`：保持默认站姿，不推理。
- `EnteringPosition`：保持默认站姿，不推理。
- `LowGainTest` / `StandingUp`：保持默认姿态（逻辑不变）。
- 进入 `ActiveRL`：设置 `rl_start_s`、重置 5 帧历史、`last_action=0`、相位 0、启动交接淡入。
- 离开 `ActiveRL`：历史失效、动作清零。

操作流程变为：**A → PC 站立；X → ActiveRL 策略**。这与原手柄语义一致，但 PC 不再被策略驱动。

### 3.8 配置

`deploy/configs/opendoge_deploy.conf` 关键值：

```ini
inference_hz=50
target_hz=200
control_hz=200
kp=12.0
kd=0.5
safe_kd=2.0
action_scale_hip=0.25
action_scale_thigh=0.4
action_scale_calf=0.4
gait_period_s=0.5
gait_command_threshold=0.1
```

删除 `lin_vel_source`。启动时会校验：

- ONNX `observation_dim=47`、`history_length=5`、`control_dt=0.02`。
- `action_scales` 与配置逐项一致。
- `default_joint_positions` 与 `kJointSign * default_firmware` 一致。
- `joint_names` 与 `defaultJointMap()` 一致。
- `inference_hz` 必须是 50（`1/50 = 0.02`）。

不一致时拒绝启动，避免误加载旧 48 维模型。

### 3.9 sim2sim

`sim2sim/deploy_mujoco.py`：

- 47/5/235/50 Hz 常量；`build_observation_frame`；`gait_phase_features`；`HistoryBuffer`。
- `OnnxPolicy` 动态取输入名 `obs_history`，校验 235/12 与 metadata。
- `DeployController` 仅 ActiveRL 推理；RL 入口重置；逐关节 scale；PC 保持默认。
- 新增 `--training-root`（默认找同级 `../OpenDoge_train`）、`--terrain`、`--terrain-level`、`--terrain-seed`。
- 删除 `--lin-vel-mode`（盲策略不需要 base_lin_vel）。
- `--joint-convention` 默认改为 `training`。

### 3.10 训练物理对齐

裸 MuJoCo 加载训练 XML 时，mjlab 的运行时覆盖不会自动生效。sim2sim 在 `--joint-convention training` 时显式应用：

| 项目 | 训练值 |
|------|--------|
| `timestep` | 0.005 |
| `iterations` | 10 |
| `ls_iterations` | 20 |
| `ccd_iterations` | 50 |
| `cone` | pyramidal |
| `impratio` | 1.0 |
| 关节 `armature` | 0.005 |
| 足端 `condim` | 3 |
| 足端 `friction` | `[1.0, 0.005, 0.0005]` |
| 足端 `solref` | `[0.01, 1.0]` |
| 足端 `priority` | 1 |
| 非足碰撞 `condim` | 1 |
| 关节力矩上限 | hip/thigh 6 N·m，calf 9 N·m |

### 3.11 训练资产 vs 固件资产的差异

这是本次迁移最重要的发现之一：`OpenDoge_train` 的训练 XML/STL 与 `Lain_Code/docs/URDF` 的固件资产**不是同一套模型**。

| 项目 | 训练 XML | 固件 XML |
|------|----------|----------|
| 关节轴 | FR thigh/calf、RL/RR hip、RR thigh/calf 镜像 | 全部 `+axis` |
| 执行器 | `nu=0`（mjlab 运行时注入） | 12 个 `<motor>` |
| 总质量 | **4.591 kg** | **4.796 kg** |
| FL_hip 位置 | `[0.095, 0.04, 0.002]` | `[0.1425, 0.03999, 0]` |
| FL_thigh 相对位置 | `[0.04105, 0.022, 0]` | `[0, 0.08615, 0]` |
| 小腿质量 | 0.0327 kg | 0.0231 kg |
| STL 网格原点 | 例如 FL_hip bbox `x=[0,0.0826]` | `x=[-0.0411,0.032]` |

因此：

- **不能**把固件 STL 塞进训练 XML，或反过来；几何会错位。
- sim2sim 默认直接引用 `OpenDoge_train` 的训练资产（`--training-root`），**不复制 19 MB 到 `sim2sim/`**。
- 地形场景生成到临时目录：复制 `opendoge.xml`（小文件）+ 为训练 STL 建软链接 + 写地形 XML；不修改训练仓库。
- 如果以后需要 `Lain_Code` 自包含，可把训练资产放到 `docs/URDF/training/`，而不是 `sim2sim/`。
- 固件 XML 仍用于实机部署的关节限位/标定，不用于训练策略的 sim2sim。

### 3.12 地形测试

新增 `sim2sim/terrain.py`：

- `flat`：训练 `scene_opendoge.xml`。
- `rough`：hfield 随机起伏（0.02–0.065 m），前 0.8 m 平台。
- `stairs`：0.03–0.09 m 台阶。
- `boxes`：0.02–0.056 m 方块网格。
- `slope`：5°–17° 斜坡。

新增 `test/terrain_sweep.py`：批量运行并输出平均速度、距离、z、fault。

---

## 4. 验证记录

### 4.1 ONNX 契约

`test/policy_contract_test.py`：**21 passed, 0 failed**。

- 输入 `obs_history [batch,235]`，输出 `actions [batch,12]`。
- `observation_dim=47`、`history_length=5`、`control_dt=0.02`。
- `history_order=frame_major_oldest_first`、`history_reset=repeat_initial_frame`。
- `action_scales=[0.25,0.4,0.4]×4`、默认姿态、关节名、命令系均匹配。
- 固定站立历史输出与 golden action 一致。

### 4.2 C++ 契约

```bash
./install/opendoge_deploy/bin/opendoge_deploy \
  --policy-backend onnx \
  --policy-path policy/opendoge_supermimic_rough_20000.onnx \
  --verify-policy
```

输出与 Python golden 一致：

```
0.0132175 0.00700896 0.215339 0.0740029 -0.00248241 -0.166981
0.128942 -0.015194 0.321856 -0.0187535 -0.0232369 -0.292397
```

### 4.3 C++ dry-run

```bash
./install/opendoge_deploy/bin/opendoge_deploy \
  --policy-backend onnx \
  --policy-path policy/opendoge_supermimic_rough_20000.onnx \
  --start-rl --allow-missing-imu --duration-sec 3
```

结果：

- `state=active_rl`，退出码 0。
- `infer_ticks=50`/秒（50 Hz），`ctrl_ticks=200`/秒。
- 无 fault，动作有限。

### 4.4 sim2sim 平坦地形

```bash
cd sim2sim
python3 deploy_mujoco.py \
  --training-root ../../OpenDoge_train \
  --start-pose default --mode rl \
  --policy ../policy/opendoge_supermimic_rough_20000.onnx \
  --cmd 0.4 0 0 --duration 10 --no-render
```

结果：

| 指标 | 值 |
|------|-----|
| 平均 vx | **0.337 m/s** |
| 行走距离 | **3.37 m** |
| base z | 0.145–0.161 m，末态 0.158 m |
| fault | 0 |
| 动作范围 | -0.50 ~ +0.32 |

### 4.5 sim2sim 地形（10 s，vx=0.4）

| 地形 | 平均 vx | 距离 | 末态 z | fault |
|------|---------|------|--------|-------|
| flat | 0.337 | 3.37 m | 0.158 | 0 |
| rough | 0.333 | 3.33 m | 0.184 | 0 |
| stairs (8 s) | 0.302 | 2.41 m | 0.369 | 0 |
| boxes (8 s) | 0.319 | 2.55 m | 0.172 | 0 |

4 s 快速 sweep（flat/rough/stairs/boxes/slope）全部无 fault、平均 vx 0.28–0.30 m/s。

### 4.6 回归测试

`test/deploy_gap_regression.py`：**49 passed, 0 failed**。

覆盖：47 维 frame、`HistoryBuffer`、相位公式/门控、逐关节 scale、符号映射、配置默认值。

---

## 5. 部署操作变化

### 5.1 模型路径

```bash
POLICY_PATH=policy/opendoge_supermimic_rough_20000.onnx \
  bash scripts/start_robot.sh policy
```

旧模型 `policy/opendoge_flat.onnx`、`policy/opendoge_r5.onnx` 已归档到 `policy/legacy_48dim/`，新运行时不会加载。

### 5.2 启动流程

```bash
bash scripts/start_robot.sh verify
bash scripts/start_robot.sh damping
bash scripts/start_robot.sh low-gain
POLICY_PATH=policy/opendoge_supermimic_rough_20000.onnx bash scripts/start_robot.sh policy
```

操作员：

1. **A**：进入 PC 站立（新逻辑下不跑策略，保持默认站姿）。
2. **X**：进入 ActiveRL，策略开始运行。
3. 左摇杆前后/侧移，右摇杆转向。
4. **B**：回阻尼；急停键保持原映射。

注意：按 A 前机器人必须在默认姿态 ±0.25 rad 内，否则会先进入 `StandingUp` 斜坡。

### 5.3 监控字段

重点看：

- `state`：应为 `active_rl`。
- `infer_ticks`：约 50/秒。
- `obs_imu_age_ms` / `obs_joint_age_ms`：观测年龄。
- `tau_j`：关节侧力矩（calf 6 N·m 安全阈值）。
- `temp`：电机温度。
- `rl_fb`：推理失败降级标志。
- `policy_phase`：0–0.96 循环；`history_ready` / `rl_episode`：活跃回合状态。

---

## 6. 已知风险与待办

| 风险 | 说明 | 建议 |
|------|------|------|
| 训练/固件 XML 不同 | 总质量 4.591 vs 4.796 kg，关节偏置/网格原点不同 | 真机 transfer 需实测；不要混用两套资产 |
| 关节限位 | 固件 RL/RR thigh 下限 −0.85，训练 −1.0468 | 真机先走平坦；rough 可能顶限位 |
| 力矩阈值 | 实机 hip/thigh 4 N·m、calf 6 N·m，训练 6/6/9 | 先保留安全阈值，监控饱和后再评估 |
| kd 调参 | 训练 0.5，旧实机曾用 0.8 | 先 0.5；振荡再 `--kd 0.65/0.8` |
| 50 Hz 策略 | 实机 CAN/IMU 延迟、PD 与训练 MuJoCo 仍有差异 | 先吊装、低增益、PC 站立，再小速度 RL |
| 旧模型兼容 | 48 维模型不可用 | 归档 `policy/legacy_48dim/` |
| 真机粗糙地形 | 本阶段未做 | 先平地，后续单独评估 |

---

## 7. 回滚方法

```bash
# 查看全部改动
git diff

# 仅回滚 C++/配置
git checkout -- deploy/

# 仅回滚 sim2sim
git checkout -- sim2sim/deploy_mujoco.py sim2sim/terrain.py

# 仅回滚测试/文档
git checkout -- test/ docs/

# 整体回滚（会丢失本轮全部修改）
git checkout -- .
```

注意：回滚 `types.hpp` / `observer.cpp` 后必须同时回退 `sim2sim/deploy_mujoco.py`、
`test/deploy_gap_regression.py`、`test/policy_contract_test.py`，否则两条管线观测契约不一致。

---

## 8. 复现命令汇总

```bash
# 构建
export ONNXRUNTIME_ROOT=$(realpath build/deps/onnxruntime)
colcon build --symlink-install --packages-select opendoge_deploy
source install/setup.bash

# ONNX 契约
python3 test/policy_contract_test.py

# 回归
python3 test/deploy_gap_regression.py

# C++ 单次推理
./install/opendoge_deploy/bin/opendoge_deploy \
  --policy-backend onnx \
  --policy-path policy/opendoge_supermimic_rough_20000.onnx \
  --verify-policy

# C++ dry-run
./install/opendoge_deploy/bin/opendoge_deploy \
  --policy-backend onnx \
  --policy-path policy/opendoge_supermimic_rough_20000.onnx \
  --start-rl --allow-missing-imu --duration-sec 3

# sim2sim 平坦
cd sim2sim
python3 deploy_mujoco.py \
  --training-root ../../OpenDoge_train \
  --start-pose default --mode rl \
  --policy ../policy/opendoge_supermimic_rough_20000.onnx \
  --cmd 0.4 0 0 --duration 10 --no-render

# sim2sim 地形
python3 deploy_mujoco.py \
  --training-root ../../OpenDoge_train \
  --terrain rough --terrain-level 1 \
  --start-pose default --mode rl \
  --policy ../policy/opendoge_supermimic_rough_20000.onnx \
  --cmd 0.4 0 0 --duration 10 --no-render

# 地形批量
cd ..
python3 test/terrain_sweep.py \
  --policy policy/opendoge_supermimic_rough_20000.onnx \
  --terrains flat,rough,stairs,boxes,slope --duration 10
```

---

## 9. 越障/掉线优化记录 (2026-11-18)

### 9.1 日志证据

`logs/monitor_status/` 的 1 Hz 快照（2026-11-18）显示：

- `active_rl` 期间最大关节侧力矩约 **3.8 N·m**，最大跟踪误差 **0.46 rad**（安全阈值 0.5），最大 |dq| 约 **6.6 rad/s**；
- `standing_up` 期间 RL_hip q 达 **0.58**（配置上限 0.5218）、RR_thigh q 达 **1.50**（上限 1.3958），说明实际位置会超过训练/配置限位；
- 多次 `damping_fault` 没有伴随 `FAULT/TAU/STALE/NOFDBK` 标志，旧 CSV 又不记录 `fault_reason` 和 IMU，所以无法判断是倒地、跟踪误差、急停还是 CAN 反馈超时。

`logs/highrate/` 的 100 Hz 旧日志（2026-11-07，旧策略，但 CAN 链路同源）给出了更明确的证据：

- 一次 `RR_calf_joint: feedback timeout` 触发 `damping_fault`；
- 同一时刻 **12 个关节**的 `status_age_ms` 和 `target_age_ms` 都达到 **121-131 ms**，说明不是单个电机代码路径，而是**整条部署/CAN 链路约 120 ms 的全局停顿**；
- 同一次日志还有 `TGT_STALE/GAP/STALE_STATUS`、`TRACK`、`OUT`，最大关节侧力矩约 **4.0 N·m**，最大 |dq| 约 **7.4 rad/s**；
- 现有 `state_timeout_s=0.1` 会在这种 120 ms 停顿中先由 RR_calf 触发超时，导致整机进入 DampingFault（操作员感知为“电机掉线”）。

结论：**“掉线”更像 100-130 ms 的全局 CAN/调度/电源瞬态，而不是单个电机协议错误**。同时，`OUT` 表明实际关节位置会超过训练限位，需要软限位/目标余量。

### 9.2 已加入的诊断与安全优化

- `hardware/motor/monitor_status.py` 的 CSV 新增：
  `fault_reason`、`imu_wz/imu_gz/imu_age_ms`、`can_rx/can_tx/can_err/can_tx_queue_full`、
  `max_late_us/missed_ctrl/can_buses`、`policy_phase/history_ready/rl_episode`。
  以后即使只开 monitor，也能看到 DampingFault 的根因和 CAN 状态。
- 新增 `test/analyze_highrate.py`：把 100-200 Hz highrate CSV 汇总为状态转换、
  fault reason、flag 计数、每关节最大力矩/速度/跟踪误差、最大反馈/目标年龄和 worst events。
- `state_timeout_s`：0.1 → **0.15 s**，吸收 122-131 ms 的 burst；真正持续掉线仍会 fault。
- 新增 `rl_tracking_error_threshold=0.65`、`rl_tracking_error_timeout_s=0.5`：
  仅放宽 ActiveRL（台阶接触丰富）的跟踪误差；PC/起立仍用原阈值。
- 新增 `rl_joint_limit_margin=0.03 rad`：ActiveRL 目标在配置限位内保留 3 cm 量级余量，
  降低冲击顶到机械硬限位的概率。
- 这些值都在 `deploy/configs/opendoge_deploy.conf` 中，可随时改回
  `0.1 / 0.5 / 0.3 / 0.0` 做对照。

### 9.3 建议的下一步 A/B 测试

按“先证据、后改参”的顺序：

1. **每次台阶测试都开 highrate**（monitor 只有 1 Hz，会漏掉 10-100 ms 瞬态）：
   ```bash
   sudo python3 hardware/motor/record_highrate.py --rate-hz 200 --duration 120
   python3 test/analyze_highrate.py logs/highrate/highrate_*.csv
   ```
2. **CAN/系统健康**：
   ```bash
   ip -details -statistics link show can0
   ip -details -statistics link show can1
   ip -details -statistics link show can2
   ip -details -statistics link show can3
   dmesg -T | tail -100
   ```
   重点看 `bus-off`、`error-passive`、`restarts`、`tx_queue_full`、USB 重新枚举。
3. **降低总线/调度压力**（逐项 A/B，不要一次全改）：
   - `control_hz=100`（策略仍 50 Hz，200→100 只减少冗余发送）；
   - `fault_poll_hz=2`（10→2，减少 12 帧/100 ms 的额外读）；
   - `feedback_retry_hz=25`（50→25）；
   - `REALTIME_ARGS` 里把 `--cpu 0` 换到 CPU1/2/3，避开 USB IRQ 所在核；
   - 确认 `can_tx_queue_full`、`max_late_us`、`missed_ctrl` 是否下降。
4. **台阶速度限制**：手柄桥先降到
   `--max-vx 0.2 --max-vy 0.1 --max-yaw-rate 0.3`，确认能上再逐步加；
   实机台阶上限约 level3，建议从 level1/2 开始。
5. **力矩阈值分级**：当前 hip/thigh 4 N·m、calf 6 N·m 是安全报警阈值。
   若 highrate 显示 hip/thigh 长时间接近 4 N·m 且温度 < 55°C，可试
   4.5/4.5/7；不要直接跳到训练上限 6/6/9。
6. **硬件排查**：RR_calf/can3 的连接器、线束应变、120Ω 终端、共地、
   电机动力线与 CAN 分开、USB Hub 外部供电、电池电压跌落/加 bulk 电容。
   先交换 RR_calf 与相邻电机/线束，确认“掉线”跟随电机、线束还是端口。

### 9.4 参数辨识建议

目标不是“把策略重训”，而是把**执行器真实动力学**测出来，减少 sim2real gap：

1. **实验设计**：机器人吊起/单腿悬空，一次只动一个关节，其他关节阻尼；
   输入位置阶跃（±0.1、±0.2 rad）和 0.1-10 Hz 小幅度 chirp；
   用 `record_highrate.py` 以 200 Hz 记录 q/dq/tau/target。
2. **模型**：在关节坐标拟合
   `tau = J qdd + b dq + Fc sign(dq) + Fv dq + k(q-q0) + delay`
   或简化的 `J qdd + b dq + Fc sign(dq) + k(q-target)`；
   输出 `J(armature)`、`b(damping)`、`Fc/Fv(friction)`、`delay`，以及力矩常数/饱和。
3. **写回**：把 `armature/damping/frictionloss` 写回 sim2sim 的训练 XML 覆盖；
   把 delay 写进控制/观测链路（当前已有 `--imu-delay-ms`/`--joint-delay-ms` 注入）；
   必要时在 C++ PD 前加摩擦前馈/延迟补偿。
4. **训练侧**：在 `OpenDoge_train` 的 domain randomization 中加入关节摩擦、
   armature、执行器延迟、力矩饱和、关节限位惩罚和更低台阶/更低速度课程；
   这比部署侧“打补丁”更能根治。项目当前已有 `delay_max_lag=4`、
   `delay_hold_prob=0.5`，但还没有关节摩擦/armature 随机化。

参考：
- 关节摩擦常用 Coulomb + Viscous + Stribeck 组合，四足 SysID 中摩擦参数对
  sim2real 跟踪误差和跌倒率影响显著：
  <https://github.com/ImChong/Robotics_Notebooks/blob/main/wiki/concepts/joint-friction-models.md>
- 关节执行器参数辨识（armature、damping、Coulomb/viscous、delay）：
  <https://github.com/ImChong/Robotics_Notebooks/blob/main/wiki/methods/joint-actuator-parameter-identification.md>
- ANYmal/PACE 的位置 chirp + 多负载执行器辨识（`I_a,d,tau_f`）：
  <https://arxiv.org/abs/2509.06342>
- 静摩擦对 sim2real 的影响：
  <https://arxiv.org/abs/2503.01255>
- 执行器网络（用神经网络拟合摩擦/延迟/非线性）：
  <https://github.com/ImChong/Robotics_Notebooks/blob/main/wiki/methods/actuator-network.md>

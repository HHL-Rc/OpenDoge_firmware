# OpenDoge MuJoCo 仿真测试

本目录只放**仿真脚本与地形生成**，不放机器人 XML/STL。机器人描述仍归属训练仓库
`OpenDoge_train`（训练资产）和 `docs/URDF/`（固件/实机资产）。

## 当前策略契约

- ONNX 输入：`obs_history` `[batch, 235]`（5 帧 × 47 维）
- ONNX 输出：`actions` `[batch, 12]`
- 策略 50 Hz，PD 200 Hz
- 无 `base_lin_vel`（盲本体策略）
- 动作缩放 hip 0.25 / thigh 0.4 / calf 0.4

## 平坦地形（推荐入口）

```bash
cd sim2sim
python3 deploy_mujoco.py \
  --training-root ../../OpenDoge_train \
  --start-pose default --mode rl \
  --policy ../policy/opendoge_supermimic_rough_20000.onnx \
  --cmd 0.4 0 0 --duration 10 --no-render
```

- `--training-root` 默认自动查找同级 `../OpenDoge_train`。
- `--joint-convention` 默认 `training`，使用训练 XML（镜像轴、无执行器）。
- 不指定 `--model` 时使用训练 `scene_opendoge.xml`。
- 观测/目标/相位与 `deploy/src/observer.cpp`、`controller.cpp` 同步。

## 手柄

```bash
python3 deploy_mujoco.py \
  --training-root ../../OpenDoge_train \
  --start-pose default \
  --joystick \
  --policy ../policy/opendoge_supermimic_rough_20000.onnx
```

手柄默认适配 BEITONG BTP-KP70C（`20bc:5174`，原生 js0）：

- A/B/X/Y = btn0/1/3/4；
- 左摇杆 axis1/0，右摇杆 axis2；
- X 直接进入 ActiveRL（不需要先按 A；A 仍用于清 fault/PC）；
- M6=btn13（Back/low_gain）、M5=btn14（Start/使能）默认不指定；
  需要时加 `--joy-btn-back 13 --joy-btn-start 14`。

A：PC 站立（不跑策略）→ X：ActiveRL → 左摇杆 vx/vy，右摇杆 yaw → B 回阻尼。

## 地形

```bash
# rough
python3 deploy_mujoco.py \
  --training-root ../../OpenDoge_train \
  --terrain rough --terrain-level 1 \
  --start-pose default --mode rl \
  --policy ../policy/opendoge_supermimic_rough_20000.onnx \
  --cmd 0.4 0 0 --duration 10 --no-render

# stairs / boxes / slope
python3 deploy_mujoco.py ... --terrain stairs --terrain-level 1
python3 deploy_mujoco.py ... --terrain boxes  --terrain-level 1
python3 deploy_mujoco.py ... --terrain slope  --terrain-level 1
```

地形场景生成在临时目录：复制训练 `opendoge.xml` + 为训练 STL 建软链接 +
写地形 XML。**不复制 19 MB 资产到 sim2sim**，也不修改训练仓库。

摔倒是 `projected_gravity` 超过 `--fall-angle-deg`（默认 70°）判定；
`--auto-reset` 会在摔倒后重置到默认姿态并重新开始 ActiveRL 回合。

批量测试：

```bash
python3 test/terrain_sweep.py \
  --policy policy/opendoge_supermimic_rough_20000.onnx \
  --terrains flat,rough,stairs,boxes,slope --duration 10
```

## 键盘

| 按键 | 功能 |
|---|---|
| A / Space | PC 站立（不跑策略） |
| X | 进入 ActiveRL |
| Y | 回 ActivePC |
| B | 回阻尼 |
| ↑ / ↓ / ← / → | 速度命令 |
| Backspace | 急停 |
| Esc / Q | 退出 |

## 物理对齐

`--joint-convention training` 时使用 mjlab 运行时参数：`timestep=0.005`、
`iterations=10`、`ls_iterations=20`、`armature=0.005`、足端 `condim=3` /
`friction=[1.0,0.005,0.0005]` / `solref=[0.01,1.0]`、力矩上限 6/6/9 N·m。

## 已知边界

- 训练 XML/STL 与 `docs/URDF/` 固件资产**不是同一套模型**（总质量、关节偏置、
  网格原点不同）；不要混用。
- 旧 48 维模型不兼容当前脚本；归档在 `policy/legacy_48dim/`。
- 真机部署见 `docs/opendoge_supermimic_migration.md`。

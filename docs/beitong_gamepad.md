# BEITONG BTP-KP70C 手柄接入记录

> 日期: 2026-11-22
> 设备: BEITONG BTP-KP70C BFM DONGLE
> USB ID: `20bc:5174`
> 节点: `/dev/input/js0`、`/dev/input/event22`（同一接收器还创建 `event23`/`mouse3`，忽略）

---

## 1. 设备识别

`/proc/bus/input/devices`：

```
I: Bus=0003 Vendor=20bc Product=5174
N: Name="BEITONG BTP-KP70C BFM DONGLE"
H: Handlers=event22 js0
```

`/dev/input/by-id`：

```
usb-BEITONG_BTP-KP70C_BFM_DONGLE-joystick -> ../js0
```

结论：

- 内核 `hid-generic` + `joydev` 已经创建原生 `js0`；
- 旧 `LinuxJoystick`（读 `/dev/input/js*`）可以直接使用；
- **不需要 evdev backend**；
- `jstest` 可用；`evtest` 选 `event22`（不是 `event0` 的 Lid Switch）。

## 2. jstest 映射

来自 `jstest /dev/input/js0`：

| 物理输入 | js 事件 | 功能 | 备注 |
|---|---|---|---|
| 左摇杆前后 | axis1 | vx | 前推为负、后推为正 |
| 左摇杆左右 | axis0 | vy | 右为正、左为负 |
| 右摇杆前后 | axis3 | 未使用 | |
| 右摇杆左右 | axis2 | yaw_rate | 右为正、左为负 |
| A | btn0 | 激活/位置控制 | |
| B | btn1 | 退出位置控制/RL → 阻尼 | |
| X | btn3 | 进入 RL | |
| Y | btn4 | 退出 RL | |
| LT | axis4 | 未使用 | 不动 -32767，按下变大 |
| RT | axis5 | 未使用 | 同上 |
| LB | btn6 | 未使用 | |
| RB | btn7 | 可配置为 `--require-rb` | |
| M1–M4 | btn0/1/3/4 | 与 A/B/X/Y 别名 | |
| M5 / M6 | btn14 / btn13 | 可选 Start/Back | 默认不指定 |
| Back | 未指定 | 切换 low_gain_mode | 可选 M6=btn13 |
| Start | 未指定 | 切换使能 | 可选 M5=btn14 |

### 2.1 Back/Start 原功能与是否写死

旧 Xbox 布局里：

- `START`：切换 `active_requested`（使能/失能）。按一次 active=true，再按一次
  active=false（回到阻尼）。A 已经能启动、B 已经能回阻尼，所以它是冗余的；
  误触时反而会让机器人突然阻尼。
- `BACK`：切换 `low_gain_mode`（低增益模式），开启时 active=false、速度清零。
  主要用于 bring-up/低增益站立，正常 RL 行走不需要。

因此 **不建议写死 Back/Start**。当前 `--profile auto` 检测到 BEITONG 时把
`btn_back`/`btn_start` 设为 `None`，所以按 **LB/RB（6/7）不会误触发**
low_gain/使能。A/B/X/Y 和摇杆完全不受影响。

如果你确实想要“实体 M5/M6”：

- M5 = `btn14` → START（使能/失能）
- M6 = `btn13` → BACK（low_gain）

在命令里加：

```bash
python3 hardware/gamepad/xbox_command_bridge.py \
  --profile beitong --device /dev/input/js0 \
  --btn-back 13 --btn-start 14
```

或在 `daemons/opendoge-joystick.service` 的 `ExecStart` 里加：

```
--profile auto --btn-back 13 --btn-start 14
```

> 如果你只关心正常行走，保持不指定即可；A 启动、X 进 RL、B 回阻尼已经完整。

## 3. 代码改动

### 3.1 `hardware/gamepad/opendoge_joystick.py`

- 恢复 `BUTTON_X=2`、`BUTTON_Y=3`、`BUTTON_LB=4`、`BUTTON_RB=5` 的 Xbox 默认值；
- 不再把 BEITONG 的编号写进全局常量，避免破坏旧 Xbox/Zikway 默认；
- BEITONG 的 A/B/X/Y/LB/RB 由 `xbox_command_bridge.py` 的 `beitong` profile 覆盖；
- X 现在可以直接进入 ActiveRL，不要求先按 A（A 仍用于清 fault/PC）。

### 3.2 `hardware/gamepad/xbox_command_bridge.py`

- `--profile` 新增 `beitong`；
- `auto` 模式通过设备名 `BEITONG` / `BTP-KP70C` 识别；
- profile 设置：A=0、B=1、X=3、Y=4、RB=7；
- `btn_back`/`btn_start` 默认置空，避免与 LB/RB 冲突；
- 启动时打印提示，未指定时显示 `unassigned`；
- Zikway `3537:1041` 分支保持原样。

### 3.3 `scripts/start_gamepad.sh`

- 新增 `20bc:5174` 分支；
- 检测到 BEITONG 后等待 `/dev/input/js0` 出现，然后 `exec sleep infinity`；
- 不再 30 s 后退出导致 `opendoge-xboxdrv` 反复重启；
- 旧 `413d:2104`、`3537:1041`、`3537:2106` 分支不变。

### 3.4 `sim2sim/deploy_mujoco.py`

- `JoystickHandler` 默认改为 BEITONG：A=0、B=1、X=3、Y=4，轴 1/0/2；
- 删除了旧的 Zikway `BACK=7/START=6` 默认；
- `--joy-btn-back` / `--joy-btn-start` 默认 `None`，不指定时 M5/M6 无功能；
- 需要实体键时传 `--joy-btn-back 13 --joy-btn-start 14`；
- X 直接进 ActiveRL 的行为与实机 bridge 一致。

## 4. 验证步骤

```bash
# 0. 不上硬件的映射回归
python3 test/gamepad_profile_test.py

# 1. 节点
ls -l /dev/input/js0
cat /proc/bus/input/devices | grep -A8 -B1 -i beitong

# 2. 按键探索
jstest /dev/input/js0
# 或
sudo evtest /dev/input/event22

# 3. 手动桥接（先不带 Back/Start）
python3 hardware/gamepad/xbox_command_bridge.py \
  --profile beitong --device /dev/input/js0

# 4. 看命令文件
watch -n 0.2 cat /tmp/opendoge_command.state

# 5. 服务
systemctl status opendoge-xboxdrv
systemctl --user status opendoge-joystick
journalctl --user -u opendoge-joystick -f
```

期望：

- 推左摇杆前后，`vx` 变化；
- 推左摇杆左右，`vy` 变化；
- 推右摇杆左右，`yaw_rate` 变化；
- 按 X/Y/A/B，`rl_inference`/`active`/`position_control` 按预期变化；
- 桥接启动时如果没给 Back/Start，会打印 `[BEITONG] Back/Start are unassigned`。

## 5. sim2sim 注意事项

`sim2sim/deploy_mujoco.py --joystick` 复用的是同一个 `XboxCommandMapper`，但
`JoystickHandler` 的默认按钮布局是 Zikway（START=6、BACK=7）。BEITONG 的
LB/RB 正好是 6/7，所以在 sim 里也应显式指定：

```bash
python3 sim2sim/deploy_mujoco.py \
  --joystick /dev/input/js0 \
  --joy-btn-a 0 --joy-btn-b 1 --joy-btn-x 3 --joy-btn-y 4 \
  ...
```

sim2sim 默认已经用 BEITONG 的 A/B/X/Y 和 axis1/0/2；Back/Start 默认不指定。
需要 M5/M6 时加 `--joy-btn-back 13 --joy-btn-start 14`。

## 6. 已知待办

- [ ] 可选：如果你想要实体 M5/M6，把 `--btn-back 13 --btn-start 14` 写入 service；
- [ ] 可选：M5/M6 是否另作急停键（会与 Back/Start 冲突，二选一）；
- [ ] 实机确认 LB/RB 不再误触发 low_gain/使能；
- [ ] 实机确认 X 可以直接进入 ActiveRL（无需先按 A）。

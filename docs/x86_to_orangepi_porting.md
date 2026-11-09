# x86 → Orange Pi 5 移植步骤手册

> 适用：把本仓库从 x86 Ubuntu 开发机迁移到 **Orange Pi 5（RK3588，aarch64）
> + Ubuntu 22.04 + `orangepi` 用户**。
>
> 外设：赤乌（CHIWU）4×CAN（4×`gs_usb`，`1d50:606f`）、DM-IMU-L1（USB 串口）、
> Zikway `3537:1041` 手柄。
>
> 本手册只做移植/运行环境；**不改** 48 维观测、`action_scale`、默认姿态、
> 控制频率、EL05 协议和策略 ONNX。

## 0. 一次性清单

```text
[x] 仓库已复制到 Orange Pi 的 ~/Opendoge_code（或实际路径）
[x] aarch64 编译通过，ldd 能找 libonnxruntime
[x] opendoge_deploy 有 cap_sys_nice + cap_net_raw
[x] opendoge-can / opendoge-xboxdrv 开机自启
[x] opendoge-imu / opendoge-joystick 用户服务自启 + linger
[x] can0-3 UP @ 1 Mbps, verify_can_mapping 12/12
[x] /dev/ttyACM* 0666, /dev/input/js0 可读
[x] start_robot.sh verify 通过；damping 先悬挂验证
[ ] low-gain / policy（机器人支撑好后再做）
```

## 1. 从 x86 电脑传输代码到 Orange Pi

推荐 **rsync over SSH**（增量、可排除 x86 构建产物）。

### 1.1 在 Orange Pi 上准备

```bash
hostname -I                         # 记下 IP，例如 192.168.1.50
sudo apt update
sudo apt install -y openssh-server rsync
sudo systemctl enable --now ssh
```

### 1.2 在 x86 上同步

```bash
# 源：x86 上的工作副本（不是旧的 x86 build/install）
rsync -av --delete --info=progress2 \
  --exclude 'build/' --exclude 'install/' --exclude 'log/' \
  --exclude 'logs/' --exclude '__pycache__/' --exclude '.venv/' \
  /home/hhl/orangepi_backup/Opendoge_code/ \
  orangepi@<orange-pi-ip>:~/Opendoge_code/
```

- 首次会传全部源码；以后只传差异；
- 不想删除目标端额外文件就去掉 `--delete`；
- 传完在 Orange Pi 上确认：

```bash
cd ~/Opendoge_code
git status --short
git log --oneline -3
```

### 1.3 备选传输方式

```bash
# scp（简单，无增量/排除）
scp -r /home/hhl/orangepi_backup/Opendoge_code orangepi@<ip>:~/

# git bundle（带完整历史）
cd /home/hhl/orangepi_backup/Opendoge_code
git bundle create /tmp/opendoge_repo.bundle --all
scp /tmp/opendoge_repo.bundle orangepi@<ip>:~/
# Orange Pi:
git clone ~/opendoge_repo.bundle ~/Opendoge_code

# U 盘（无网络）
rsync -av /media/<u盘>/Opendoge_code/ ~/Opendoge_code/
```

> 不要从 x86 直接拷贝 `build/`、`install/` 或 `build/deps/onnxruntime-*`：
> 它们是 x86_64 ELF / .so，在 ARM 上不能运行。

## 2. 系统依赖（Ubuntu 22.04）

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake wget curl \
  iproute2 usbutils can-utils \
  libcap2-bin rsync \
  python3 python3-dev
# colcon（二选一）
sudo apt install -y python3-colcon-common-extensions || \
  pip3 install --user colcon-common-extensions
```

可选（只有 413d:2104 Xbox dongle 才需要）：

```bash
sudo apt install -y xboxdrv
```

Zikway `3537:1041` 走内核 `hid-generic`/`joydev`，不需要 xboxdrv。

## 3. aarch64 编译

不要复用 x86 的 `build/`/`install/`。使用独立的 `build_aarch64`/`install_aarch64`。

```bash
cd ~/Opendoge_code

# ONNX Runtime：setup_onnx.sh 会根据 uname -m 选择 aarch64 包
DEPS_DIR="$PWD/build_aarch64/deps" ./scripts/setup_onnx.sh

export ONNXRUNTIME_ROOT="$(realpath build_aarch64/deps/onnxruntime)"
# 注意：--log-base 是 colcon 全局参数，必须放在 build 子命令之前
colcon --log-base log_aarch64 build --symlink-install --packages-select opendoge_deploy \
  --build-base build_aarch64 \
  --install-base install_aarch64

source install_aarch64/setup.bash
```

验证：

```bash
file install_aarch64/opendoge_deploy/bin/opendoge_deploy   # ELF 64-bit aarch64
ldd  install_aarch64/opendoge_deploy/bin/opendoge_deploy | grep onnx
```

如果 Orange Pi 不能联网：

1. 在能联网的机器下载 `onnxruntime-linux-aarch64-1.18.1.tgz`；
2. 拷到 Orange Pi 的 `build_aarch64/deps/`；
3. 再运行 `DEPS_DIR=... ./scripts/setup_onnx.sh`（会用缓存 tgz）。

如果 1.18.1 没有 aarch64 包，用 `ONNX_VERSION=<有 aarch64 的版本>`；
先确认 releases 页面对应 tarball 存在。

## 4. deploy capabilities（必须）

`opendoge_deploy` 以普通用户运行，需要：

- `cap_sys_nice`：`--realtime`/SCHED_FIFO/绑核；
- `cap_net_raw`：打开 `AF_CAN` raw socket。

使用仓库脚本：

```bash
bash scripts/setup_deploy_caps.sh
# 或显式：
bash scripts/setup_deploy_caps.sh install_aarch64/opendoge_deploy/bin/opendoge_deploy
```

输出应类似：

```text
capabilities set on: .../opendoge_deploy
.../opendoge_deploy cap_net_raw,cap_sys_nice=ep
```

> **每次 `colcon build` 重新链接后 caps 会丢失**；重建后重跑一次。

## 5. 自动启动 CAN / IMU / 手柄桥接

安装一次即可；`install_services.sh` 会 enable 四个服务并开启 linger：

```bash
cd ~/Opendoge_code
sudo bash scripts/install_services.sh
```

它做这些事：

- 安装/启动 `opendoge-can.service`（system, oneshot）：
  `prepare_can_interfaces.sh` → 等 can0-3 → `setup_can_interfaces.sh`（1 Mbps/fd off/UP）；
- 安装/启动 `opendoge-xboxdrv.service`（system）：
  Zikway 原生 js0 时保持 active；413d:2104 时启动 xboxdrv；
- 安装 `opendoge-imu.service` / `opendoge-joystick.service`（user）：
  unit 里的 `/home/orangepi/Opendoge/OpenDoge_firmware` 会替换为实际仓库路径；
- 安装 IMU udev 规则（`/dev/ttyACM*` 0666）；
- 安装手柄 udev 规则（`/dev/input/js0` 0666）；
- `systemctl --user enable` + `loginctl enable-linger orangepi`。

验证：

```bash
systemctl is-active opendoge-can opendoge-xboxdrv
systemctl --user is-active opendoge-imu opendoge-joystick
loginctl show-user orangepi | grep Linger
ls -l /dev/ttyACM* /dev/input/js0
```

如果 js0 仍不可读：

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger --action=change --subsystem-match=input
# 或改 GROUP="input" 并把用户加入 input：
sudo usermod -aG input orangepi   # 重新登录
```

如果 IMU `Permission denied`：

```bash
sudo usermod -aG dialout orangepi   # 重新登录；udev 0666 规则通常已够
```

> **deploy 不随开机自动启动**：上电后电机不应自动动。仍由
> `scripts/start_robot.sh` 手动启动。

## 6. CAN 接口映射（安全项，必须先验证）

CHIWU 的 4 个 `gs_usb` 在 Orange Pi 的 USB 拓扑/枚举顺序可能与 x86 不同。
`prepare_can_interfaces.sh` 按 USB 路径排序，必须实测：

```bash
# 1) 只读检查接口/驱动/bitrate
python3 scripts/verify_can_mapping.py --interfaces-only

# 2) 电机上电但不使能，校验 12 个 ID
sudo python3 scripts/verify_can_mapping.py
```

必须看到：

```text
can0 -> {1,2,3}   (FL)
can1 -> {4,5,6}   (FR)
can2 -> {7,8,9}   (RL)
can3 -> {10,11,12}(RR)
```

如果顺序不对：

- 先交换 CHIWU 板 CAN1-4 的接线，或调整接口名；
- 确认正确后冻结当前 USB 路径，避免下次启动再变：

```bash
sudo bash scripts/setup_can_names.sh --freeze-current
```

**映射错误时不要使能电机。**

## 7. CPU / 实时性能

RK3588：

```text
CPU0-3: Cortex-A55 (little)
CPU4-7: Cortex-A76 (big)
```

现有 unit：IMU/joystick `CPUAffinity=6 7`（A76）；deploy 默认
`REALTIME_ARGS="--realtime --cpu 0"`（A55）。

先实测：

```bash
bash scripts/start_robot.sh damping   # 机器人悬挂/支撑
cat /tmp/opendoge_status.json | python3 -m json.tool | \
  grep -E 'ctrl_ticks|infer_ticks|max_late_us|missed_ctrl|obs_imu_age_ms|obs_joint_age_ms'
```

- `ctrl_ticks≈200`、`infer_ticks≈100`、`missed_ctrl=0`、`max_late_us` 很小 → 正常；
- 如果明显变差，试 A76：

```bash
REALTIME_ARGS="--realtime --cpu 4" bash scripts/start_robot.sh damping
```

再决定是否改 `start_robot.sh` 默认。**不要先改 `inference_hz`/`target_hz`。**

## 8. 上电/验证顺序

```bash
# 1) 服务
bash scripts/start_robot.sh verify

# 2) 悬挂/支撑，先阻尼
bash scripts/start_robot.sh damping

# 3) 低位姿站立
bash scripts/start_robot.sh low-gain

# 4) 策略
bash scripts/start_robot.sh policy
```

首次运行前：

- 确认 `can0-3` 映射 12/12；
- 确认机器人悬挂/支撑，急停/断电手边可用；
- 观察 `/tmp/opendoge_status.json` 的 `state`、`fault_reason`、`obs_*_age_ms`、
  每关节 `rx_ratio`、`can_buses`；
- 手柄 B 回阻尼，btn11 急停锁存，按 A 清除。

## 9. 排查

| 现象 | 检查 |
|---|---|
| `x86` 的 deploy 拷贝到 ARM 跑不了 | 不要在 ARM 上用 x86 ELF/`.so`；在 Orange Pi 上 `setup_onnx.sh` + `colcon` 重编 |
| `ldd` 找不到 onnxruntime | `ONNXRUNTIME_ROOT` 指到 `build_aarch64/deps/onnxruntime`；确认 `libonnxruntime.so` 是 aarch64 |
| `setcap` 后仍 `Operation not permitted` | 重编后 caps 丢失；重跑 `scripts/setup_deploy_caps.sh` |
| CAN raw `Permission denied` | deploy 缺 `cap_net_raw`；诊断 Python 用 `sudo` |
| `can0-3` 不存在 | `lsusb`/`dmesg`/`modinfo gs_usb`；`systemctl status opendoge-can`；USB 供电 |
| can0-3 顺序不对 | `verify_can_mapping.py`；交换接线或 `setup_can_names.sh --freeze-current` |
| `gs_usb ... err=-110` | CHIWU 被电机断电/复位；`sudo scripts/reset_can_usb.sh`；板子 24V 独立 |
| IMU `Permission denied` | `ls -l /dev/ttyACM*` 应为 0666；重装 udev 或加 dialout |
| 手柄 `js0 not found` | `lsusb | grep -E '3537:1041|3537:2106'`；2106 模式需拔插回 1041；`jstest` |
| `js0` 存在但 bridge 打不开 | `ls -l /dev/input/js0`；安装 0666 规则或加 `input` 组 |
| IMU age 100 ms 尖峰 | USB 口/线/供电、bridge CPU 调度；看 `journalctl --user -u opendoge-imu` |
| `max_late_us` 大/`missed_ctrl` 非 0 | 试 `REALTIME_ARGS="--realtime --cpu 4"`；确认 caps；不要先降控制频率 |
| 自转/侧移偶发震荡 | 先看 `obs_imu_age_ms/obs_joint_age_ms` 和 IMU p99；再查 kd/延迟，按主仓库记录 |

## 10. 回滚/注意

- 本手册的脚本改动只在移植工作副本里；`RC/Lain_Code` 不受影响。
- `js0 0666` 是放宽权限；介意改 `input` 组。
- `setcap` 回滚：`sudo setcap -r <binary>`。
- CAN udev 回滚：删除 `/etc/udev/rules.d/70-opendoge-can.rules`，重新插拔/重启。
- CPU 绑核先用 `REALTIME_ARGS` 临时覆盖，不改默认。
- 不要复制 x86 的 `build/`/`install/`/ONNX Runtime。
- Zikway 原生不需要 xboxdrv；413d:2104 才安装。

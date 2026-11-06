# EL05 Motor Control Layer

OpenDoge 电机控制层 — EL05/RobStride 准直驱电机的完整 CAN 协议栈和调测工具。

协议: RobStride 私有 29-bit 扩展 CAN 帧 (非 CANopen)
手册: `docs/EL05使用说明书2600428.pdf`

## 文件清单

| 文件 | 语言 | 用途 |
|------|------|------|
| `el05_socketcan.cpp` | C++ | 生产部署 — EL05 CAN 协议实现 |
| `el05_socketcan.hpp` | C++ | 协议头文件 (comm 类型、参数索引、float↔uint 映射) |
| `el05_motor_menu.py` | Python | 交互式电机调测菜单 (使能、停止、置零、小伺服、参数读写) |
| `el05_calibrate.py` | Python | 电机标定 — 计算 JointCalibration 偏移量 (只读位置，不写零位) |
| `protocol_selftest.py` | Python | 协议自检 — 无硬件离线验证帧打包/解包一致性 |
| `scan_motors_readonly.py` | Python | 只读扫描 — 发现全部电机并读取参数 (不使能、不写参数) |
| `watch_angles.py` | Python | 实时角度观察 — 只读 mechPos/mechVel/faultSta，支持重试/补扫/CSV |
| `zero_all_motors.py` | Python | 12 电机站立位一键置零 + 写 Flash (运控→SET_ZERO→SAVE) |
| `el05_set_can_timeout.py` | Python | 读写 0x7028 canTimeout (失联自动释放) |
| `discover_direction.py` | Python | 只读方向检测 (手推 ΔmechPos) |
| `record_standing.py` | Python | 只读站立位角度记录 (旧参考角，勿直接用于配置) |

> 旧 DINGLAB/SOULDE USB2CAN 传输代码已归档到 `~/DingLab usb转can代码`，
> 当前使用赤乌（CHIWU）原生 SocketCAN `can0-can3`。

## 协议一致性

CAN ID 构造、float↔uint 映射范围 (P_MIN=-12.57, P_MAX=12.57, V_MIN=-50, V_MAX=50,
T_MIN=-6, T_MAX=6, KP_MAX=500, KD_MAX=5)、comm_type 常量 (0x00-0x19)、
参数索引 (0x7005, 0x7019, 0x701B, 0x3022) 在以上三处实现 (C++ + menu + selftest)
必须保持一致。

## 快速使用

### 只读扫描 (安全，推荐首次使用)

```bash
python3 hardware/motor/scan_motors_readonly.py                    # 标准 ID 1-12
python3 hardware/motor/scan_motors_readonly.py --all-ids         # 含出厂 ID 127
python3 hardware/motor/scan_motors_readonly.py --channel can3    # 指定通道
```

### 交互式调测 (⚠️ 可写操作)

```bash
python3 hardware/motor/el05_motor_menu.py --channel can0 --master-id 0xfd
python3 hardware/motor/el05_motor_menu.py --channel can3 --ids 10,11,12,127
```

### 标定 (只读位置)

```bash
python3 hardware/motor/el05_calibrate.py --channel can0           # 逐关节
python3 hardware/motor/el05_calibrate.py --channel can0 --batch   # 批量
```

### 协议自检 (无硬件)

```bash
python3 hardware/motor/protocol_selftest.py
```

## 电机出厂默认值

新电机 **CAN ID 默认为 127 (0x7F)**，不是 OpenDoge 标准 ID 1-12。
需要使用 Windows 上位机 **EDULITE-TOOL** (robstride.com) 逐个改为标准 ID。

## CAN 通道映射

```
can0 = 左前 FL: motor 1/2/3  (hip/thigh/calf)
can1 = 右前 FR: motor 4/5/6
can2 = 左后 RL: motor 7/8/9
can3 = 右后 RR: motor 10/11/12
```

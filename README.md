# PiPER 机械臂视觉抓取

用装在夹爪上的深度相机，让 AgileX **PiPER** 六轴机械臂自己找到桌上的物体并抓住。
纯经典几何 + 闭环伺服，**不用神经网络、不训练模型**，四个任务都在真机上跑通、可一键复跑。

```fish
set V /home/po/vision
set PY /home/po/arm-env/bin/python

$PY $V/armctl.py status      # 固件 / 关节 / 法兰位姿 / 控制模式
$PY $V/armctl.py cap --keep  # 夹瓶盖，夹住不放
$PY $V/armctl.py home        # 回零停放（PiPER 没有抱闸，别断电）
```

---

## 1. 硬件与坐标链

| 部件 | 型号 / 参数 | 备注 |
|---|---|---|
| 机械臂 | AgileX **PiPER**，6 关节 + 平行夹爪 | 固件 S-V1.9-0，关节反馈 200 Hz |
| 通信 | candleLight USB-CAN → `can0` | 1 Mbps |
| 相机 | **Orbbec Dabai DC1**（彩色 + 深度） | 装在夹爪上方，eye-in-hand，画面倒装 → 必须旋转 180° |
| 控制库 | pyAgxArm | 旧 piper_sdk 只兼容 ≤ S-V1.8-8，本机固件 1.9-0 必须用新组 |
| 视觉库 | pyorbbecsdk（源码编译） | PyPI wheel 是 macOS 二进制；必须用 main 分支（SDK 1.10.16） |

一条抓取要串起整条链：

```
像素 (u,v) + 深度 d
   │ (1) 相机内参 K + 畸变
   ▼
相机系 P_cam ──(2) 手眼矩阵 T_flange→cam──▶ 法兰系 P_flange
   │ (3) 正运动学（控制器给）
   ▼
基座系 P_base ──(4) 抓取几何（高度、姿态、开口）──▶ move_p / move_j
```

关节限位（度）：J1 ±150、J2 0~180、J3 -170~0、J4 ±100、J5 ±69.9、J6 ±120；夹爪 0~70 mm。

---

## 2. 快速开始

```fish
set V /home/po/vision
set PY /home/po/arm-env/bin/python     # 动臂：pyAgxArm + opencv + numpy

$PY $V/armctl.py status               # 只读：固件/关节/法兰/控制模式
$PY $V/armctl.py home                 # 连贯回零：仿真预检 → 一段走到零位
$PY $V/armctl.py look                 # 腕部相机拍一张 → out/look_color.png
```

| 命令 | 做什么 | 常用参数 |
|---|---|---|
| `armctl.py box` | 抓**方盒**（棕色牛皮纸盒） | `--release` 放回；`--calib` 现场标伺服雅可比 |
| `armctl.py screw-top` | 抓**3D 打印螺丝**，**自动转角** | `--dry --lift --release` |
| `armctl.py bottle` | **侧夹饮料瓶身**（工具轴前倾 45° 斜插） | `--dry --keep` |
| `armctl.py body` | **正上方夹瓶身**（张开手指压下去再闭合） | `--dry --keep` |
| `armctl.py cap` | **夹瓶盖**（像素伺服 + 沿工具轴下探 + 闭合判宽） | `--dry --keep --place X Y` |

---

## 3. 三个核心设计决定

### 3.1 为什么不需要神经网络

目标物体都有稳定的颜色/形状特征（棕色纸盒、深色螺杆、白色瓶盖），检测只要
「HSV 掩膜 + 连通域 + 几何筛选」或「Hough 圆 + 白度校验」，30 ms 一帧、CPU 跑。
之后靠**图像闭环伺服**把目标推到目标像素，误差从 200+ px 收敛到 3~5 px，
末端定位误差 < 2 mm —— 比回归模型稳，而且失败时能说清为什么失败。

### 3.2 像素伺服，而不是硬算手眼

手眼矩阵标出来了（跨位姿重投影 ~7 px ≈ 7 mm），但**像素级定位不够用**：
相机不在夹爪轴线上（法兰→相机 (16, 3, 82) mm），画面里目标居中 ≠ 两指对准目标。

所以主路径是「**图像域闭环**」：读 `calib/servo_calib.json` 里的像素→基座雅可比 Jinv，
每步算 `step = Jinv @ Rm(theta) @ (目标像素 - 当前像素)`，单步 ≤15 mm，带线搜索
（误差变大就减半并回退，收敛良好就放大）。Rm 是必要的：腕子一转，图像轴跟着转，
不补偿的话 ±90° 后会单向发散（踩过）。

手眼矩阵仍保留，用于**深度图反算物体三维坐标**（瓶盖定位就靠它）。

### 3.3 任何 move_p 之前先 IK 预检，长距离走关节空间

`kin.py` 用官方 URDF 在 PyBullet 里解 IK（模型与真机 FK 差 0.06 mm），
返回可解性与每个关节的余量：

```python
kin.check([300, -60, 340], [180, 30, 168])   # -> {'ok': True, 'err': 0.4, 'margins': {...}}
```

教训来自一次事故：直接 move_p 一个「目标姿态要腕子转 90°、起点又接近竖直」的位姿，
落在腕部奇异区，IK 病态（J2 要扫 138°），真机抽搐、把桌上的东西砸了。现在的规矩：

1. 笛卡尔增量 ≤2 cm / ≤5°（`move_p_limited` 自动拆步 + 重发 + 停滞判据）
2. 长距离、大姿态变化一律 `goto_q`（关节空间分段，每段单关节 ≤32°）
3. 每步之后量误差，超差先重发，连续 3 步不前进就刹车

---

## 4. 判据速查（同一物体不同夹法读数不同，别记死一个数）

| 场景 | 夹住 | 空夹 / 别的东西 |
|---|---|---|
| 正上方夹瓶盖（capgrasp） | 宽 27~40 mm 且 \|力\| > 0.9 N（实测 32.8 / -2.09） | 宽 26.4 / -0.17 空；宽 >46 是瓶身 |
| 45° 斜插夹瓶身（bottleside） | 宽 50~72 mm 且 \|力\| > 1.5 N（实测 57.3 / -2.08） | 宽 26.2 / -0.09 |
| 正上方夹瓶身（bodygrasp） | 宽 50~72 mm 且 \|力\| > 1.5 N（实测 58.8 / -2.09） | 宽 34~38 是瓶盖/颈环 |
| 方盒（autograsp） | 宽 > 60 mm 且 \|力\| > 0.5 N（实测 68.0 / -2.08） | — |

「夹住」的最终确认永远是**回抬 6~8 mm 后宽度与力是否保持**，不是闭合瞬间的读数。

---

## 5. 实测记录

| 任务 | 结果 |
|---|---|
| 方盒 | 伺服 51→3.4 px，闭合 68.0 mm / -2.08 N，抬起 67.9 mm 仍夹持，可精确放回 |
| 螺丝（自动转角） | 扫 J6 −90°~+90° 选各向异性峰值，同一天不同位置分别选中 −15°/+10°/+45° |
| 侧夹瓶身 | 45° 前倾斜插 19 段（每段误差 0.3~2.2 mm），闭合 57.3 mm / -2.08 N，抬 40 mm 保持 |
| 正上方夹瓶身 | z=294 / z=300 两次：58.8~59.1 mm / -2.09 N，抬起后瓶子离桌 |
| 夹瓶盖 | 2 步伺服 98→6 px，z=324 闭合 32.8 mm / -2.09 N，抬 30 mm 保持 |

硬件侧：关节 200 Hz 反馈；彩色 640×480@30 MJPG、深度 640×400@30（对齐到彩色）；
单帧抓取 ~0.3 s（常驻进程 ~0.05 s）；伺服一步 1.5~2.5 s。

---

## 6. 踩过的坑（都写进了代码）

1. **控制器会整帧丢弃 move_p 目标**，状态还报 REACH_TARGET_POS_SUCCESSFULLY。
   残差恰好等于一个步长，重发两次也无效 → 现在每步都量误差、重发、判停滞。
2. **低速陷阱**：speed=6 时 0.9 s 的连发窗口走不完一个 14 mm 的步，回读只有一半。
   接近/下降统一 speed 9~10。
3. **示教模式会吞掉所有运动指令**：手拖过臂之后 ctrl_mode=TEACHING(0x2)，
   enable() 返回 True 但不动。要先发一次 move_j 才切回 CAN_CTRL(0x1)。
4. **深度近端限**：DC1 在相机距物体 <~165 mm 时整帧深度为 0（不是故障）。
   高位观察不可达时才被迫 close-up，此时判断只能靠「闭合宽度 + 回抬保持」。
5. **遥测「发疯」的真相**：掉线后 get_joint_angles() 返回 None，是因为控制器报了**多圈绝对值**
   （J2 = −361.68°），SDK 校验器拒绝 → 发一次正常 move_j 就归一化。
6. **USB 掉线**：这台机器上整条 hub 掉过 3 次（相机 2bc5 与 CAN 1d50 一起消失）。
   先查 `lsusb` + `ip -br link show can0`，再谈脚本。相机线要绑在臂上留余量。
7. **控制器不限位拦截**（get_joint_limits_enabled()==False）——越限不报错，只会变成乱姿态。
8. **PiPER 无抱闸**：断电即塌。disable / 拔线 / 断电前先托住末端。

---

## 7. 目录

```
.
├─ armctl.py              统一入口（status/home/look/box/screw/bottle/body/cap）
├─ autograsp.py           方盒主流程 + 公共层（connect/pose/move_p/goto_q/gripper/capture）
├─ bodygrasp.py           正上方夹瓶身
├─ bottleside.py          45° 斜插侧夹瓶身
├─ capgrasp.py            夹瓶盖（Hough 检测 + 像素伺服 + 轴向下降）
├─ screwgrasp.py          螺丝（自动转角）   screwvision.py  螺丝视觉特征
├─ kin.py                 纯仿真 IK 预检（不连真机）
├─ simpath.py             关节空间直线路径碰撞预检（含夹爪 URDF + 桌面）
├─ capture.py             RGB-D 采集（可常驻：--serve）
├─ detect_box.py          单帧方盒检测
├─ handeye.py             手眼标定（棋盘格 9×6 / 25 mm）
├─ intrinsic_arm.py       用机械臂大范围姿态标内参
├─ calibrate_intrinsic.py 手持标内参
├─ safemove.py            低速小步移动 + 看门狗
├─ teach.py               示教录制/回放（100 Hz 关节日志）
├─ armraw.py              CAN 原始帧解析（SDK 读数异常时对照）
├─ act/                   ACT（LeRobot）采集 → 训练 → 回放脚本
├─ pi05/                  π0.5 客户端（连远端 policy 服务）
├─ tools/                 一次性诊断脚本（定位/验证/仿真检查）
├─ docs/                  流程文档、ACT 方案、π0.5 评估、部署笔记
├─ urdf/                  两个 URDF（mesh 路径是占位符，见 scripts/setup_urdf.sh）
├─ scripts/setup_urdf.sh  拉取 agx_arm_urdf 并生成可用 URDF
├─ calib/                 标定数据（不入库）
└─ out/                   图片与深度（不入库）
```

---

## 8. 安装

两个 Python 环境（版本不同，别混）：

| 环境 | Python | 用途 | 内容 |
|---|---|---|---|
| arm-env | 3.13 | 动臂 | pyAgxArm、python-can、opencv-headless 4.12、numpy 2 |
| vision-env | 3.11 | 只拍照 | pyorbbecsdk（源码编译）、opencv 4.11、numpy 1.26 |

```bash
uv venv --python 3.13 ~/arm-env
uv pip install --python ~/arm-env/bin/python -r requirements-arm.txt

uv venv --python 3.11 ~/vision-env
uv pip install --python ~/vision-env/bin/python -r requirements-vision.txt
# pyorbbecsdk 必须源码编译（PyPI wheel 是 macOS 二进制），见 docs/PIPER_抓取全流程.md

./scripts/setup_urdf.sh          # 仿真用的 URDF + mesh
```

CAN 1 Mbps 自启（systemd 单元在 deploy/）。

---

## 9. 学习路线（两条，都在进行中）

脚本抓取在**已知类别**上够用，但换物体就得改阈值。想让它看什么抓什么，走两条线：

- **act/（先做这条）**：采集真机演示（相机 + 关节同步 100 Hz）→ LeRobot 数据集 → ACT 训练 → 闭环回放。
  本机 8 GB 显存够跑（320×240、batch 8，实测 11.5 step/s、显存 0.23 GB）。
  合成数据上整条链路已跑通，等真实演示数据。
- **pi05/**：π0.5（3.35 B 参数）部署在 GPU 服务器上，本机只做客户端。
  本地 8 GB 显存放不下（权重 12.5 GB），至少 16 GB。服务端实测：加载 374 s、首帧 65 s、
  稳态 0.171 s/次、显存 6.94 GB。详见 [docs/ACT方案.md](docs/ACT方案.md)、[docs/pi05-itbk.md](docs/pi05-itbk.md)。

---

## 10. 文档

- [docs/PIPER_抓取全流程.md](docs/PIPER_抓取全流程.md) —— 从接线到抓取的完整流程（附 PDF）
- [docs/ACT方案.md](docs/ACT方案.md) —— ACT 采集/训练/回放的方案与实测
- [docs/pi05-itbk.md](docs/pi05-itbk.md) —— π0.5 在 GPU 服务器上的部署记录
- [docs/π0.5评估.md](docs/π0.5评估.md) —— 本地 8 GB 跑不动 π0.5 的实测数据
- [docs/GRASP_PLAN.md](docs/GRASP_PLAN.md) —— 最早的抓取方案草稿

## 11. 安全

1. **PiPER 没有抱闸，断电就塌** —— disable / 拔线 / 断电前先托住末端。
2. 速度 5~10%，新位姿先 --dry。
3. 任何 move_p 之前先 kin.check；单关节单步 ≤32°。
4. 抓取前确认工作区内没人手、桌面没有易倒物。
5. 急停：物理按钮，或 `armctl.py stop`。

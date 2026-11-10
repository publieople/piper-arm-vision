# 视觉抓取方案（方盒抓取）

## 0. 当前状态（2026-11-04 实测）
- 相机 Orbbec Dabai DC1 装在**夹爪上方**，当前腕部水平，**镜头朝前看房间**
- 方盒在画面**最底边**、被黑色部件（夹爪/支架）挡住一半 → 不能用于抓取
- 色块检测（棕色）会先抓到 1.24m 外的木椅 → 说明**视角必须先修**
- 结论：抓取前必须把腕部转下去，让相机对着桌面/盒子（约 -50~-60°）

## 1. 四个阶段
| 阶段 | 需要动臂 | 产物 |
|---|---|---|
| A 视角摆正（示教模式手拖即可） | 不需要指令，手拖 | 盒子在画面中央，距离 0.4~0.8m |
| B 相机内参标定 | 不需要 | `out/camera_intrinsics.npz` (rms < 0.5px) |
| C 手眼标定 eye-in-hand | **需要**走 12~15 个位姿 | `out/hand_eye.npz` (T_flange→camera) |
| D 抓取执行 | **需要** move_p + 夹爪 | 抓起来并抬起 |

## 2. 抓取数学
```
像素 (u,v) + 深度 d  --内参-->  相机系 P_c = ((u-cx)*d/fx, (v-cy)*d/fy, d)
P_base = T_base_flange(t) · T_flange_camera · P_c        # 拍照那一瞬间的法兰位姿
目标位姿 = [P_base.x, P_base.y, P_base.z + 80mm, R_top_down]
```
- `T_flange_camera` 来自 C 阶段
- `T_base_flange` 从 pyAgxArm 的 `get_flange_pose()` 取（拍照时立即读，不要延迟）
- `R_top_down`：先手动把腕部摆到"夹爪竖直向下"的姿态，读一次法兰欧拉角，存成常量

## 3. 执行动作序列（速度 10%）
1. `move_p` 到盒子上方 80mm（approach）
2. `grip` 张开到 盒宽-10mm
3. `move_p` 直线下降到 盒顶 + 盒高/2（抓取高度）
4. `grip` 闭合（力 1~2N），停 0.3s
5. `move_p` 抬起 80mm
6. 每步检查 `motion_status==0`、无错误码、无通信报警，否则 abort

## 4. 安全检查
- 深度无效（0 值）→ 不抓，报错
- 目标点超出工作空间 / 逆解失败 → 不抓
- 夹爪行程 0~70mm：盒子宽度最好 15~55mm
- 抓取高度用盒顶 + 一半盒高，不要用盒底（会撞桌面）

## 5. 文件
- `capture.py` 抓 RGB-D
- `detect_box.py` 色块+深度找盒子，输出相机系 3D
- `calibrate_intrinsic.py` 棋盘格内参（`calib/chessboard_9x6_25mm.pdf` A4 打印）
- 待写：`handeye.py`、`grasp.py`

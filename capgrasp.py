#!/usr/bin/env python3
"""capsgrasp —— 夹瓶盖（白色/深色瓶盖通用）

流程（全部实机验证过）：
  1. 观察位拍一帧，HoughCircles 找瓶盖圆（远视位最稳；近距会误检瓶肩，故只在观察位用）
  2. 像素伺服：把瓶盖像素拉到 TARGET=(369,315)（= 瓶盖正好落在工具轴上的像素，
     由闭爪指尖实测 (409,352) 与相机偏置反推；两个独立推导都给 367~369）
  3. 沿工具轴竖直下探：从 z=START_Z 每 6mm 一段，每段"闭合 -> 判宽 -> 回抬 6mm 复测"
     - 宽 ~59mm = 瓶身/颈环，松开继续往下
     - 宽 27~46mm 且回抬仍保持 = 夹到瓶盖，停
     - 宽 26.4 / 力 -0.15 = 空夹，继续往下
  4. 抬 30mm 复测，可选 --place 放下

为什么用"闭合+回抬复测"而不是按高度算：白盖不反深度（DC1 近端又 <165mm），
  盖顶 z 测不到；而"夹住并抬 6mm 仍保持"是唯一可靠的判据（实机在 z=324 只接触不保持、
  z=318 才真正夹住）。

用法：
  capgrasp.py [--keep] [--dry] [--place X Y] [--look X Y Z] [--start-z 336] [--target-px U V]
"""
import os
import argparse, json, math, sys, time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import autograsp as A
import kin

RPY = [180.0, 30.0, 168.0]          # 实测唯一可达的俯视姿态（竖直 rpy(180,0,168) 在该距离 IK 无解）
LOOK_XYZ = [300.0, -60.0, 340.0]    # 观察位：IK 余量 12~13°
TARGET_PX = np.array([369.0, 315.0])
CALIB = "/home/po/vision/calib/servo_calib.json"

OPEN_W, CLOSE_W, CLOSE_F = 76.0, 26.0, 2.0
W_BODY = 46.0        # 宽 >46 判为瓶身/颈环（可乐盖 33.8，白盖 34.3，瓶身 57~59）
W_CAP_LO = 27.0      # 夹住瓶盖的下限
F_HIT = 0.8          # 有接触的力门限
F_HOLD = 0.9         # 真夹住的力门限

START_Z = 336.0      # 下探起点（瓶盖顶通常在这附近或更低）
STEP = 6.0
Z_MIN = 292.0
HOLD_LIFT = 6.0      # 每次闭合后回抬这个距离复测


def load_calib():
    return json.load(open(CALIB))


def euler_to_R(rx, ry, rz):
    rx, ry, rz = math.radians(rx), math.radians(ry), math.radians(rz)
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _white_frac(bgr, u, v, r):
    """圆内白色像素占比，用来把白色瓶盖和胶带卷/暗色圆物分开。"""
    h, w = bgr.shape[:2]
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (int(round(u)), int(round(v))), max(6, int(r * 0.55)), 255, -1)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sel = hsv[m > 0]
    if sel.size == 0:
        return 0.0
    return float(np.mean((sel[:, 1] < 80) & (sel[:, 2] > 140)))


def find_cap_circle(bgr, hint=None, white=True):
    """HoughCircles 找瓶盖；返回 (u, v, r) 或 None。

    white=True 时优先选"圆内大部分是白色"的候选，能排除胶带卷、桌面暗圆之类。
    hint 存在时优先选离 hint 近的，保证伺服不会在相邻两帧之间跳目标。
    """
    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (7, 7), 1.5)
    for p2 in (40, 32, 26):
        c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=120,
                             param1=110, param2=p2, minRadius=35, maxRadius=112)
        if c is None:
            continue
        cs = [tuple(map(float, x)) for x in c[0]]
        if white:
            good = [x for x in cs if _white_frac(bgr, x[0], x[1], x[2]) >= 0.45]
            if good:
                cs = good
        if hint is not None:
            near = [x for x in cs if math.hypot(x[0] - hint[0], x[1] - hint[1]) < 200]
            cs = near or cs
            cs.sort(key=lambda x: math.hypot(x[0] - hint[0], x[1] - hint[1]))
        else:
            cs.sort(key=lambda x: -x[2])
        return cs[0]
    return None


def servo_pixel(arm, target=TARGET_PX, max_iter=8, gain=0.6, tol=10.0, log=print):
    """像素闭环：把瓶盖像素拉到 target。只用 Jinv + J6 旋转补偿，不需要绝对坐标。"""
    cfg = load_calib()
    Jinv = np.array(cfg["Jinv"], float)
    yaw_gain = float(cfg.get("yaw_gain", 1.0))
    cj6 = float(cfg.get("calib_j6", 0.0))
    bgr, _ = A.capture("srv0")
    if bgr is None:
        return None, None
    px = find_cap_circle(bgr)
    if px is None:
        return None, None
    hint = (px[0], px[1])
    sg = gain
    for it in range(max_iter):
        err = np.asarray(target, float) - np.array(px[:2], float)
        n = float(np.linalg.norm(err))
        log("  [伺服 %d] px(%.0f,%.0f) 误差 %.0f px" % (it, px[0], px[1], n))
        if n < tol:
            break
        q = A.cur_joints()
        j6 = float(q[5]) if q is not None else cj6
        th = math.radians(yaw_gain * (j6 - cj6))
        Rm = np.array([[math.cos(th), math.sin(th)], [-math.sin(th), math.cos(th)]])
        stp = Jinv @ (Rm @ err) * sg
        sn = float(np.linalg.norm(stp))
        if sn > 18.0:
            stp = stp / sn * 18.0
        xyz, _ = A.pose(arm)
        tgt = [float(xyz[0] + stp[0]), float(xyz[1] + stp[1]), float(xyz[2])]
        if not move_p_joint(arm, tgt, RPY, speed=8, steps=1, label="[srv]"):
            log("  伺服步进失败")
            break
        bgr2, _ = A.capture("srv%d" % (it + 1))
        if bgr2 is None:
            break
        p2 = find_cap_circle(bgr2, hint)
        if p2 is None:
            log("  丢了瓶盖")
            break
        if float(np.linalg.norm(np.asarray(target, float) - np.array(p2[:2]))) > n * 1.05:
            sg *= 0.55
            log("  误差变大，增益降到 %.2f" % sg)
        px = p2
        hint = (px[0], px[1])
    xyz, _ = A.pose(arm)
    return px, xyz


def desc_end_grasp(arm, eff, start=np.array([336.0]), axis=None, log=print):
    """沿工具轴下探，返回夹住的 z 与宽度，或 None。"""
    a = axis
    zs = []
    z = float(start[0])
    while z >= Z_MIN:
        zs.append(z)
        z -= STEP
    for zt in zs:
        cur, _ = A.pose(arm)
        t = np.array(cur) + a * ((zt - cur[2]) / a[2])
        if not move_p_joint(arm, [float(t[0]), float(t[1]), float(t[2])], RPY, speed=8, steps=1, label="[下探]"):
            log("  z=%.0f 不可达，跳过" % zt)
            continue
        w, f = A.gripper(eff, CLOSE_W, CLOSE_F, seconds=1.7)
        engaged = (w > W_CAP_LO * 0.9) and abs(f) > F_HIT
        tag = ""
        if not engaged:
            tag = "空夹"
        elif w > W_BODY:
            tag = "瓶身"
        else:
            tag = "疑似瓶盖"
        log("  [z %.0f] 宽 %.1f 力 %.2f -> %s" % (zt, w, f, tag))
        if engaged:
            c2, _ = A.pose(arm)
            back = np.array(c2) - a * HOLD_LIFT
            move_p_joint(arm, [float(back[0]), float(back[1]), float(back[2])], RPY, speed=6, steps=1, label="[复测]")
            w2, f2 = A.gripper(eff, CLOSE_W, CLOSE_F, seconds=1.4)
            log("     回抬 %.0fmm 宽 %.1f 力 %.2f" % (HOLD_LIFT, w2, f2))
            if w2 > W_CAP_LO and abs(f2) > F_HOLD and w2 <= W_BODY:
                return zt, w2, f2
            if w2 > W_BODY and abs(f2) > F_HOLD:
                log("     夹到瓶身，松开继续往下")
            A.gripper(eff, OPEN_W, 1.0, seconds=1.0)
            move_p_joint(arm, [float(c2[0]), float(c2[1]), float(c2[2])], RPY, speed=6, steps=1, label="[回位]")
        else:
            A.gripper(eff, OPEN_W, 1.0, seconds=1.0)
    return None



def move_p_joint(arm, xyz, rpy, speed=10, steps=2, label="[vj]", tol=3.0):
    """笛卡尔目标 -> 关节空间执行（控制器自己的笛卡尔 IK 会偶发整帧静默丢目标，
    实测只有走 move_j 才稳定；到位误差 0.5~1.6mm）。返回 True/False。"""
    q0 = A.cur_joints()
    try:
        info = kin.check(list(xyz), list(rpy), seed=q0)
    except Exception as e:
        print("  %s IK 预检异常: %s" % (label, e))
        return False
    if not info.get("ok") or info.get("q") is None:
        print("  %s 不可达: %s" % (label, info.get("msg")))
        return False
    q1 = list(info["q"])
    n = max(1, int(steps))
    path = [[round(a + (b - a) * k / n, 2) for a, b in zip(q0, q1)] for k in range(1, n + 1)]
    for q in path:
        mx = max(abs(a - b) for a, b in zip(q0, q))
        ok, err = A.move_j_go(arm, q, speed=speed, timeout=max(15.0, mx / 3.0 * 4.0), label=label)
        if not ok:
            print("  %s 关节移动未收敛" % label)
            return False
        q0 = q
    xyz2, _ = A.pose(arm)
    d = float(np.linalg.norm(np.array(xyz2) - np.array(xyz)))
    print("%s 到位偏差 %.1f mm（IK 余量 %.0f deg）" % (label, d, min(info["margins"].values())))
    return d <= 6.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="夹住后保持不放")
    ap.add_argument("--dry", action="store_true", help="只检测+对中，不抓")
    ap.add_argument("--place", nargs=2, type=float, metavar=("X", "Y"))
    ap.add_argument("--look", nargs=3, type=float, metavar=("X", "Y", "Z"))
    ap.add_argument("--start-z", type=float, default=START_Z)
    ap.add_argument("--target-px", nargs=2, type=float, default=list(TARGET_PX))
    ap.add_argument("--tag", default="cap")
    args = ap.parse_args()

    look = list(args.look) if args.look else list(LOOK_XYZ)
    target = np.array(args.target_px, float)
    log = print

    arm, eff = A.connect()
    a = euler_to_R(*RPY) @ np.array([0.0, 0.0, 1.0])

    A.gripper(eff, OPEN_W, 1.0, seconds=1.0)
    log("[1/4] 去观察位 %s" % look)
    if not move_p_joint(arm, look, RPY, speed=10, steps=1, label="[观察位]"):
        log("  到不了观察位")
        A.close_capture()
        return 1

    log("[2/4] 像素伺服对中 → (%.0f,%.0f)" % (target[0], target[1]))
    px, xyz = servo_pixel(arm, target=target, log=log)
    if px is None:
        log("  观察位没找到瓶盖（%s）：先把瓶子放到画面里，或用 --look 换观察位" % args.tag)
        A.close_capture()
        return 1
    log("  对中完成 法兰 %s 瓶盖像素 (%0.f,%0.f)" % (np.round(xyz, 1).tolist(), px[0], px[1]))
    if args.dry:
        log("[dry] 到此为止")
        A.close_capture()
        return 0

    log("[3/4] 沿工具轴下探夹取（起点 z=%.0f，每 %.0fmm 复测一次）" % (args.start_z, STEP))
    got = desc_end_grasp(arm, eff, start=np.array([args.start_z]), axis=a, log=log)
    if got is None:
        log("  没夹到瓶盖；机械臂停在 z=%.1f" % A.pose(arm)[0][2])
        A.close_capture()
        return 2
    zt, w, f = got
    log("  [夹住] z=%.0f 宽 %.1f 力 %.2f" % (zt, w, f))

    log("[4/4] 抬升验证")
    for dz in (15.0, 30.0):
        cur, _ = A.pose(arm)
        if not move_p_joint(arm, [float(cur[0]), float(cur[1]), float(cur[2] + dz)], RPY, speed=6, steps=1, label="[抬升]"):
            log("  抬到 +%.0f 不可达（已到工作空间边缘），停在当前高度" % dz)
            break
        w2, f2 = A.gripper(eff, CLOSE_W, CLOSE_F, seconds=1.2)
        log("  +%.0fmm 法兰 %s 宽 %.1f 力 %.2f" % (dz, np.round(A.pose(arm)[0], 1).tolist(), w2, f2))
        if not (w2 > W_CAP_LO and abs(f2) > F_HOLD):
            log("  夹持丢失")
            A.close_capture()
            return 3
    bgr, _ = A.capture("held")
    if bgr is not None:
        p = "/home/po/vision/out/%s_held.png" % args.tag
        cv2.imwrite(p, bgr)
        log("  证据图 %s" % p)

    if args.place:
        px_, py_ = args.place
        cur, _ = A.pose(arm)
        log("[放置] → (%.0f, %.0f)" % (px_, py_))
        move_p_joint(arm, [px_, py_, float(cur[2])], RPY, speed=6, steps=1, label="[平移]")
        move_p_joint(arm, [px_, py_, float(cur[2] - 40.0)], RPY, speed=4, steps=1, label="[下降]")
        A.gripper(eff, OPEN_W, 1.0, seconds=1.5)
        move_p_joint(arm, [px_, py_, float(cur[2] + 40.0)], RPY, speed=6, steps=1, label="[抬起]")
        log("  已放下")

    if args.keep:
        log("[keep] 保持夹持，机械臂停在这（别断电）")
    else:
        log("  完成")
    A.close_capture()
    return 0


if __name__ == "__main__":
    sys.exit(main())

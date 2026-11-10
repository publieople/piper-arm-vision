#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""侧向（前倾 45°）夹瓶身。

流程：
 1. 观察位，像素伺服（tol 5px, 15 次）把瓶盖对到工具轴 → 瓶盖基座坐标 = 法兰 + 168×工具轴
 2. 瓶盖 xy 就是瓶子轴线，抓取点 = (x, y, z_grip)，z_grip 从 85mm 往下试
 3. 45° 倾角，沿工具轴斜插到抓取点（分 15mm 段），闭合；宽 50~72mm 且力 >1.5N 算夹住
 4. 回抬 8mm 复测；成功抬 40mm，默认自动放回

用法：
  python bottleside.py            # 夹 → 抬 40mm → 放回
  python bottleside.py --keep     # 夹住不放
  python bottleside.py --dry      # 只对中 + 打印计划
"""
import os
import argparse, math, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import autograsp as A
import capgrasp as G
import kin

TIP_D = 168.0
PRE_BACK = 110.0
TILTS = [45.0, 40.0, 50.0, 35.0, 55.0]
Z_GRIPS = [85.0, 65.0, 45.0]
W_BODY_MIN, W_BODY_MAX = 50.0, 72.0
F_HIT, F_HOLD = 1.5, 1.0
INSERT_STEP = 15.0
HOLD_LIFT = 8.0
LIFT = 40.0


def axis_of(rpy):
    return G.euler_to_R(*rpy) @ np.array([0.0, 0.0, 1.0])


def plan_tilt(bottle_xy, zg, log=print):
    """选可达的前倾角；返回 (tilt, rpy, pre, grip)"""
    yaw = math.degrees(math.atan2(bottle_xy[1], bottle_xy[0])) + 180.0
    yaw = (yaw + 180.0) % 360.0 - 180.0
    for tilt in TILTS:
        rpy = [180.0, tilt, yaw]
        a = axis_of(rpy)
        gp = np.array([bottle_xy[0], bottle_xy[1], zg], float)
        grip = gp - TIP_D * a
        pre = gp - (TIP_D + PRE_BACK) * a
        try:
            r1 = kin.check([round(v, 1) for v in pre], rpy)
            r2 = kin.check([round(v, 1) for v in grip], rpy)
        except Exception as e:
            log("  [tilt %.0f°] IK 异常 %s" % (tilt, e)); continue
        ok = r1["ok"] and r2["ok"] and min(r1["worst"], r2["worst"]) >= 8.0
        log("  [tilt %2.0f°] 预 %s %s | 夹 %s %s" % (
            tilt, np.round(pre, 0).tolist(), ("余量 %.0f°" % r1["worst"]) if r1["ok"] else "无解",
            np.round(grip, 0).tolist(), ("余量 %.0f°" % r2["worst"]) if r2["ok"] else "无解"))
        if ok:
            return tilt, rpy, pre, grip
    return None


def insert_and_close(arm, eff, pre, gp, rpy, a, log=print):
    """从 pre 沿 +a 斜插到 gp，再往深里多推 0/20/40mm 试闭合"""
    for extra in (0.0, 20.0, 40.0):
        tgt = gp + extra * a
        d = float(np.linalg.norm(tgt - pre))
        n = max(1, int(math.ceil(d / INSERT_STEP)))
        okmove = True
        for i in range(1, n + 1):
            p = pre + (tgt - pre) * (i / n)
            if not G.move_p_joint(arm, [round(float(v), 1) for v in p], rpy, speed=8, steps=1,
                                  label="[插%d/%d]" % (i, n), tol=3.0):
                log("  斜插第 %d 段失败" % i); okmove = False; break
        if not okmove:
            return None
        w, f = A.gripper(eff, G.CLOSE_W, G.CLOSE_F, seconds=1.7)
        log("  插入 %+.0fmm 闭合 宽 %.1f 力 %.2f" % (extra, w, f))
        if W_BODY_MIN < w < W_BODY_MAX and abs(f) > F_HIT:
            return w, f
        A.gripper(eff, G.OPEN_W, 1.0, seconds=1.0)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--look", nargs=3, type=float, metavar=("X", "Y", "Z"))
    ap.add_argument("--tilt", type=float)
    ap.add_argument("--z", type=float, help="抓取高度（瓶身中心线到桌面上的高度），默认 85 起试")
    args = ap.parse_args()

    arm, eff = A.connect()
    A.gripper(eff, G.OPEN_W, 1.0, seconds=1.0)
    look = args.look or G.LOOK_XYZ
    print("[1/5] 观察位 %s" % (list(look),))
    if not G.move_p_joint(arm, list(look), G.RPY, speed=10, steps=2, label="[观察位]"):
        print("到不了观察位"); A.close_capture(); return 2

    print("[2/5] 像素伺服（tol 5px）把瓶盖对到工具轴 (%.0f,%.0f)" % tuple(G.TARGET_PX))
    px, fl = G.servo_pixel(arm, target=G.TARGET_PX, max_iter=26, tol=5.0)
    if px is None or np.linalg.norm(np.asarray(G.TARGET_PX) - px[:2]) > 12.0:
        print("  没对中（px=%s）" % (None if px is None else np.round(px[:2], 1).tolist()))
        A.close_capture(); return 1
    cap = np.array(fl, float) + TIP_D * axis_of(G.RPY)
    print("  对中完成 法兰 %s → 瓶盖基座 (%.0f,%.0f,%.0f)"
          % (np.round(fl, 1).tolist(), cap[0], cap[1], cap[2]))

    if args.dry:
        for zg in ([args.z] if args.z else Z_GRIPS):
            print("[dry] z=%.0f" % zg); plan_tilt(cap[:2], zg)
        A.close_capture(); return 0

    zs = [args.z] if args.z else Z_GRIPS
    res = None
    for zg in zs:
        print("[3/5] 抓取点 (%.0f,%.0f,%.0f)" % (cap[0], cap[1], zg))
        p = plan_tilt(cap[:2], zg)
        if p is None:
            print("  没有可行倾角"); continue
        tilt, rpy, pre, grip = p
        a = axis_of(rpy)
        print("      tilt %.0f° 预 %s 夹 %s" % (tilt, np.round(pre, 0).tolist(), np.round(grip, 0).tolist()))
        if not G.move_p_joint(arm, [round(float(v), 1) for v in pre], rpy, speed=10, steps=2, label="[预接近]"):
            print("  到不了预接近点"); continue
        print("[4/5] 沿工具轴斜插 + 闭合")
        r = insert_and_close(arm, eff, pre, np.array([cap[0], cap[1], zg], float), rpy, a)
        if r is None:
            print("  这一档没夹到，抬回预接近点")
            G.move_p_joint(arm, [round(float(v), 1) for v in pre], rpy, speed=10, steps=2, label="[退回]")
            continue
        w, f = r
        c2, _ = A.pose(arm)
        back = np.array(c2, float) - a * HOLD_LIFT
        G.move_p_joint(arm, [round(float(v), 1) for v in back], rpy, speed=6, steps=1, label="[复测]")
        w2, f2 = A.gripper(eff, G.CLOSE_W, G.CLOSE_F, seconds=1.2)
        print("  回抬 %.0fmm 后 宽 %.1f 力 %.2f" % (HOLD_LIFT, w2, f2))
        if W_BODY_MIN < w2 < W_BODY_MAX and abs(f2) > F_HOLD:
            res = (zg, w2, f2); break
        print("  没保持，松开换下一档")
        A.gripper(eff, G.OPEN_W, 1.0, seconds=1.0)
        G.move_p_joint(arm, [round(float(v), 1) for v in pre], rpy, speed=10, steps=2, label="[退回]")

    if res is None:
        print("[x] 没夹到瓶身，法兰 %s" % np.round(A.pose(arm)[0], 0).tolist())
        A.gripper(eff, G.OPEN_W, 1.0, seconds=1.0); A.close_capture(); return 4

    zg, w, f = res
    print("[5/5] 夹住瓶身 z=%.0f 宽 %.1f 力 %.2f，抬 %.0fmm" % (zg, w, f, LIFT))
    p3 = A.pose(arm)[0] - a * LIFT
    G.move_p_joint(arm, [round(float(v), 1) for v in p3], rpy, speed=8, steps=1, label="[抬升]")
    w3, f3 = A.gripper(eff, G.CLOSE_W, G.CLOSE_F, seconds=1.0)
    print("  抬升后 宽 %.1f 力 %.2f" % (w3, f3))
    A.capture("side_held")
    print("[done] 侧向夹住瓶身")
    if args.keep:
        print("  --keep：保持夹住"); A.close_capture(); return 0

    cur, _ = A.pose(arm)
    back = np.array(cur, float) + a * ((zg - (cur[2] + (grip[2] - (grip[2]))) ) / a[2])
    # 沿轴落回到抓取高度
    back = np.array(cur, float) + a * ((grip[2] - cur[2]) / a[2])
    G.move_p_joint(arm, [round(float(v), 1) for v in back], rpy, speed=4, steps=1, label="[落回]")
    A.gripper(eff, G.OPEN_W, 1.0, seconds=1.2)
    cur, _ = A.pose(arm)
    up = np.array(cur, float) - a * 90.0
    G.move_p_joint(arm, [round(float(v), 1) for v in up], rpy, speed=8, steps=1, label="[抬离]")
    A.capture("side_after")
    print("[done] 已放回并抬离，法兰 %s" % np.round(A.pose(arm)[0], 1).tolist())
    A.close_capture()
    return 0


if __name__ == "__main__":
    sys.exit(main())

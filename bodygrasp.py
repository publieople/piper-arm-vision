#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从正上方夹瓶身（不是瓶盖）。

思路：复用 capgrasp 的"瓶盖像素伺服 + 沿工具轴下探"，但**跳过瓶盖**：
先张开手指沿轴下探到瓶肩以下，再逐段闭合，闭合宽度落在 50~72mm
（瓶身 57~59mm，瓶盖 33~38mm）且力 >1.5N 时才算夹到瓶身。

用法：
  /home/po/arm-env/bin/python /home/po/vision/bodygrasp.py            # 夹住保持
  ... --dry          只对中，不抓
  ... --z-start 300 --z-min 240
"""
import os
import argparse, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import autograsp as A
import capgrasp as G

W_BODY_MIN = 50.0
W_BODY_MAX = 72.0
F_HIT = 1.5
F_HOLD = 1.0
HOLD_LIFT = 8.0


def move_open(arm, eff):
    A.gripper(eff, G.OPEN_W, 1.0, seconds=1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--look", nargs=3, type=float, metavar=("X", "Y", "Z"))
    ap.add_argument("--z-start", type=float, default=300.0)
    ap.add_argument("--z-min", type=float, default=240.0)
    ap.add_argument("--step", type=float, default=6.0)
    ap.add_argument("--lift", type=float, default=30.0)
    args = ap.parse_args()

    arm, eff = A.connect()
    xyz, _ = A.pose(arm)
    print("起始法兰 %s" % np.round(xyz, 1).tolist())
    move_open(arm, eff)

    look = args.look or G.LOOK_XYZ
    print("[1/4] 观察位 %s" % (list(look),))
    if not G.move_p_joint(arm, list(look), G.RPY, speed=10, steps=2, label="[观察位]"):
        print("到不了观察位"); A.close_capture(); return 2

    print("[2/4] 像素伺服：把瓶盖对到工具轴 (%.0f,%.0f)" % tuple(G.TARGET_PX))
    px, flange = G.servo_pixel(arm, target=G.TARGET_PX)
    if px is None:
        print("没找到瓶盖"); A.close_capture(); return 1
    print("  对中完成 法兰 %s 瓶盖像素 (%.0f,%.0f)" % (np.round(flange, 1).tolist(), px[0], px[1]))

    a = G.euler_to_R(*G.RPY) @ np.array([0.0, 0.0, 1.0])
    if args.dry:
        print("[dry] 对中完成，到此为止（下探起点 z=%.0f，找宽 %.0f~%.0fmm 的瓶身）"
              % (args.z_start, W_BODY_MIN, W_BODY_MAX))
        A.close_capture(); return 0
    print("[3/4] 沿工具轴下探，找瓶身（宽 %.0f~%.0fmm）" % (W_BODY_MIN, W_BODY_MAX))
    z = args.z_start
    got = None
    while z >= args.z_min:
        cur, _ = A.pose(arm)
        t = np.array(cur, float) + a * ((z - cur[2]) / a[2])
        if not G.move_p_joint(arm, [float(t[0]), float(t[1]), float(t[2])], G.RPY,
                              speed=8, steps=1, label="[下探 %.0f]" % z):
            print("  z=%.0f 不可达，跳过" % z); z -= args.step; continue
        w, f = A.gripper(eff, G.CLOSE_W, G.CLOSE_F, seconds=1.7)
        if W_BODY_MIN < w < W_BODY_MAX and abs(f) > F_HIT:
            c2, _ = A.pose(arm)
            back = np.array(c2, float) - a * HOLD_LIFT
            G.move_p_joint(arm, [float(back[0]), float(back[1]), float(back[2])], G.RPY,
                           speed=6, steps=1, label="[回抬]")
            w2, f2 = A.gripper(eff, G.CLOSE_W, G.CLOSE_F, seconds=1.2)
            print("  [z %.0f] 宽 %.1f 力 %.2f -> 瓶身？回抬 %.0fmm 后 宽 %.1f 力 %.2f"
                  % (z, w, f, HOLD_LIFT, w2, f2))
            if W_BODY_MIN < w2 < W_BODY_MAX and abs(f2) > F_HOLD:
                got = (z, w2, f2)
                break
            print("      回抬后没保持，松开继续")
            move_open(arm, eff)
            z -= args.step
            continue
        tag = "空夹" if not (abs(f) > 0.5) else ("瓶盖/颈环" if w < W_BODY_MIN else "太宽")
        print("  [z %.0f] 宽 %.1f 力 %.2f -> %s" % (z, w, f, tag))
        move_open(arm, eff)
        z -= args.step

    if got is None:
        print("[x] 没夹到瓶身，机械臂停在 %s" % np.round(A.pose(arm)[0], 0).tolist())
        move_open(arm, eff); A.close_capture(); return 4

    print("[4/4] 夹住瓶身 z=%.0f 宽 %.1f 力 %.2f，抬 %.0fmm" % (got[0], got[1], got[2], args.lift))
    p3 = A.pose(arm)[0] - a * args.lift
    G.move_p_joint(arm, [float(p3[0]), float(p3[1]), float(p3[2])], G.RPY, speed=8, steps=1, label="[抬升]")
    w3, f3 = A.gripper(eff, G.CLOSE_W, G.CLOSE_F, seconds=1.0)
    print("  抬升后 宽 %.1f 力 %.2f" % (w3, f3))
    A.capture("body_held")
    print("[done] 夹住瓶身（瓶子离开桌面 %.0fmm）" % args.lift)
    if args.keep:
        print("  --keep：保持夹住，要放回用 /tmp/putbody.py")
        A.close_capture(); return 0

    # 放回：沿轴落回抓取高度 -> 张开 -> 抬离
    cur, _ = A.pose(arm)
    back = np.array(cur, float) + a * ((got[0] - cur[2]) / a[2])
    G.move_p_joint(arm, [float(back[0]), float(back[1]), float(back[2])], G.RPY,
                   speed=4, steps=1, label="[落回]")
    move_open(arm, eff)
    cur, _ = A.pose(arm)
    up = np.array(cur, float) - a * 80.0
    G.move_p_joint(arm, [float(up[0]), float(up[1]), float(up[2])], G.RPY, speed=8, steps=1, label="[抬离]")
    A.capture("after_body_put")
    print("[done] 已放回并抬离，法兰 %s" % np.round(A.pose(arm)[0], 1).tolist())
    A.close_capture()
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""cap3d.py — 纯彩色瓶盖定位（不依赖深度）

思路：黑色瓶盖在绿垫/白色瓶身上是高对比色块；用椭圆拟合拿像素中心与主轴长度，
再用小孔成像按已知盖径(30mm)反推距离 Z = fx * D / 主轴像素。
"""
import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import autograsp as A  # noqa: E402  只为了 pose/capture


def euler_to_R(rx, ry, rz):
    """与 capgrasp/autograsp 一致：R = Rz(rz) @ Ry(ry) @ Rx(rx)"""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

INTR = np.load("/home/po/vision/out/camera_intrinsics.npz")
FX, FY = float(INTR["fx"]), float(INTR["fy"])
CX, CY = float(INTR["cx"]), float(INTR["cy"])
DIST = INTR["dist"].reshape(-1)
HE = np.load("/home/po/vision/out/hand_eye.npz")
R_FC, T_FC = HE["R"], HE["t"].reshape(3)


def black_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    m = ((v < 90) & (s < 160)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # 去掉贴边（夹爪手指、画面外物体）
    m[:6, :] = 0
    m[-6:, :] = 0
    m[:, :6] = 0
    m[:, -6:] = 0
    return m


def find_cap_color(bgr, D_mm=30.0, want=(20.0, 120.0)):
    """返回 dict(u,v,area,major,minor,Z_mm,P_cam_mm,ell) 或 None"""
    m = black_mask(bgr)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 1500:
            continue
        if len(c) < 5:
            continue
        (u, v), (d1, d2), ang = cv2.fitEllipse(c)
        major, minor = max(d1, d2), min(d1, d2)
        if minor < 8:
            continue
        ar = major / minor
        if not (1.0 <= ar <= 1.45):
            continue
        if not (want[0] <= major <= want[1]):
            continue
        Z = FX * D_mm / major
        if not (90.0 <= Z <= 260.0):
            continue
        score = abs(ar - 1.15) + abs(a / (np.pi * major * minor / 4.0) - 0.75)
        if best is None or score < best[0]:
            best = (score, dict(u=u, v=v, area=a, major=major, minor=minor, ar=ar,
                                Z=Z, ang=ang, ell=((u, v), (d1, d2), ang)))
    return None if best is None else best[1]


def deproject_to_base(u, v, Z, xyz_mm, rpy_deg):
    """像素+光轴距离 -> 基座系点(mm)"""
    pts = np.array([[[u, v]]], np.float32)
    und = cv2.undistortPoints(pts, np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]]), DIST)
    x, y = float(und[0, 0, 0]), float(und[0, 0, 1])
    Pc = np.array([x * Z, y * Z, Z])  # mm, 相机系
    # 相机 -> 法兰 -> 基座
    R_fb = euler_to_R(*[np.radians(v) for v in rpy_deg])
    p_f = R_FC @ Pc + T_FC * 1000.0
    pb = R_fb @ p_f + np.array(xyz_mm)
    return Pc, p_f, pb


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "c3d"
    xyz = [float(v) for v in sys.argv[2:5]] if len(sys.argv) >= 5 else None
    rpy = [float(v) for v in sys.argv[5:8]] if len(sys.argv) >= 8 else None
    arm, eff = A.connect()
    if xyz:
        ok = A.move_p(arm, xyz + (rpy or [180, 35, 168]), speed=10, timeout=20.0, force=True)
        print("move", ok)
    bgr, dep = A.capture(tag)
    A.close_capture()
    p, r = A.pose(arm)
    print("pose", np.round(p, 1).tolist(), np.round(r, 1).tolist())
    c = find_cap_color(bgr)
    if c is None:
        print("没找到瓶盖色块")
        return 1
    Pc, pf, pb = deproject_to_base(c["u"], c["v"], c["Z"], p, r)
    print("盖: px(%d,%d) 主轴 %.0fpx 副轴 %.0fpx 圆度 %.2f  距离 %.0fmm" % (
        c["u"], c["v"], c["major"], c["minor"], c["ar"], c["Z"]))
    print("相机系", np.round(Pc, 1).tolist(), " 基座系", np.round(pb, 1).tolist())
    vis = bgr.copy()
    cv2.ellipse(vis, c["ell"], (0, 255, 0), 2)
    cv2.circle(vis, (int(c["u"]), int(c["v"])), 4, (0, 0, 255), -1)
    cv2.imwrite("/home/po/vision/out/%s_cap3d.png" % tag, vis)
    return 0


if __name__ == "__main__":
    sys.exit(main())

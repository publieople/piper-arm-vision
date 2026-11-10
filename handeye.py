#!/home/po/arm-env/bin/python
"""eye-in-hand 手眼标定 + 顺带内参标定（棋盘格 9x6，方格 25mm）

流程:
  1. 机械臂在当前视角附近走 N 个位姿，每个位姿: 等稳 -> 拍图 -> 记录法兰位姿
  2. 用所有图做 calibrateCamera（内参，若已有可复用）
  3. 每张图 solvePnP 得到标定板在相机系下的位姿
  4. cv2.calibrateHandEye 求 T_flange->camera (AX=XB)

用法:
  /home/po/arm-env/bin/python handeye.py collect --poses 15
  /home/po/arm-env/bin/python handeye.py solve

注意: 标定板必须固定在桌面上（不动），机械臂动、板子不动。
"""
import argparse
import itertools
import json
import math
import os
import subprocess
import sys
import time

import cv2
import numpy as np

PATTERN = (9, 6)
SQUARE_MM = 25.0  # 可用 --square 覆盖（打印/屏幕缩放后实测值）
OUT = "/home/po/vision/out"
CALIB = "/home/po/vision/calib"
DATA = f"{CALIB}/handeye_data.npz"


def joints_deg(arm):
    j = arm.get_joint_angles()
    return [x * 180.0 / math.pi for x in j.msg]


def flange_pose(arm):
    fp = arm.get_flange_pose().msg
    return [fp[0], fp[1], fp[2], fp[3], fp[4], fp[5]]  # m, rad


def euler_to_R(rx, ry, rz):
    """SDK 约定：extrinsic XYZ == R = Rz(rz) @ Ry(ry) @ Rx(rx)（弧度）"""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def find_corners(gray):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, cs = cv2.findChessboardCorners(gray, PATTERN, flags)
    if not ok:  # 兜底：缺墨/低对比时先做 CLAHE
        gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
        ok, cs = cv2.findChessboardCorners(gray, PATTERN, flags)
    if not ok:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, cs, (11, 11), (-1, -1), crit)


def capture(tag):
    subprocess.run(["/home/po/vision-env/bin/python", "/home/po/vision/capture.py",
                    "--rotate", "180", "--tag", tag], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def collect(n_poses):
    from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW
    cfg = create_agx_arm_config(robot=ArmModel.PIPER, firmeware_version=PiperFW.V189, channel="can0")
    arm = AgxArmFactory.create_arm(cfg)
    arm.connect()
    time.sleep(0.8)
    base = joints_deg(arm)
    print("base joints:", " ".join("%.2f" % v for v in base))

    # 关节空间小幅扰动（每个位姿都让板子留在视野内）
    deltas = [
        (0, 0, 0, 0, 0, 0),
        (6, 0, 0, 0, 0, 0), (-6, 0, 0, 0, 0, 0),
        (0, 4, 0, 0, 0, 0), (0, -4, 0, 0, 0, 0),
        (0, 0, 4, 0, 0, 0), (0, 0, -4, 0, 0, 0),
        (0, 0, 0, 10, 0, 0), (0, 0, 0, -10, 0, 0),
        (0, 0, 0, 0, 6, 0), (0, 0, 0, 0, -6, 0),
        (0, 0, 0, 0, 0, 12), (0, 0, 0, 0, 0, -12),
        (5, 3, 0, 8, 4, 6), (-5, -3, 0, -8, -4, -6),
        (4, -2, 3, -6, 5, -9), (-4, 2, -3, 6, -5, 9),
    ]
    deltas = deltas[:n_poses]
    os.makedirs(f"{CALIB}/he_imgs", exist_ok=True)
    images, poses, corners = [], [], []
    try:
        for i, d in enumerate(deltas):
            tgt = [b + x for b, x in zip(base, d)]
            arm.set_speed_percent(8)
            t_end = time.time() + 0.5
            while time.time() < t_end:
                arm.move_j([math.radians(v) for v in tgt])
                time.sleep(0.02)
            time.sleep(0.6)
            tag = "he_%02d" % i
            capture(tag)
            img = cv2.imread(f"{OUT}/{tag}_color.png")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cs = find_corners(gray)
            if cs is None:
                print("  pose %d: 没找到棋盘格，跳过" % i)
                continue
            images.append(f"{OUT}/{tag}_color.png")
            poses.append(flange_pose(arm))
            corners.append(cs)
            print("  pose %d ok  joints: %s" % (i, " ".join("%.1f" % v for v in joints_deg(arm))))
        # 回到原位
        t_end = time.time() + 0.6
        while time.time() < t_end:
            arm.move_j([math.radians(v) for v in base])
            time.sleep(0.02)
    finally:
        arm.disconnect()

    np.savez(DATA, images=np.array(images), poses=np.array(poses),
             corners=np.array(corners))
    print("saved", DATA, "with", len(images), "views")


def solve():
    d = np.load(DATA, allow_pickle=True)
    images, poses, corners = d["images"], d["poses"], d["corners"]
    objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    # 一切用米：t_g2b 来自 get_flange_pose（m），objp 必须同为 m，否则 t 相差 1000 倍
    objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * (SQUARE_MM / 1000.0)
    K = None
    if os.path.exists(f"{OUT}/camera_intrinsics.npz"):
        kd = np.load(f"{OUT}/camera_intrinsics.npz")
        K, dist = kd["K"], kd["dist"]
        print("use existing intrinsics:", np.round([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], 2))
    else:
        rms, K, dist, _, _ = cv2.calibrateCamera([objp] * len(corners), list(corners),
                                                 (640, 480), None, None)
        print("intrinsics from these views: rms %.3f" % rms)
        np.savez(f"{OUT}/camera_intrinsics.npz", fx=K[0, 0], fy=K[1, 1], cx=K[0, 2],
                 cy=K[1, 2], K=K, dist=dist, rms=np.array(rms), size=np.array([640, 480]))
    R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
    for cs, pose in zip(corners, poses):
        ok, rvec, tvec = cv2.solvePnP(objp, cs, K, dist)
        if not ok:
            continue
        R_t2c.append(cv2.Rodrigues(rvec)[0])
        t_t2c.append(tvec.reshape(3, 1))
        R_g2b.append(euler_to_R(*pose[3:]))
        t_g2b.append(np.array(pose[:3]).reshape(3, 1))
    print("valid pairs:", len(R_t2c))
    names = {0: "TSAI", 1: "PARK", 2: "HORAUD", 3: "ANDREFF", 4: "DANIILIDIS"}
    results = []
    for m in (cv2.CALIB_HAND_EYE_TSAI, cv2.CALIB_HAND_EYE_PARK,
              cv2.CALIB_HAND_EYE_HORAUD, cv2.CALIB_HAND_EYE_ANDREFF):
        try:
            R, t = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=m)
        except cv2.error as e:
            print("method", names.get(m, m), "failed:", str(e).splitlines()[-1][:60])
            continue
        results.append((m, R, t))
        print("method %-8s t(mm) %s  |t| %.1f" % (names.get(m, m),
              np.round(t.ravel() * 1000, 1), np.linalg.norm(t) * 1000))
    if not results:
        raise SystemExit("全部算法失败")
    ts = np.array([t.ravel() for _, _, t in results]) * 1000
    print("平移互差 max-min (mm):", np.round(ts.max(0) - ts.min(0), 1) if len(ts) > 1 else "n/a")
    R, t = results[0][1], results[0][2]
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.ravel()
    np.savez(f"{OUT}/hand_eye.npz", R=R, t=t, T=T)
    print("saved", f"{OUT}/hand_eye.npz")
    print("T_flange_camera (mm):")
    print(np.round(np.vstack([T[:3, :3], T[:3, 3] * 1000]), 2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "solve"])
    ap.add_argument("--poses", type=int, default=15)
    ap.add_argument("--square", type=float, default=25.0, help="方格实际边长 mm")
    a = ap.parse_args()
    SQUARE_MM = a.square
    if a.cmd == "collect":
        collect(a.poses)
    else:
        solve()

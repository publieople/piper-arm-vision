#!/usr/bin/env python3
"""相机内参标定（棋盘格 9x6 内角点，方格 25mm，A4）

无需机械臂运动：手拿棋盘格在相机前换角度/距离即可。

用法:
  # 自动模式：60 秒内自动抓 20 张（板子进入视野就存），结束后立即标定
  /home/po/vision-env/bin/python calibrate_intrinsic.py --auto

  # 也能指定已有图片目录
  /home/po/vision-env/bin/python calibrate_intrinsic.py --from-dir calib/imgs

输出: out/camera_intrinsics.npz (fx, fy, cx, cy, dist, rms) + calib/report.txt
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np
from pyorbbecsdk import (Pipeline, Config, OBSensorType, OBFormat)

PATTERN = (9, 6)          # 内角点
SQUARE_MM = 25.0
OUT = "/home/po/vision/out"
CALIB = "/home/po/vision/calib"


def open_cam():
    p = Pipeline()
    cfg = Config()
    cl = p.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    prof = cl.get_default_video_stream_profile()
    cfg.enable_stream(prof)
    p.start(cfg)
    return p, prof


def to_bgr(cf):
    fmt, h, w = cf.get_format(), cf.get_height(), cf.get_width()
    data = np.frombuffer(cf.get_data(), dtype=np.uint8)
    if fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if fmt == OBFormat.RGB:
        return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    if fmt == OBFormat.YUYV:
        return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUYV)
    return data.reshape(h, w, 3).copy()


def find_corners(gray):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(gray, PATTERN, flags)
    if not ok:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
    return corners


def collect_auto(seconds, max_views):
    p, prof = open_cam()
    print("color profile:", prof.get_width(), "x", prof.get_height(), prof.get_fps(), "fps")
    os.makedirs(f"{CALIB}/imgs", exist_ok=True)
    saved, t0, last = 0, time.time(), 0.0
    try:
        while time.time() - t0 < seconds and saved < max_views:
            fr = p.wait_for_frames(1000)
            if fr is None:
                continue
            cf = fr.get_color_frame()
            if cf is None:
                continue
            img = to_bgr(cf)
            img = cv2.rotate(img, cv2.ROTATE_180)   # 相机倒装
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cs = find_corners(gray)
            now = time.time()
            if cs is not None and now - last > 1.0:
                cv2.imwrite(f"{CALIB}/imgs/view_{saved:02d}.png", img)
                saved += 1
                last = now
                print("captured %d/%d" % (saved, max_views))
    finally:
        p.stop()
    return saved


def calibrate(image_paths):
    objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE_MM
    obj_pts, img_pts, used = [], [], []
    size = None
    for f in sorted(image_paths):
        img = cv2.imread(f)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cs = find_corners(gray)
        if cs is None:
            print("skip (no corners):", f)
            continue
        size = gray.shape[::-1]
        obj_pts.append(objp)
        img_pts.append(cs)
        used.append(f)
    if len(obj_pts) < 6:
        print("有效图片不足 (%d)，至少 6 张" % len(obj_pts))
        return None
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    print("rms reprojection error: %.3f px  (%d views, %dx%d)" % (rms, len(obj_pts), size[0], size[1]))
    os.makedirs(OUT, exist_ok=True)
    np.savez(f"{OUT}/camera_intrinsics.npz",
             fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
             K=K, dist=dist, rms=rms, size=np.array(size), views=np.array(len(obj_pts)))
    txt = ["views=%d size=%dx%d rms=%.3f px" % (len(obj_pts), size[0], size[1], rms),
           "fx=%.2f fy=%.2f cx=%.2f cy=%.2f" % (K[0, 0], K[1, 1], K[0, 2], K[1, 2]),
           "dist=%s" % json.dumps(dist.ravel().tolist()),
           "images:" ] + used
    open(f"{CALIB}/report.txt", "w").write("\n".join(txt))
    print("saved", f"{OUT}/camera_intrinsics.npz")
    print(txt[1])
    return K, dist, rms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", action="store_true", help="自动抓图（无窗口）")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--max-views", type=int, default=20)
    ap.add_argument("--from-dir", default=None)
    args = ap.parse_args()

    if args.from_dir:
        calibrate(glob.glob(os.path.join(args.from_dir, "*.png")))
        return
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CALIB, exist_ok=True)
    saved = collect_auto(args.seconds, args.max_views)
    print("collected", saved, "views")
    if saved:
        calibrate(glob.glob(f"{CALIB}/imgs/*.png"))


if __name__ == "__main__":
    main()

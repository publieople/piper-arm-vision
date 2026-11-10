#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""造几条假 ep_XXXX，用来在没有机械臂时验证 采集→LeRobot→训练 这条链。"""
import os, sys
import numpy as np
import cv2

SRC = ["/home/po/vision/out/side_held_color.png", "/home/po/vision/out/body_held_color.png",
       "/home/po/vision/out/look_now_color.png"]
HOME_Q = np.zeros(6)
GRASP_Q = np.array([-12.0, 95.0, -60.0, 0.0, 60.0, 0.0])


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    root = "/home/po/vision/act_data"
    frames = [cv2.imread(p) for p in SRC if os.path.exists(p)]
    if not frames:
        raise SystemExit("找不到素材图")
    for e in range(n):
        ep = os.path.join(root, "ep_%04d" % (e + 1))
        os.makedirs(ep, exist_ok=True)
        T = 120
        q = np.zeros((T, 6))
        for i in range(T):
            a = min(1.0, i / 40.0) * min(1.0, (T - i) / 40.0)
            q[i] = HOME_Q + (GRASP_Q - HOME_Q) * a
        grip = 75.0 - 45.0 * np.clip((np.arange(T) - 70) / 20.0, 0, 1)
        vw = cv2.VideoWriter(os.path.join(ep, "cam.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             20.0, (640, 480))
        for i in range(T):
            img = frames[i % len(frames)].copy()
            dx = int(30 * np.sin(i / 8.0))
            M = np.float32([[1, 0, dx], [0, 1, 10 * np.sin(i / 12.0)]])
            vw.write(cv2.warpAffine(img, M, (img.shape[1], img.shape[0])))
        vw.release()
        np.savez_compressed(os.path.join(ep, "ep.npz"), t=np.arange(T) / 20.0,
                            q=q.astype(np.float32), grip=grip.astype(np.float32),
                            fps=20.0, n_frames=T)
        print("fake", ep)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""capscan.py -- 转 J1 扫一圈, 找红色瓶盖在哪"""
import os
import sys, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2
import autograsp as A

def redmask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (160, 60, 40), (180, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  np.ones((5,5), np.uint8))
    return m

def main():
    arm, eff = A.connect()
    A.gripper(eff, A.CFG["open_mm"], 1.0)
    base_q = list(A.CFG["obs_q"])
    hits = []
    for j1 in (-30, -20, -10, 0, 10, 20, 30):
        q = list(base_q); q[0] = j1
        A.goto_q(arm, [round(v,2) for v in q], steps=2, speed=12)
        time.sleep(0.3)
        bgr, dep = A.capture(f"scan{j1:+03d}")
        if bgr is None: continue
        m = redmask(bgr)
        ys, xs = np.where(m > 0)
        p, rpy = A.pose(arm)
        if xs.size < 200:
            print(f"[J1={j1:+03d}] 无红  (法兰 {p.round(0)})")
            continue
        d = dep[ys, xs]; dv = d[d>0]
        print(f"[J1={j1:+03d}] 红 {xs.size}px 质心 ({np.median(xs):.0f},{np.median(ys):.0f}) "
              f"深中值 {np.median(dv) if dv.size else -1:.0f}mm  法兰 {p.round(0)}")
        cv2.imwrite(f"/home/po/vision/out/scanhit_{j1:+03d}.png", bgr)
        hits.append((j1, float(np.median(xs)), float(np.median(ys))))
    print("hits:", hits)
    A.close_capture()

main()

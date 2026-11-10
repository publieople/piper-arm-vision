#!/usr/bin/env python3
"""scene.py -- 换个高位姿态看全场, 找瓶子在哪、多高"""
import os
import sys, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2
import autograsp as A

arm, eff = A.connect()
A.gripper(eff, A.CFG["open_mm"], 1.0)
poses = {
  "high":  [-20.0, 70.0, -40.0, 0.0, 55.0, 0.0],
  "mid":   [-20.0, 55.0, -25.0, 0.0, 45.0, 0.0],
  "obs":   list(A.CFG["obs_q"]),
}
for name, q in poses.items():
    A.goto_q(arm, [round(v, 2) for v in q], steps=3, speed=12)
    time.sleep(0.3)
    p, rpy = A.pose(arm)
    bgr, dep = A.capture("scene_" + name)
    dv = dep[dep > 0]
    print(f"[{name}] 法兰 {p.round(0)} rpy {rpy.round(0)} 深度有效 {dv.size*100//dep.size}%")
A.close_capture()

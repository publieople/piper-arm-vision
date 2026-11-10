#!/usr/bin/env python3
"""tipmeas.py -- 用深度图量指尖到法兰的距离: 画面下缘两块黑色夹爪"""
import os
import sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2
import autograsp as A

KD = np.load("/home/po/vision/out/camera_intrinsics.npz"); K, DIST = KD["K"], KD["dist"]
Xd = np.load("/home/po/vision/out/hand_eye.npz")
X = np.eye(4); X[:3,:3] = Xd["R"]; X[:3,3] = Xd["t"].ravel()

def e2R(r,p_,y_):
    Rx = np.array([[1,0,0],[0,math.cos(r),-math.sin(r)],[0,math.sin(r),math.cos(r)]])
    Ry = np.array([[math.cos(p_),0,math.sin(p_)],[0,1,0],[-math.sin(p_),0,math.cos(p_)]])
    Rz = np.array([[math.cos(y_),-math.sin(y_),0],[math.sin(y_),math.cos(y_),0],[0,0,1]])
    return Rz@Ry@Rx

arm, eff = A.connect()
A.gripper(eff, A.CFG["open_mm"], 1.0)
bgr, dep = A.capture("tip")
p, rpy = A.pose(arm)
print(f"[flange] {p.round(1)} rpy {rpy.round(1)}")
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
dark = cv2.inRange(hsv, (0,0,0), (180,255,70))          # 黑
dark[:, :120] = 0; dark[:, 520:] = 0                    # 只看中间
dark[: int(dep.shape[0]*0.55), :] = 0                    # 只看下半幅
ys, xs = np.where((dark>0) & (dep>0))
print(f"[fingers] 黑色且深度有效像素 {xs.size}")
if xs.size:
    d = dep[ys, xs].astype(np.float32)
    # 取最靠下(最靠近指尖)的 200 个像素
    idx = np.argsort(ys)[-200:]
    v = ys[idx]; u = xs[idx]; z = d[idx]
    print(f"  采样行 {v.min()}..{v.max()} 深 {np.median(z):.0f}mm")
    und = cv2.undistortPoints(np.stack([u,v],1).astype(np.float64).reshape(-1,1,2), K, DIST, P=K).reshape(-1,2)
    pc = np.stack([(und[:,0]-K[0,2])*z/K[0,0], (und[:,1]-K[1,2])*z/K[1,1], z],1)/1000.0
    T = np.eye(4); T[:3,:3] = e2R(*np.radians(rpy)); T[:3,3] = p/1000.0
    Tf = np.linalg.inv(T) @ np.linalg.inv(X)   # 世界->法兰的反向: 用 T @ X 把相机点变到基座
    pb = (T @ X @ np.hstack([pc, np.ones((pc.shape[0],1))]).T)[:3].T*1000.0
    pf = (np.linalg.inv(X) @ pc.T).T*1000.0   # 法兰系? 需先转法兰: p_f = inv(X)*p_c? 见下
    print(f"  指尖基座坐标 中值 {np.median(pb,0).round(1)}  z 范围 {pb[:,2].min():.0f}..{pb[:,2].max():.0f}")
    print(f"  -> 法兰 z {p[2]:.1f} - 指尖 z {np.median(pb[:,2]):.1f} = {p[2]-np.median(pb[:,2]):.1f} mm")
    # 相机系 -> 法兰系: p_f = X^{-1} p_c
    Xinv = np.linalg.inv(X)
    pfc = (Xinv[:3,:3] @ pc.T + Xinv[:3,3:4]).T*1000.0
    print(f"  指尖在法兰系 z 中值 {np.median(pfc[:,2]):.1f} mm  (范围 {pfc[:,2].min():.0f}..{pfc[:,2].max():.0f})")
    print(f"  指尖在法兰系 x/y 中值 {np.median(pfc[:,0]):.1f}, {np.median(pfc[:,1]):.1f}")
A.close_capture()

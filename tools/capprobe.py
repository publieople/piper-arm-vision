#!/usr/bin/env python3
"""capprobe.py -- 看瓶盖: 到观察位 -> 取图 -> 红色连通域 -> 反投影到基座 -> IK 可达性
不动任何抓取动作, 只做测量.
"""
import os
import sys, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2
import autograsp as A
import kin

KD = np.load("/home/po/vision/out/camera_intrinsics.npz")
K, DIST = KD["K"], KD["dist"]
Xd = np.load("/home/po/vision/out/hand_eye.npz")
X = np.eye(4); X[:3,:3] = Xd["R"]; X[:3,3] = Xd["t"].ravel()

def e2R(r,p_,y_):
    Rx = np.array([[1,0,0],[0,math.cos(r),-math.sin(r)],[0,math.sin(r),math.cos(r)]])
    Ry = np.array([[math.cos(p_),0,math.sin(p_)],[0,1,0],[-math.sin(p_),0,math.cos(p_)]])
    Rz = np.array([[math.cos(y_),-math.sin(y_),0],[math.sin(y_),math.cos(y_),0],[0,0,1]])
    return Rz@Ry@Rx

def flange_T(arm):
    p, rpy = A.pose(arm)
    T = np.eye(4); T[:3,:3] = e2R(*np.radians(rpy)); T[:3,3] = p/1000.0
    return T, p, rpy

def find_cap(bgr, dep, tag="cap"):
    """红色(瓶盖)连通域; 返回质心/深度/尺寸"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo, hi = (160, 70, 50), (180, 255, 255)
    mask = cv2.inRange(hsv, lo, hi)
    # 画面下缘的黑色夹爪会挡, 但不影响红色
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5,5), np.uint8))
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis = bgr.copy()
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(vis, [c], -1, (0,255,0), 2)
        M = cv2.moments(c)
        u, v = M["m10"]/M["m00"], M["m01"]/M["m00"]
        cv2.circle(vis, (int(u), int(v)), 5, (0,0,255), -1)
    cv2.imwrite(f"/home/po/vision/out/{tag}_probe.png", vis)
    cv2.imwrite(f"/home/po/vision/out/{tag}_mask.png", mask)
    return mask

def cap_stats(dep, mask):
    ys, xs = np.where(mask > 0)
    if xs.size == 0: return None
    d = dep[ys, xs].astype(np.float32)
    dv = d[d > 0]
    return {"u": float(np.median(xs)), "v": float(np.median(ys)),
            "n": int(xs.size), "valid": int(dv.size),
            "dmed": float(np.median(dv)) if dv.size else -1.0,
            "w": float(xs.max()-xs.min()), "h": float(ys.max()-ys.min()),
            "ytop": int(ys.min()), "ybot": int(ys.max())}

def main():
    arm, eff = A.connect()
    st = arm.get_arm_status().msg
    print(f"[arm] ctrl={st.ctrl_mode} arm={st.arm_status} motion={st.motion_status}")
    print(f"[arm] q={A.cur_joints()}")
    p, rpy = A.pose(arm)
    print(f"[arm] 法兰 {p.round(1)} rpy {rpy.round(1)}")
    A.gripper(eff, A.CFG["open_mm"], 1.0)
    print("[go] 走观察位姿", A.CFG["obs_q"])
    A.goto_obs(arm)
    p, rpy = A.pose(arm)
    print(f"[obs] 法兰 {p.round(1)} rpy {rpy.round(1)}")

    bgr, dep = A.capture("cap0")
    if bgr is None: sys.exit("取图失败")
    mask = find_cap(bgr, dep, "cap0")
    s = cap_stats(dep, mask)
    print("[cap] " + str(s))
    if s is None or s["valid"] < 200:
        sys.exit("红色瓶盖没测到(或深度无效)")

    zmm = s["dmed"]
    und = cv2.undistortPoints(np.array([[[s["u"], s["v"]]]], np.float64), K, DIST, P=K).reshape(2)
    pc = np.array([(und[0]-K[0,2])*zmm/K[0,0], (und[1]-K[1,2])*zmm/K[1,1], zmm])
    T, p, rpy = flange_T(arm)
    pb = (T @ X @ np.append(pc/1000.0, 1.0))[:3]*1000.0
    print(f"[cap] 像素 ({s['u']:.0f},{s['v']:.0f}) 深 {zmm:.0f}mm -> 基座 {pb.round(1)} mm  "
          f"半径 {math.hypot(pb[0],pb[1]):.0f}mm")

    # 头顶正上方抓取位姿: 工具竖直向下, 法兰在盖上方 dz
    for dz in (60, 80, 100, 120):
        fl = np.array([pb[0], pb[1], 0.0])
        # 先看纯几何: 法兰 z 候选
        for fz in (260, 230, 200):
            info = kin.check([fl[0], fl[1], fz], [180.0, 0.0, 168.0], seed=A.cur_joints())
            print(f"[ik] 盖上方 ({fl[0]:.0f},{fl[1]:.0f},{fz}) rpy(180,0,168) ok={info['ok']} "
                  f"err={info.get('err', -1):.1f} {info.get('msg','')}")
    A.close_capture()

if __name__ == "__main__":
    main()

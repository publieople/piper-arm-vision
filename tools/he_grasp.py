
#!/usr/bin/env python3
"""he_grasp.py -- 用手眼把盒子直接换算到基座系再抓(伺服在臂展边缘不稳时用)"""
import os
import sys, time, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2
import autograsp as A

KD = np.load("/home/po/vision/out/camera_intrinsics.npz"); K = KD["K"]; DIST = KD["dist"]
Xd = np.load("/home/po/vision/out/hand_eye.npz")
X = np.eye(4); X[:3,:3] = Xd["R"]; X[:3,3] = Xd["t"].ravel()

def e2R(r, p_, y_):
    Rx = np.array([[1,0,0],[0,math.cos(r),-math.sin(r)],[0,math.sin(r),math.cos(r)]])
    Ry = np.array([[math.cos(p_),0,math.sin(p_)],[0,1,0],[-math.sin(p_),0,math.cos(p_)]])
    Rz = np.array([[math.cos(y_),-math.sin(y_),0],[math.sin(y_),math.cos(y_),0],[0,0,1]])
    return Rz@Ry@Rx

def box_base(arm, tag="he"):
    bgr, dep = A.capture(tag)
    d = A.find_box(bgr, dep)
    if d is None:
        return None, "看不见盒子"
    z = d["depth_mm"]
    if z < 120 or z > 420:
        return None, f"深度无效 {z:.0f}mm (相机离盒子太近/太远)"
    und = cv2.undistortPoints(np.array([[[d["u"], d["v"]]]], np.float64), K, DIST, P=K).reshape(2)
    pc = np.array([(und[0]-K[0,2])*z/K[0,0], (und[1]-K[1,2])*z/K[1,1], z]) / 1000.0
    p, rpy = A.pose(arm)
    Tg = np.eye(4); Tg[:3,:3] = e2R(*np.radians(rpy)); Tg[:3,3] = p/1000.0
    pb = (Tg @ X @ np.append(pc, 1.0))[:3] * 1000.0
    return pb, f"px=({d['u']:.0f},{d['v']:.0f}) z={z:.0f} rect={d['rw']:.0f}x{d['rh']:.0f}"

def goto_xy(arm, xy, speed=5, max_step=20.0, tol=2.5, max_iter=8):
    for i in range(max_iter):
        p, rpy = A.pose(arm)
        d = np.array(xy, float) - p[:2]
        n = np.linalg.norm(d)
        if n < tol:
            print(f"[goto] 到位 残差 {n:.1f}mm")
            return True
        st = d * (min(1.0, max_step/n))
        ok, err = A.move_p(arm, [p[0]+st[0], p[1]+st[1], p[2], *rpy], speed=speed, timeout=12)
        p2, _ = A.pose(arm)
        moved = np.linalg.norm(p2[:2]-p[:2])
        print(f"[goto] {i} 目标 ({xy[0]:.0f},{xy[1]:.0f}) 我 ({p2[0]:.0f},{p2[1]:.0f}) 走了 {moved:.1f}mm 残差 {np.linalg.norm(np.array(xy,float)-p2[:2]):.1f}mm")
        if moved < 1.0:
            print("[goto] 控制器拒绝: 该方向到关节限位/臂展边缘")
            return False
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z-measure", type=float, default=190.0, help="测量的法兰高度(深度要有效)")
    ap.add_argument("--no-search", action="store_true")
    a = ap.parse_args()
    arm, eff = A.connect()
    A.gripper(eff, A.CFG["open_mm"], 1.0)
    A.goto_obs(arm)
    if not a.no_search and not A.search_box(arm):
        sys.exit("视野里找不到盒子")
    # 降到测量高度
    p, rpy = A.pose(arm)
    while p[2] > a.z_measure + 0.5:
        nz = max(a.z_measure, p[2] - 10.0)
        A.move_p(arm, [p[0], p[1], nz, *rpy], speed=6)
        p, rpy = A.pose(arm)
    print(f"[meas] 法兰 ({p[0]:.0f},{p[1]:.0f},{p[2]:.0f}) rpy ({rpy[0]:.0f},{rpy[1]:.0f},{rpy[2]:.0f})")
    pb = None
    for it in range(3):
        pb, info = box_base(arm, f"he{it}")
        if pb is None:
            sys.exit(f"测不到盒子: {info}")
        print(f"[he] 盒子基座 ({pb[0]:.0f},{pb[1]:.0f},{pb[2]:.0f}) 半径 {math.hypot(pb[0],pb[1]):.0f}mm | {info}")
        if not goto_xy(arm, pb[:2]):
            break
        p, rpy = A.pose(arm)
        if math.hypot(pb[0]-p[0], pb[1]-p[1]) < 3.0:
            break
    if not A.align_yaw(arm):
        print("[warn] 角度没收敛, 继续")
    pb, info = box_base(arm, "he_yaw")
    if pb is None: sys.exit(f"转完测不到: {info}")
    print(f"[he] 转完 盒子基座 ({pb[0]:.0f},{pb[1]:.0f}) | {info}")
    goto_xy(arm, pb[:2])
    # 下降
    p, rpy = A.pose(arm)
    cur = p[2]
    while cur > A.CFG["grasp_z"] + 0.5:
        cur = max(A.CFG["grasp_z"], cur - 10.0)
        ok, err = A.move_p(arm, [p[0], p[1], cur, *rpy], speed=A.CFG["speed"])
        q, _ = A.pose(arm)
        print(f"[down] z={cur:.0f} 到位 {q[2]:.1f} err={err}")
    w0, f0 = A.gripper(eff, A.CFG["close_mm"], A.CFG["close_n"])
    print(f"[grip] 宽={w0}mm 力={f0}N")
    if w0 < A.CFG["close_mm"] + 2 or abs(f0) < 0.5:
        print("[grip] 没夹到"); A.gripper(eff, A.CFG["open_mm"], 1.0); sys.exit(2)
    up = A.pose(arm)[0]; up[2] += A.CFG["lift_mm"]
    A.move_p(arm, [*up, *rpy], speed=A.CFG["speed"])
    w1, f1 = A.gripper(eff, A.CFG["close_mm"], A.CFG["close_n"], seconds=0.4)
    print(f"[lift] 宽={w1}mm 力={f1}N -> " + ("抓取成功" if abs(f1) > 0.5 else "掉了"))

if __name__ == "__main__":
    main()
